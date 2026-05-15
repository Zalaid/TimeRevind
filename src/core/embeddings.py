"""
Embeddings Manager
Handles InsightFace ArcFace face embeddings (via ONNX + DirectML/CUDA)
"""

import os
import time
import logging
import numpy as np
import cv2
import torch
import threading
from numpy.linalg import norm
from src.config import FACE_CONFIDENCE_THRESHOLD, INSIGHTFACE_MODEL, SCRFD_MODEL, SCRFD_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), 'lib'))


try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    logger.warning("onnxruntime not available - InsightFace ArcFace disabled")




# ArcFace canonical 112×112 landmark positions (standard InsightFace template)
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _align_face(img_bgr, kps):
    """Warp face to canonical 112×112 using 5 SCRFD keypoints."""
    M, _ = cv2.estimateAffinePartial2D(kps, _ARCFACE_DST, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img_bgr, M, (112, 112), flags=cv2.INTER_LINEAR)


class SCRFDDetector:
    """SCRFD-10G face detector via ONNX + DirectML.
    Detects small/distant faces significantly better than standard face detectors.
    Outputs: boxes (xyxy), confidence scores, 5 facial keypoints.
    """

    STRIDES = [8, 16, 32]
    NUM_ANCHORS = 2  # SCRFD uses 2 anchors per feature map location

    def __init__(self, model_path, providers, input_size=(640, 640)):
        self.session    = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_w, self.input_h = input_size
        self._anchor_centers = self._build_anchor_centers()

    def _build_anchor_centers(self):
        """Pre-compute anchor centers for all strides (row-major, 2 anchors per location)."""
        centers = []
        for stride in self.STRIDES:
            feat_h = self.input_h // stride
            feat_w = self.input_w // stride
            col_grid, row_grid = np.meshgrid(
                np.arange(feat_w, dtype=np.float32),
                np.arange(feat_h, dtype=np.float32)
            )
            cx = col_grid.flatten() * stride + stride / 2
            cy = row_grid.flatten() * stride + stride / 2
            level = np.stack([cx, cy], axis=1)              # (feat_h*feat_w, 2)
            level = np.repeat(level, self.NUM_ANCHORS, axis=0)  # (feat_h*feat_w*2, 2)
            centers.append(level)
        return np.concatenate(centers, axis=0)  # (16800, 2)

    def detect(self, img_bgr, conf_threshold=0.5, min_face_size=20):
        """Detect faces in a BGR image.

        Returns:
            boxes  : (N, 4) float32 xyxy in original image coords
            scores : (N,)   float32 confidence
            kps    : (N, 5, 2) float32 keypoints in original image coords
                     order: left_eye, right_eye, nose, left_mouth, right_mouth
        """
        h, w = img_bgr.shape[:2]

        # Letterbox resize to input_size
        scale = min(self.input_w / w, self.input_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h))

        blob = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        blob[:new_h, :new_w] = resized

        # Normalize: BGR→RGB, (pixel − 127.5) / 128.0
        blob = blob[:, :, ::-1].astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, :]  # (1, 3, H, W)

        outputs = self.session.run(None, {self.input_name: blob})
        # outputs layout (9 tensors): scores×3, boxes×3, kps×3 (one per stride)
        scores_raw_list = outputs[0:3]
        boxes_raw_list  = outputs[3:6]
        kps_raw_list    = outputs[6:9]

        all_boxes, all_scores, all_kps = [], [], []
        anchor_offset = 0

        for i, stride in enumerate(self.STRIDES):
            n = scores_raw_list[i].shape[0]
            scores = scores_raw_list[i].flatten()  # model applies sigmoid internally

            mask = scores >= conf_threshold
            if not mask.any():
                anchor_offset += n
                continue

            ac = self._anchor_centers[anchor_offset:anchor_offset + n]  # (n, 2)

            # Decode boxes: raw distances × stride, then offset from anchor center
            dist = boxes_raw_list[i] * stride    # (n, 4)
            boxes = np.stack([
                ac[:, 0] - dist[:, 0],  # x1
                ac[:, 1] - dist[:, 1],  # y1
                ac[:, 0] + dist[:, 2],  # x2
                ac[:, 1] + dist[:, 3],  # y2
            ], axis=1)

            # Decode keypoints: raw offsets × stride, added to anchor center
            kraw = kps_raw_list[i] * stride  # (n, 10)
            kps = np.stack([
                ac[:, 0:1] + kraw[:, 0::2],  # x coords of 5 kps  (n, 5)
                ac[:, 1:2] + kraw[:, 1::2],  # y coords of 5 kps  (n, 5)
            ], axis=2)  # (n, 5, 2)

            all_boxes.append(boxes[mask])
            all_scores.append(scores[mask])
            all_kps.append(kps[mask])
            anchor_offset += n

        if not all_boxes:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0, 5, 2))

        boxes  = np.concatenate(all_boxes,  axis=0)
        scores = np.concatenate(all_scores, axis=0)
        kps    = np.concatenate(all_kps,    axis=0)

        # NMS
        keep = self._nms(boxes, scores)
        boxes, scores, kps = boxes[keep], scores[keep], kps[keep]

        # Filter by minimum face size
        widths  = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        size_mask = (widths >= min_face_size) & (heights >= min_face_size)
        boxes, scores, kps = boxes[size_mask], scores[size_mask], kps[size_mask]

        # Scale back to original image coords
        boxes /= scale
        kps   /= scale

        return boxes, scores, kps

    @staticmethod
    def _nms(boxes, scores, iou_threshold=0.4):
        """Non-maximum suppression."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, (x2 - x1)) * np.maximum(0.0, (y2 - y1))
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            order = order[1:][iou <= iou_threshold]
        return keep


class EmbeddingManager:
    """Manages computation of face embeddings"""

    def __init__(self):
        self.scrfd_detector = None          # SCRFD-10G face detector
        self.face_embedding_session = None  # InsightFace ArcFace ONNX session
        self.face_embedding_input_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._initialize_models()

    def _initialize_models(self):
        """Initialize embedding models"""
        logger.info("=" * 55)
        logger.info("🖥️  DEVICE SUMMARY")
        logger.info(f"   PyTorch device : {'GPU ✅ ' + torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU ⚠️  (no CUDA)'}")
        logger.info(f"   SCRFD-10G (face detect) : GPU via CUDA ✅")
        logger.info(f"   ArcFace R50 (face embed): GPU via CUDA ✅")
        logger.info("=" * 55)

        # SCRFD-10G face detector via ONNX + DirectML (detects small/distant faces)
        if ONNXRUNTIME_AVAILABLE:
            try:
                if os.path.exists(SCRFD_MODEL):
                    self.scrfd_detector = SCRFDDetector(
                        model_path=SCRFD_MODEL,
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                        input_size=(640, 640)
                    )
                    active = self.scrfd_detector.session.get_providers()
                    label = "GPU via CUDA ✅" if "CUDA" in active[0] else "CPU ⚠️"
                    logger.info(f"SCRFD-10G face detector loaded ({label})")
                    input_info = self.scrfd_detector.session.get_inputs()[0]
                    logger.info(f"🔍 SCRFD input shape: {input_info.shape}")
                    logger.info(f"🔍 SCRFD active provider: {active[0]}")
                else:
                    logger.error(f"SCRFD model not found at {SCRFD_MODEL}")
            except Exception as e:
                logger.error(f"Failed to load SCRFD-10G: {e}")
                self.scrfd_detector = None

        # InsightFace ArcFace ResNet50 via ONNX + CUDA
        if ONNXRUNTIME_AVAILABLE:
            try:
                if os.path.exists(INSIGHTFACE_MODEL):
                    _arcface_opts = ort.SessionOptions()
                    _arcface_opts.log_severity_level = 3  # suppress shape-mismatch warnings (static {1,512} vs batch {N,512})
                    self.face_embedding_session = ort.InferenceSession(
                        INSIGHTFACE_MODEL,
                        sess_options=_arcface_opts,
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                    )
                    self.face_embedding_input_name = self.face_embedding_session.get_inputs()[0].name
                    active_provider = self.face_embedding_session.get_providers()[0]
                    provider_label = "GPU via CUDA ✅" if "CUDA" in active_provider else "CPU ⚠️"
                    logger.info(f"InsightFace ArcFace R50 loaded ({provider_label})")
                else:
                    logger.error(f"InsightFace ArcFace model not found at {INSIGHTFACE_MODEL}")
            except Exception as e:
                logger.error(f"Failed to load InsightFace ONNX model: {e}")
                self.face_embedding_session = None

        # Warmup: trigger CUDA kernel compilation now so first real inference is instant
        self._warmup_models()

    def _warmup_models(self):
        """Run dummy inference to compile CUDA kernels at startup."""
        logger.info("⏳ Warming up CUDA kernels (compiles once, fast after)...")
        try:
            if self.scrfd_detector is not None:
                dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                self.scrfd_detector.detect(dummy_img, conf_threshold=0.9)
        except Exception:
            pass
        try:
            if self.face_embedding_session is not None:
                dummy_face = np.zeros((1, 3, 112, 112), dtype=np.float32)
                self.face_embedding_session.run(None, {self.face_embedding_input_name: dummy_face})
        except Exception:
            pass
        logger.info("✅ CUDA warmup complete — inference ready")

    def compute_face_embeddings_batch(self, frames):
        """Process all frames sequentially through SCRFD + one batched ArcFace call.

        Returns:
            embeddings  : list[np.ndarray | None]  — 512D unit vector or None per frame
            confidences : list[float]              — SCRFD score or 0.0 per frame
            kps_list    : list[np.ndarray | None]  — (5,2) keypoints or None per frame
        """
        n = len(frames)
        embeddings  = [None] * n
        confidences = [0.0]  * n
        kps_list    = [None] * n

        if not frames or self.scrfd_detector is None or self.face_embedding_session is None:
            return embeddings, confidences, kps_list

        face_tensors = []
        valid_idx    = []

        valid_frames = [(i, f) for i, f in enumerate(frames)
                        if f is not None and isinstance(f, np.ndarray) and f.size > 0]

        t0 = time.time()
        for (orig_idx, frame) in valid_frames:
            try:
                # Run detect() individually — batch mode is broken for this model
                boxes, scores, kps = self.scrfd_detector.detect(
                    frame,
                    conf_threshold=SCRFD_CONFIDENCE_THRESHOLD,
                    min_face_size=20
                )
                if len(boxes) == 0:
                    continue

                best_idx = int(np.argmax(scores))
                conf     = float(scores[best_idx])

                # Store kps for pose estimation upstream
                kps_list[orig_idx] = kps[best_idx]

                # Landmark-aligned crop (fixes rotated/angled faces for ArcFace)
                face = _align_face(frame, kps[best_idx])
                if face is None:
                    # Fallback to plain bbox crop if alignment fails
                    x1, y1, x2, y2 = [int(v) for v in boxes[best_idx]]
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                    face = cv2.resize(face_crop, (112, 112))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
                face = (face / 127.5) - 1.0
                face_tensors.append(face.transpose(2, 0, 1))
                valid_idx.append(orig_idx)
                confidences[orig_idx] = conf

            except Exception as e:
                logger.debug(f"SCRFD single detect error on frame {orig_idx}: {e}")

        scrfd_time = (time.time() - t0) * 1000
        logger.info(f"⏱ SCRFD sequential: {scrfd_time:.0f}ms for {len(valid_frames)} frames ({len(face_tensors)} faces found)")

        if not face_tensors:
            return embeddings, confidences, kps_list

        # ArcFace batch inference (this model DOES support batching correctly)
        t1 = time.time()
        try:
            batch = np.stack(face_tensors, axis=0).astype(np.float32)
            batch_embs = self.face_embedding_session.run(
                None, {self.face_embedding_input_name: batch}
            )[0]
        except Exception:
            logger.debug("Batch ArcFace failed — running sequentially")
            seq_embs = []
            for tensor in face_tensors:
                emb = self.face_embedding_session.run(
                    None, {self.face_embedding_input_name: tensor[np.newaxis, :]}
                )[0][0]
                seq_embs.append(emb)
            batch_embs = np.stack(seq_embs, axis=0)
        arcface_time = (time.time() - t1) * 1000
        logger.info(f"⏱ ArcFace phase: {arcface_time:.0f}ms for {len(face_tensors)} faces")

        try:
            norms = np.linalg.norm(batch_embs, axis=1, keepdims=True)
            batch_embs = batch_embs / (norms + 1e-6)
            for j, frame_idx in enumerate(valid_idx):
                embeddings[frame_idx] = batch_embs[j]
        except Exception as e:
            logger.debug(f"ArcFace post-processing error: {e}")

        return embeddings, confidences, kps_list

    def compute_face_embedding(self, person_crop, pick_center_face=False):
        try:
            if person_crop is None or person_crop.size == 0:
                return None, None, None, None

            if self.scrfd_detector is None or self.face_embedding_session is None:
                return None, None, None, None

            # SCRFD-10G face detection (handles small/distant faces)
            boxes, scores, kps_all = self.scrfd_detector.detect(
                person_crop,
                conf_threshold=SCRFD_CONFIDENCE_THRESHOLD,  # 0.5 — SCRFD standard threshold
                min_face_size=20  # 20px minimum — catches far-away faces
            )

            if len(boxes) == 0:
                return None, None, None, None

            # Pick face by strategy: center-most (for overlapping boxes) or highest confidence
            if pick_center_face and len(boxes) > 1:
                # Pick face closest to center of crop (avoids wrong faces in overlapping boxes)
                crop_h, crop_w = person_crop.shape[:2]
                center_x, center_y = crop_w / 2, crop_h / 2
                distances = []
                for box in boxes:
                    face_cx = (box[0] + box[2]) / 2
                    face_cy = (box[1] + box[3]) / 2
                    dist = np.sqrt((face_cx - center_x)**2 + (face_cy - center_y)**2)
                    distances.append(dist)
                best_idx = int(np.argmin(distances))
                logger.debug(f"[CENTER-FACE] Picked face {best_idx} (center distance: {distances[best_idx]:.1f})")
            else:
                # Pick highest-confidence detection
                best_idx = int(np.argmax(scores))

            best_conf = float(scores[best_idx])
            best_kps  = kps_all[best_idx]   # (5, 2) — used for pose estimation upstream
            x1, y1, x2, y2 = [int(v) for v in boxes[best_idx]]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(person_crop.shape[1], x2)
            y2 = min(person_crop.shape[0], y2)

            # Landmark-aligned crop — same as compute_face_embeddings_batch
            face_resized = _align_face(person_crop, best_kps)
            if face_resized is None:
                # Fallback to plain bbox crop if alignment fails
                face_crop = person_crop[y1:y2, x1:x2]
                if face_crop.size == 0:
                    return None, None, None, None
                face_resized = cv2.resize(face_crop, (112, 112))

            # InsightFace ArcFace R50 via ONNX + DirectML
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
            face_rgb = (face_rgb / 127.5) - 1.0  # InsightFace normalization
            face_tensor = face_rgb.transpose(2, 0, 1)[np.newaxis, :]  # (1, 3, 112, 112)
            embedding = self.face_embedding_session.run(
                None, {self.face_embedding_input_name: face_tensor}
            )[0][0]
            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

            # Return bbox as (x1, y1, x2, y2) for sharpness assessment + kps for pose
            bbox = (x1, y1, x2, y2)
            return embedding, best_conf, bbox, best_kps

        except Exception as e:
            logger.debug(f"Face embedding error: {e}")
            return None, None, None, None

    def embedding_l2_distance(self, embedding1, embedding2):
        """
        Compute L2 (Euclidean) distance between two embeddings
        Used for ArcFace embeddings (lower = better match)

        Returns:
            Distance value (0 = perfect match, higher = worse match)
        """
        if embedding1 is None or embedding2 is None:
            return float('inf')

        embedding1 = np.array(embedding1).flatten()
        embedding2 = np.array(embedding2).flatten()

        return norm(embedding1 - embedding2)

    def find_best_embedding_match(self, query_embedding, gallery_embeddings, threshold=1.24):
        """
        Find best matching embedding from a gallery of embeddings
        Matches against multiple embeddings (far away, close up, different angles, etc.)

        Args:
            query_embedding: Single embedding to match
            gallery_embeddings: List of embeddings to search through
            threshold: L2 distance threshold (default 1.24 for ArcFace)

        Returns:
            Tuple of (best_distance, confidence_percent) or (None, 0.0) if no match
            best_distance: L2 distance to best match (lower is better)
            confidence_percent: Confidence percentage (100 - distance_percent)
        """
        if not gallery_embeddings or query_embedding is None:
            return None, 0.0

        best_distance = float('inf')
        for gallery_emb in gallery_embeddings:
            if gallery_emb is not None:
                dist = self.embedding_l2_distance(query_embedding, gallery_emb)
                if dist < best_distance:
                    best_distance = dist

        if best_distance <= threshold:
            confidence = max(0.0, (1 - best_distance / threshold) * 100)
            return best_distance, confidence
        else:
            return best_distance, 0.0


# Global instance
embedding_manager = None
_embedding_manager_lock = threading.Lock()


def get_embedding_manager():
    """Get or create global embedding manager (thread-safe)"""
    global embedding_manager
    with _embedding_manager_lock:
        if embedding_manager is None:
            embedding_manager = EmbeddingManager()
    return embedding_manager


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = EmbeddingManager()
    print("Embedding manager initialized")

