"""
Vision Pipeline Engine for Real-Time Industrial Defect Detection.
Runs on Raspberry Pi 5 Arm CPU (via ONNX/TFLite/ExecuTorch) or Hailo NPU.
Outputs structured JSON findings with rule-grounded severity scoring.
"""

import time
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

@dataclass
class StructuredDefectFinding:
    unit_id: str
    timestamp: str
    defect_detected: bool
    defect_class: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL (Computed from type, size, location; not confidence)
    confidence: float
    bbox: List[int] # [x1, y1, x2, y2]
    defect_area_px: int
    location: str
    pre_process_ms: float
    inference_ms: float
    post_process_ms: float
    total_vision_latency_ms: float
    fps: float

class IndustrialVisionDetector:
    """
    Real-Time Edge Defect Detector.
    Supports ONNX Runtime with Arm NEON optimizations or simulated edge execution.
    """

    CRITICAL_DEFECT_CLASSES = {"short_circuit", "missing_component", "open_circuit"}
    MODERATE_DEFECT_CLASSES = {"solder_bridge", "mousebite"}
    MINOR_DEFECT_CLASSES = {"scratch", "pinhole", "spur"}

    def __init__(self, model_path: str = None, conf_threshold: float = 0.45, iou_threshold: float = 0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = [
            "solder_bridge",
            "short_circuit",
            "missing_component",
            "scratch",
            "pinhole",
            "mousebite"
        ]
        self.runtime_name = "Arm CPU (ONNX Runtime / NEON Vectorized)"
        self.model = None
        self._init_runtime()

    def _init_runtime(self):
        """Initializes model engine with CPU/Arm SIMD execution."""
        if self.model_path and self.model_path.endswith(".onnx"):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.intra_op_num_threads = 4
                self.model = ort.InferenceSession(self.model_path, opts, providers=['CPUExecutionProvider'])
                self.runtime_name = f"ONNX Runtime ONNX-INT8 ({self.model_path})"
            except Exception as e:
                print(f"[VisionDetector] Fallback to simulated edge runtime: {e}")
                self.model = None

    def compute_severity(self, defect_class: str, area_px: int, bbox: List[int], frame_shape: Tuple[int, ...]) -> str:
        """
        Determines severity (LOW, MEDIUM, HIGH, CRITICAL) from type, size, and location.
        Per Slide 10: 'Set Low, Medium, High from type, size and location... Confidence is not severity.'
        """
        c_name = defect_class.lower().replace(" ", "_")
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox

        # Check location: center region (active component area) vs periphery
        is_central_critical_zone = (0.2 * w <= (x1 + x2) / 2 <= 0.8 * w) and (0.2 * h <= (y1 + y2) / 2 <= 0.8 * h)

        if c_name in self.CRITICAL_DEFECT_CLASSES:
            return "CRITICAL"
        elif c_name in self.MODERATE_DEFECT_CLASSES:
            if area_px > 3000 or is_central_critical_zone:
                return "HIGH"
            return "MEDIUM"
        else:
            # Minor defect types (scratch, pinhole)
            if area_px > 5000:
                return "MEDIUM"
            return "LOW"

    def get_location_description(self, bbox: List[int], frame_shape: Tuple[int, ...]) -> str:
        """Provides human-readable semantic location for Llama reasoning."""
        h, w = frame_shape[:2]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        col = "Left (Col A)" if cx < w / 3 else ("Center (Col B)" if cx < 2 * w / 3 else "Right (Col C)")
        row = "Top (Row 1)" if cy < h / 3 else ("Middle (Row 2)" if cy < 2 * h / 3 else "Bottom (Row 3)")
        return f"{row}, {col} [x={int(cx)}, y={int(cy)}]"

    def process_frame(self, frame: np.ndarray, unit_id: str) -> Tuple[StructuredDefectFinding, np.ndarray]:
        """
        Runs full visual inspection pipeline:
        1. Pre-process (OpenCV)
        2. Vision Model (Detect & Classify)
        3. Structured Defect Finding generation (JSON hand-off)
        """
        # Step 1: Pre-process
        t0 = time.perf_counter()
        resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        t1 = time.perf_counter()
        pre_ms = (t1 - t0) * 1000.0

        # Step 2: Inference (Runs locally on Arm CPU or Hailo NPU)
        if self.model is not None:
            input_blob = np.transpose(normalized, (2, 0, 1))[np.newaxis, :]
            outputs = self.model.run(None, {self.model.get_inputs()[0].name: input_blob})
            t2 = time.perf_counter()
            infer_ms = (t2 - t1) * 1000.0
            # Post-process
            detections = self._parse_detections(outputs, frame.shape)
        else:
            # Simulated edge inference on Arm Cortex-A76 (15-22 ms for quantized tiny model)
            time.sleep(0.016)
            t2 = time.perf_counter()
            infer_ms = (t2 - t1) * 1000.0
            detections = self._simulate_edge_detection(frame.shape)

        t3 = time.perf_counter()
        post_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0

        # Step 3: Package into Structured JSON Finding
        timestamp_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        annotated_frame = frame.copy()

        if detections:
            det = detections[0]
            bbox = det["bbox"]
            area_px = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            severity = self.compute_severity(det["class_name"], area_px, bbox, frame.shape)
            loc_str = self.get_location_description(bbox, frame.shape)

            finding = StructuredDefectFinding(
                unit_id=unit_id,
                timestamp=timestamp_str,
                defect_detected=True,
                defect_class=det["class_name"],
                severity=severity,
                confidence=det["confidence"],
                bbox=bbox,
                defect_area_px=area_px,
                location=loc_str,
                pre_process_ms=round(pre_ms, 2),
                inference_ms=round(infer_ms, 2),
                post_process_ms=round(post_ms, 2),
                total_vision_latency_ms=round(total_ms, 2),
                fps=round(fps, 1)
            )

            # Draw visual bounding box & metrics overlay
            color = (0, 0, 255) if severity in ["HIGH", "CRITICAL"] else (0, 165, 255)
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(annotated_frame, f"{det['class_name'].upper()} [{severity}]",
                        (bbox[0], max(bbox[1] - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        else:
            finding = StructuredDefectFinding(
                unit_id=unit_id,
                timestamp=timestamp_str,
                defect_detected=False,
                defect_class="none",
                severity="NONE",
                confidence=0.99,
                bbox=[0, 0, 0, 0],
                defect_area_px=0,
                location="Surface Clear",
                pre_process_ms=round(pre_ms, 2),
                inference_ms=round(infer_ms, 2),
                post_process_ms=round(post_ms, 2),
                total_vision_latency_ms=round(total_ms, 2),
                fps=round(fps, 1)
            )

        # Draw HUD banner
        hud_text = f"Vision FPS: {fps:.1f} | Latency: {total_ms:.1f}ms (Pre:{pre_ms:.1f} Inf:{infer_ms:.1f} Post:{post_ms:.1f})"
        cv2.putText(annotated_frame, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        return finding, annotated_frame

    def _simulate_edge_detection(self, shape: Tuple[int, ...]) -> List[Dict[str, Any]]:
        """Simulates realistic industrial defect detections."""
        h, w = shape[:2]
        defect_class = np.random.choice(self.classes)
        x1 = int(np.random.uniform(0.15, 0.65) * w)
        y1 = int(np.random.uniform(0.15, 0.65) * h)
        bw = int(np.random.uniform(45, 110))
        bh = int(np.random.uniform(45, 110))
        return [{
            "class_name": defect_class,
            "confidence": float(np.random.uniform(0.82, 0.98)),
            "bbox": [x1, y1, min(x1 + bw, w - 5), min(y1 + bh, h - 5)]
        }]
