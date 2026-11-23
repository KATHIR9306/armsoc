# Phase 1 Technical Project Report
**ARM Bharat AI-SoC Challenge (2nd Edition 2026-2027)**  
**Problem Statement No. 5**: Real-Time Industrial Defect Detection & Quality-Control Assistant  
**Team Name**: [Insert Team Name]  
**Team Leader**: [Insert Name & Email]  
**Team Members**: [Insert Names]  
**Faculty / Industry Mentor**: [Insert Mentor Name]  
**Institution / Organization**: [Insert College/Org Name]  
**GitHub Repository Link**: [Insert GitHub Repo URL]  
**Demo Video Link**: [Insert YouTube Unlisted / Google Drive Link]  

---

## 1. Executive Summary
- Brief problem description: The need for automated, high-speed optical inspection in high-throughput manufacturing lines.
- Proposed solution summary: An edge AI-powered industrial defect detector coupled with an operator-assistive quality control system running on Arm-optimized architecture.
- Key quantitative outcomes (Target): Processing speed (>30 FPS), inference latency (<35 ms), detection accuracy (mAP@0.5 > 90%), model size (<15 MB via INT8 quantization).

---

## 2. Problem Statement Formulation & Industrial Use Case
- Selected manufacturing domain (e.g., PCB SMT assembly line / Metal sheet manufacturing).
- Types of defects detected (e.g., Missing component, short circuit, solder bridge, surface scratch, crack, pitting).
- Challenges of existing manual or cloud-based inspection:
  - Latency bottleneck of cloud APIs.
  - Bandwidth consumption and IP security risks of uploading factory video feeds.
  - Human operator fatigue leading to defect escape rates.

---

## 3. System Architecture & Hardware-Software Co-Design
- **End-to-End Architecture Diagram**:
  - Image Acquisition (Camera / Line-scan interface)
  - Preprocessing & Frame Normalization
  - Edge AI Inference Engine (Arm Cortex-A / Ethos-U / Arm NN)
  - Defect Post-Processing & Severity Scoring Engine
  - Quality-Control Assistant & Operator Dashboard
  - Industrial Actuator / Rejection Bagger Trigger Simulation

---

## 4. Edge AI Model & Optimization Strategy
- **Base Model Architecture Selection**: Justification of YOLOv8n / MobileNetV4 vs heavy models.
- **Model Training & Loss Function**: Dataset split, augmentation pipeline, training curves.
- **Quantization & Compression**:
  - Full Integer Quantization (INT8 PTQ).
  - Arm NEON / SIMD vectorization.
  - Model size comparison: FP32 (MB) vs INT8 (MB).

---

## 5. Quality-Control Assistant Subsystem
- **Severity Scoring Logic**: Matrix categorizing defect size, type, and location into `Critical`, `Major`, `Minor`.
- **Operator Advisory Mechanism**: Automated remedial actions (e.g., "Halt line: Reflow oven temperature drift suspected").
- **Statistical Quality Control (SQC)**: Batch defect rates, trend analysis, and shift summary generation.

---

## 6. Experimental Results & Benchmark Analysis
| Metric | Baseline (FP32) | Optimized (INT8 on Arm) | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Size** | ~12.5 MB | ~3.2 MB | 3.9x compression |
| **Inference Latency** | ~75 ms | ~22 ms | 3.4x faster |
| **Throughput (FPS)** | 13.3 FPS | 45.4 FPS | Line-rate capable |
| **mAP@0.5** | 92.4% | 91.6% | <0.8% loss |
| **Memory Footprint** | ~280 MB RAM | ~85 MB RAM | 70% reduction |

---

## 7. Demonstration & Repository Overview
- Walkthrough of the recorded demo video.
- Instructions to reproduce the pipeline from the GitHub repository.

---

## 8. Phase 2 Proof-of-Concept (PoC) Roadmap
- Deployment onto dedicated physical Arm SoC hardware / heterogeneous FPGA SoC (e.g., Arm Corstone, Xilinx Zynq-7000 / UltraScale+ MPSoC).
- Integration with Arm Ethos-U microNPU using CMSIS-NN / Vela compiler.
- Hardware in the Loop (HIL) conveyor belt actuator triggering.
