# Phase 1 Execution Roadmap (Nov 10 – Nov 23, 2026)
**Target**: Complete submission for Phase 1 Open Qualifying Round of the ARM Bharat AI-SoC Challenge.

---

## ⏱️ Critical Timeline Summary
- **Current Date**: November 10, 2026
- **Next Event**: Training & Doubt Clearing Session on **Friday, Nov 13, 2026 (4:00 PM – 5:00 PM IST)**
- **Final Deadline**: **Monday, Nov 23, 2026** (13 days from today)
- **Deliverables**:
  1. Public/Accessible GitHub Repository with full source code and setup instructions.
  2. 3–5 minute Demonstration Video (walkthrough of real-time detection, QC assistant, and latency/FPS).
  3. Comprehensive Technical Report (PDF) covering problem context, architecture, model selection, quantization/SoC hardware strategy, benchmarks, and Phase 2 roadmap.

---

## 🗓️ Phase-by-Phase Day-by-Day Plan

### Phase 1A: Setup & Preparation (Nov 10 – Nov 12)
- [ ] **Watch Mentoring Video**: Watch the PS5 mentoring video link ([Google Drive Link](https://drive.google.com/file/d/1RKUbEBscquEvgs3_VfJZ3F_RH6HSlEbT/view?usp=sharing)) and note any specific dataset/framework guidelines stated by Arm mentors.
- [ ] **Select Target Industrial Domain & Dataset**:
  - *Option A (Recommended - High Relevance to Electronics & SoC)*: **PCB Defect Detection** (e.g., DeepPCB or PKU-Market-PCB dataset: 6 defect types: pinhole, scratch, open, short, mousebite, spur).
  - *Option B*: **Surface Defect Detection** (NEU Surface Defect Database: rolled-in scale, patches, crazing, pitted surface, inclusion, scratches).
  - *Option C*: **MVTec Anomaly Detection (AD)** for unsupervised anomaly detection.
- [ ] **Define Architecture & Arm Deployment Target**:
  - Arm Cortex-A target (e.g., Raspberry Pi 4/5, i.MX8M, or Arm Virtual Hardware AVH Cortex-A) using Arm NN or ONNX Runtime with Arm NEON acceleration.
  - Model family: Ultralytics YOLOv8-nano / YOLO11-nano or lightweight MobileNetV4-SSD quantized to INT8.
- [ ] **Formulate Technical Questions** for the Nov 13 training session (see `docs/MENTOR_QUESTIONS.md`).

---

### Phase 1B: Training Session & Alignment (Nov 13)
- [ ] **Attend Zoom Training Session (4:00 PM – 5:00 PM IST)**:
  - Join via: `https://armltd.zoom.us/j/92822528862?pwd=62Dx9B1nPvibrWquzVZOTg9EESj1tE.1&from=addon`
  - Meeting ID: `928 2252 8862` | Passcode: `572824`
- [ ] Clarify:
  - Is deployment on physical Arm hardware mandatory for Phase 1, or is Arm Virtual Hardware (AVH) / host emulation with Arm profiling accepted?
  - Are there preferred AI runtimes (Arm NN, CMSIS-NN, ExecuTorch, TFLite)?
  - Exact evaluation criteria / rubrics for Phase 1.
- [ ] Incorporate mentor feedback immediately into the implementation plan.

---

### Phase 1C: Core Model & Pipeline Implementation (Nov 14 – Nov 17)
- [ ] **Train / Fine-tune Lightweight Vision Model**:
  - Train YOLOv8n / YOLO11n on the chosen defect dataset with data augmentation.
  - Track mAP@0.5, Precision, Recall, and confusion matrix across defect classes.
- [ ] **Quantization & Edge Optimization**:
  - Quantize model from FP32 to INT8 (Post-Training Quantization / PTQ) using ONNX / TFLite.
  - Measure accuracy retention (ensure < 1.5% drop in mAP).
  - Export to Arm-friendly format (`.onnx`, `.tflite`, or Arm NN `.armnn`).
- [ ] **Build the QC Assistant Subsystem**:
  - Implement rule-based and heuristics severity classifier (Critical, Moderate, Low).
  - Add real-time defect telemetry (defect counter per batch, cycle time, pass/fail rate).
  - Add operator recommendation engine (actionable mitigation steps for operator).

---

### Phase 1D: Dashboard & Operator UI (Nov 18 – Nov 19)
- [ ] **Interactive Quality-Control Dashboard**:
  - Build a responsive Streamlit / Gradio / Web UI.
  - Features:
    - Live video feed / simulated conveyor belt camera feed.
    - Real-time bounding box & defect tag overlay.
    - Latency (ms) and FPS ticker.
    - Defect breakdown chart (Pareto analysis).
    - Audio/visual buzzer alert on "Critical" defect.
    - Exportable CSV/PDF inspection report for the factory shift supervisor.

---

### Phase 1E: Benchmarking, Demo Recording & Report (Nov 20 – Nov 22)
- [ ] **Benchmarking**:
  - Measure FP32 vs INT8 latency, FPS, model size reduction (MB), and CPU utilization.
  - Generate comparison graphs for the report.
- [ ] **Record Demo Video (3–5 minutes)**:
  - Slide 1: Problem statement & Arm SoC relevance (30 sec).
  - Slide 2: System Architecture & Hardware acceleration flow (45 sec).
  - Live Demo: Video feed processing, real-time defect localization, QC assistant flagging and reporting (2 mins).
  - Slide 3: Performance metrics (FPS, Latency, Accuracy) and Phase 2 roadmap (45 sec).
  - Upload to YouTube (Unlisted) or Google Drive (public view permission).
- [ ] **Technical Report Preparation**:
  - Complete the Phase 1 Report using the standard template in `docs/Phase1_Report.md`.
  - Export as a high-quality PDF.

---

### Phase 1F: Review & Final Submission (Nov 23)
- [ ] Verify GitHub repository is public, clean, has a comprehensive `README.md`, setup instructions, and clean code comments.
- [ ] Double-check demo video link permissions (anyone with link can view).
- [ ] Complete the submission form on `www.armbharatchallenge.com` well before midnight.
