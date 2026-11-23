# Technical Questions for the Mentoring & Doubt Clearing Session
**Date**: Friday, November 13, 2026 | **Time**: 4:00 PM – 5:00 PM IST  
**Meeting Link**: [Join Zoom](https://armltd.zoom.us/j/92822528862?pwd=62Dx9B1nPvibrWquzVZOTg9EESj1tE.1&from=addon) (ID: 928 2252 8862, Passcode: 572824)  
**Problem Statement**: PS5 - Real-time industrial defect detection & quality-control assistant  

---

### Category 1: Target Hardware & SoC Deployment Environment
1. **Target Hardware for Phase 1 vs Phase 2**:
   - For the Phase 1 Qualifying submission, is demonstration on a physical Arm board (such as Raspberry Pi 4/5, NXP i.MX8, or NVIDIA Jetson with Arm cores) mandatory, or is evaluation on an Arm emulator / Arm Virtual Hardware (AVH) / host machine with simulated Arm profiling acceptable?
2. **Arm Toolchains and Runtimes**:
   - Do the judges have a preference between **Arm NN**, **CMSIS-NN**, **ExecuTorch**, or **ONNX Runtime (with Arm NEON SIMD)**?
   - Will access or credits to Arm Virtual Hardware (AVH) cloud instances be provided for the shortlisted teams in Phase 2?

---

### Category 2: Problem Scope & Quality-Control Assistant Expectations
3. **Domain & Dataset Flexibility**:
   - Is there a specific industrial defect dataset mandated by the challenge (e.g., PCB inspection, metal surface NEU, or MVTec AD), or are teams free to choose a representative industrial defect dataset as long as we demonstrate end-to-end edge inference?
4. **Definition of "Quality-Control Assistant"**:
   - What are the core expected capabilities of the "assistant" aspect? Should it focus primarily on real-time operator alerts, defect severity categorization, and statistical defect logging (SPC/Pareto), or are LLM/VLM-based conversational inspection assistants (e.g., edge-quantized SLMs) also encouraged if latency allows?

---

### Category 3: Evaluation Rubrics & Metrics
5. **Key Benchmark Priorities**:
   - In the evaluation rubric, how are weights distributed between:
     - Model accuracy (mAP / F1-score),
     - Edge throughput/latency (FPS, latency < 30ms),
     - Hardware efficiency (quantization INT8, memory footprint, power consumption),
     - Operator UI and usability of the assistant?
6. **Submission Format**:
   - What are the specific length/format requirements for the Phase 1 Technical Report (e.g., page limit, IEEE format, or standard proposal format)?
