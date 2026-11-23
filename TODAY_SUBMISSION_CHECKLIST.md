# Phase 1 Submission Checklist (Nov 23, 2026)
**Hard Deadline**: Today, 23rd November, 2026 before midnight!  
**Target**: Arm Bharat AI-SoC Challenge (2nd Edition) | Problem Statement 5  

---

## ⏱️ Action Plan for Today

### Block 1: Verification & Local Testing (11:30 AM – 1:00 PM)
- [x] Python codebase verified and tested (`src/detector.py`, `src/executorch_llama_advisor.py`, `src/assistant.py`).
- [x] Run the interactive dashboard locally:
  ```bash
  streamlit run dashboard/app.py
  ```
  Verify that the two pipelines (Vision + Llama ExecuTorch) render cleanly in your browser (`http://localhost:8501`).
- [x] Run the benchmark test:
  ```bash
  python benchmarks/benchmark.py
  ```

---

### Block 2: Record & Upload Demo Video (1:00 PM – 2:30 PM)
- [ ] Open [docs/DEMO_VIDEO_GUIDE.md](file:///f:/projects/armsoc/docs/DEMO_VIDEO_GUIDE.md).
- [ ] Start screen recording software (OBS Studio or `Win + G`).
- [ ] Run through the 3-minute script demonstrating:
  1. Real-time vision detection & bounding box overlay.
  2. Structured JSON hand-off view.
  3. Llama + ExecuTorch advisory reasoning card.
  4. Shift inspection log & CSV export.
- [ ] Upload video:
  - **YouTube**: Upload as **Unlisted** (Title: *Arm Bharat AI-SoC Challenge 2026-27 | PS5 Demo*).
  - **Google Drive**: Verify share settings: **"Anyone with the link can view"**.
- [ ] Copy the shareable link.

---

### Block 3: Prepare GitHub Repository (2:30 PM – 3:30 PM)
- [ ] Initialize Git in `f:\projects\armsoc` (if not done yet):
  ```bash
  git init
  git add .
  git commit -m "Complete Phase 1 submission for Bharat AI-SoC Challenge PS5"
  ```
- [ ] Create a new public repository on GitHub (e.g. `bharat-ai-soc-ps5-qc-assistant`).
- [ ] Push local repository to GitHub:
  ```bash
  git remote add origin https://github.com/<your-username>/bharat-ai-soc-ps5-qc-assistant.git
  git branch -M main
  git push -u origin main
  ```
- [ ] Verify repository is **Public** and visible to judges.

---

### Block 4: Verify Technical Report PDF (3:30 PM – 4:30 PM)
- [ ] Check generated PDF: [docs/Phase1_Technical_Report.pdf](file:///f:/projects/armsoc/docs/Phase1_Technical_Report.pdf).
- [ ] Ensure all team members' names, institute, and mentor names are filled in.
- [ ] Re-compile if you edit any names:
  ```bash
  python build_phase1_report_pdf.py
  ```

---

### Block 5: Submit Challenge Form (Before 6:00 PM)
- [ ] Open the official submission form link sent by the organizers (or visit [armbharatchallenge.com](http://www.armbharatchallenge.com/)).
- [ ] Enter:
  - **Team Leader & Member details**
  - **Problem Statement**: PS5 - Real-time industrial defect detection & quality-control assistant
  - **GitHub Repository URL**
  - **Demo Video URL** (YouTube / Google Drive)
  - **Technical Report File / Link** (`docs/Phase1_Technical_Report.pdf`)
- [ ] Submit and save the confirmation email / screenshot!
