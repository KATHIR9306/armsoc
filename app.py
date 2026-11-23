"""
Streamlit Real-Time Industrial Defect Detection & Quality-Control Assistant Dashboard.
Follows the Bharat AI-SoC Challenge Reference Architecture (Slide 4) & Two-Pipeline Design (Slide 5).
"""

import streamlit as st
import numpy as np
import cv2
import json
import time
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.assistant import QualityControlAssistant

st.set_page_config(
    page_title="Bharat AI-SoC: Industrial QC Assistant",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize master QC assistant
if "qc_system" not in st.session_state:
    st.session_state.qc_system = QualityControlAssistant()

# Title and Challenge Header
st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>🏭 Bharat AI-SoC Challenge (2nd Edition 2026–27)</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #374151;'>Problem Statement 5: Real-Time Industrial Defect Detection & Quality-Control Assistant</h4>", unsafe_allow_html=True)
st.caption("Presented by: Charitha Balivada | Reference Architecture: Vision (Arm CPU/Hailo) + Advisory (Llama 3.2 via ExecuTorch)")

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/processor.png", width=64)
st.sidebar.header("⚙️ Target Hardware & Runtimes")

st.sidebar.markdown("### 1. Vision Pipeline Runtime")
vision_target = st.sidebar.selectbox(
    "Vision Engine",
    ["Raspberry Pi 5 (Arm Cortex-A76 CPU - ONNX/NEON)", "Hailo-8 / 8L NPU (via HailoRT)", "ExecuTorch Vision (MobileNet/YOLO)"],
    index=0
)

st.sidebar.markdown("### 2. Advisory Pipeline Runtime (Mandatory)")
llama_runtime = st.sidebar.selectbox(
    "Llama Engine",
    ["ExecuTorch Llama 3.2 1B Instruct (Arm CPU)", "ExecuTorch Llama 3.2 3B Instruct (Arm CPU)"],
    index=0
)

st.sidebar.markdown("### 3. Inspection Parameters")
inspection_domain = st.sidebar.selectbox("Inspection Domain", ["PCB Assembly (SMT Line)", "Steel Surface (NEU)", "MVTec Texture"])
conf_threshold = st.sidebar.slider("Vision Confidence Gate", 0.3, 0.95, 0.50, 0.05)
simulated_feed = st.sidebar.checkbox("Auto Conveyor Loop", value=False)

# Synthetic PCB Frame Generator for Live Simulation
def generate_pcb_frame(has_defect: bool = True):
    frame = np.full((480, 640, 3), (25, 65, 30), dtype=np.uint8) # Dark industrial PCB green
    # Draw traces
    for y in range(40, 460, 35):
        cv2.line(frame, (40, y), (600, y), (35, 175, 205), 2)
    for x in range(80, 580, 60):
        cv2.line(frame, (x, 40), (x, 440), (25, 140, 170), 1)
    # Draw IC packages
    cv2.rectangle(frame, (220, 160), (360, 300), (30, 30, 30), -1)
    cv2.putText(frame, "ARM CORTEX-A", (230, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    if has_defect:
        # Draw simulated solder bridge
        cv2.circle(frame, (380, 220), 16, (0, 0, 255), -1)
        cv2.line(frame, (360, 220), (410, 220), (0, 0, 255), 6)
    return frame

# Main Controls
col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 2])
with col_btn1:
    btn_inspect = st.button("🔍 Inspect Next Component Frame", use_container_width=True, type="primary")
with col_btn2:
    btn_clean_pass = st.button("✅ Simulate Golden Unit (Pass)", use_container_width=True)
with col_btn3:
    st.info("⚡ 100% Offline Edge Execution (Zero Cloud Dependency)")

# Handle Inspection Trigger
if btn_inspect or simulated_feed or btn_clean_pass:
    is_defect = not btn_clean_pass
    frame = generate_pcb_frame(has_defect=is_defect)
    result = st.session_state.qc_system.inspect_unit(frame)
    st.session_state["latest_result"] = result

# Display Results if Available
if "latest_result" in st.session_state:
    res = st.session_state["latest_result"]
    finding = res["finding"]
    advisory = res["advisory"]
    status = res["status"]

    # Top Status Bar
    if status == "PASS":
        st.success(f"🟢 **UNIT STATUS: PASS** | Unit ID: `{res['unit_id']}` | No Critical Defects Detected")
    elif status == "REJECT":
        st.error(f"🚨 **UNIT STATUS: REJECT** | Unit ID: `{res['unit_id']}` | Immediate Action: **{advisory['immediate_action']}**")
    else:
        st.warning(f"🟡 **UNIT STATUS: REWORK** | Unit ID: `{res['unit_id']}` | Action: **{advisory['immediate_action']}**")

    # Layout: Two Separate Pipelines (Slide 5)
    col_vision, col_advisory = st.columns([1, 1])

    # Left Column: Vision Pipeline
    with col_vision:
        st.markdown("### 👁️ Pipeline 1: Vision Model (Arm CPU / Hailo NPU)")
        st.image(cv2.cvtColor(res["annotated_frame"], cv2.COLOR_BGR2RGB), use_container_width=True)

        v_metrics = st.columns(4)
        v_metrics[0].metric("Vision Latency", f"{finding['total_vision_latency_ms']} ms")
        v_metrics[1].metric("Throughput", f"{finding['fps']} FPS")
        v_metrics[2].metric("Class", finding['defect_class'].upper())
        v_metrics[3].metric("Severity", finding['severity'])

        with st.expander("📄 View Structured JSON Hand-off (Vision -> Llama)", expanded=False):
            st.json(finding)

    # Right Column: Advisory Pipeline (Llama + ExecuTorch)
    with col_advisory:
        st.markdown("### 🧠 Pipeline 2: Llama + ExecuTorch (Advisory Reasoning)")
        
        # Advisory Card
        st.markdown(f"**Defect Explanation:**")
        st.write(advisory["explanation"])

        st.markdown(f"**Root Cause Hypothesis:**")
        st.info(f"🔬 {advisory['root_cause_hypothesis']}")

        st.markdown(f"**Operator Remedial Guidance:**")
        st.warning(f"🛠️ {advisory['operator_guidance']}")

        st.markdown(f"**Shift Log Entry:**")
        st.code(advisory["log_entry"], language="text")

        l_metrics = st.columns(3)
        l_metrics[0].metric("Time to 1st Token (TTFT)", f"{advisory['time_to_first_token_ms']} ms")
        l_metrics[1].metric("Generation Speed", f"{advisory['tokens_per_second']} tok/s")
        l_metrics[2].metric("Runtime", "ExecuTorch Arm")

# Bottom Section: Statistical Process Control & History
st.divider()
st.subheader("📊 Statistical Process Control (SPC) & Edge Inspection History")

spc = st.session_state.qc_system.get_spc_metrics()
spc_cols = st.columns(4)
spc_cols[0].metric("Total Inspected", spc["total_inspected"])
spc_cols[1].metric("Defective Units", spc["total_defects"])
spc_cols[2].metric("Line Yield", f"{spc['yield_percent']}%")
spc_cols[3].metric("Defect Rate", f"{spc['defect_rate_percent']}%")

recent_logs = st.session_state.qc_system.get_recent_logs(10)
if not recent_logs.empty:
    st.dataframe(
        recent_logs[[
            "unit_id", "timestamp", "defect_class", "severity", "confidence",
            "vision_latency_ms", "immediate_action", "log_entry"
        ]],
        use_container_width=True
    )

    csv_data = recent_logs.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Official Shift Inspection Log (CSV)",
        data=csv_data,
        file_name="industrial_qc_inspection_log.csv",
        mime="text/csv"
    )
