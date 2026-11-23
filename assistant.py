"""
Quality-Control (QC) Assistant Master Orchestrator.
Connects Vision Pipeline -> Structured JSON Hand-off -> Llama + ExecuTorch Advisory Pipeline.
Maintains persistent inspection logs and Statistical Process Control (SPC) metrics.
"""

import os
import json
import sqlite3
import pandas as pd
from typing import Dict, Any, List, Tuple
from dataclasses import asdict

from src.detector import IndustrialVisionDetector, StructuredDefectFinding
from src.executorch_llama_advisor import ExecuTorchLlamaAdvisor, AdvisoryOutput

class QualityControlAssistant:
    """
    Main QC System orchestrating both required pipelines (Vision + Llama ExecuTorch).
    """

    def __init__(self, db_path: str = "data/inspection_history.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

        self.vision_detector = IndustrialVisionDetector()
        self.llama_advisor = ExecuTorchLlamaAdvisor()

        self.inspection_count = 0
        self.defect_count = 0

    def _init_db(self):
        """Initializes SQLite database for structured local logs (Slide 4 Step 6)."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS inspection_log (
                unit_id TEXT PRIMARY KEY,
                timestamp TEXT,
                defect_detected INTEGER,
                defect_class TEXT,
                severity TEXT,
                confidence REAL,
                location TEXT,
                vision_latency_ms REAL,
                vision_fps REAL,
                immediate_action TEXT,
                explanation TEXT,
                operator_guidance TEXT,
                log_entry TEXT,
                llama_ttft_ms REAL,
                llama_tokens_per_sec REAL,
                llama_runtime TEXT
            )
        """)
        conn.commit()
        conn.close()

    def inspect_unit(self, frame, unit_id: str = None) -> Dict[str, Any]:
        """
        Executes end-to-end industrial inspection workflow:
        1. Camera frame processed by Vision Detector -> Structured JSON finding
        2. Structured JSON passed to Llama + ExecuTorch -> Advisory reasoning
        3. Record persisted to local SQLite inspection log
        """
        self.inspection_count += 1
        if not unit_id:
            unit_id = f"UNIT-{self.inspection_count:05d}"

        # Step 1: Vision Pipeline (Slide 4 Steps 1, 2, 3, 4)
        finding, annotated_frame = self.vision_detector.process_frame(frame, unit_id)
        finding_dict = asdict(finding)

        # Step 2: Llama + ExecuTorch Advisory Reasoning (Slide 4 Step 5)
        if finding.defect_detected:
            self.defect_count += 1
            advisory = self.llama_advisor.advise(finding_dict)
        else:
            advisory = AdvisoryOutput(
                explanation="Component surface conforms fully to nominal engineering specifications.",
                immediate_action="PASS UNIT TO NEXT ASSEMBLY STAGE.",
                root_cause_hypothesis="Process parameters within 6-sigma tolerance.",
                operator_guidance="No operator intervention required.",
                log_entry=f"UNIT: {unit_id} | STATUS: PASS | All inspection criteria satisfied.",
                time_to_first_token_ms=0.0,
                generation_time_ms=0.0,
                tokens_per_second=0.0,
                total_tokens=0,
                runtime=self.llama_advisor.runtime_name
            )

        advisory_dict = asdict(advisory)

        # Step 3: Structured Logging (Slide 4 Step 6)
        self._log_inspection(finding, advisory)

        # Aggregate full response for UI & Benchmarks
        return {
            "unit_id": unit_id,
            "status": "REJECT" if finding.severity in ["HIGH", "CRITICAL"] else ("REWORK" if finding.severity == "MEDIUM" else "PASS"),
            "finding": finding_dict,
            "advisory": advisory_dict,
            "annotated_frame": annotated_frame
        }

    def _log_inspection(self, finding: StructuredDefectFinding, advisory: AdvisoryOutput):
        """Saves inspection transaction to SQLite table."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO inspection_log VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                finding.unit_id,
                finding.timestamp,
                1 if finding.defect_detected else 0,
                finding.defect_class,
                finding.severity,
                finding.confidence,
                finding.location,
                finding.total_vision_latency_ms,
                finding.fps,
                advisory.immediate_action,
                advisory.explanation,
                advisory.operator_guidance,
                advisory.log_entry,
                advisory.time_to_first_token_ms,
                advisory.tokens_per_second,
                advisory.runtime
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[QC Assistant] Log write error: {e}")

    def get_spc_metrics(self) -> Dict[str, Any]:
        """Calculates Statistical Process Control (SPC) yield and distribution."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM inspection_log", conn)
        conn.close()

        total = len(df)
        if total == 0:
            return {
                "total_inspected": 0,
                "total_defects": 0,
                "yield_percent": 100.0,
                "defect_rate_percent": 0.0,
                "defect_breakdown": {},
                "severity_breakdown": {}
            }

        defects_df = df[df["defect_detected"] == 1]
        defect_count = len(defects_df)
        defect_rate = (defect_count / total) * 100.0
        yield_rate = 100.0 - defect_rate

        defect_breakdown = defects_df["defect_class"].value_counts().to_dict()
        severity_breakdown = df["severity"].value_counts().to_dict()

        return {
            "total_inspected": total,
            "total_defects": defect_count,
            "yield_percent": round(yield_rate, 2),
            "defect_rate_percent": round(defect_rate, 2),
            "defect_breakdown": defect_breakdown,
            "severity_breakdown": severity_breakdown
        }

    def get_recent_logs(self, limit: int = 15) -> pd.DataFrame:
        """Returns the most recent inspection transactions."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f"SELECT * FROM inspection_log ORDER BY timestamp DESC LIMIT {limit}", conn)
        conn.close()
        return df
