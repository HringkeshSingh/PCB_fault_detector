"""
Streamlit GUI for the PCB defect detection + RAG retrieval pipeline.

Upload a PCB image (or take a webcam snapshot), send it to the FastAPI
backend's /analyze endpoint, and display detections, retrieved historical
cases, standards excerpts, and confidence scoring.

Run:
    1. Start the API:      uvicorn src.api.main:app --reload
    2. Start the GUI:      streamlit run src/gui/app.py
"""

from __future__ import annotations

import io

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

API_URL_DEFAULT = "http://127.0.0.1:8000"

CONFIDENCE_COLORS = {
    "high": "#2ecc71",
    "medium": "#f1c40f",
    "low": "#e67e22",
    "uncertain": "#e74c3c",
}

# Report severity band -> color, for the Phase 4 inspection report headline.
SEVERITY_COLORS = {
    "critical": "#e74c3c",
    "major": "#e67e22",
    "minor": "#f1c40f",
    "observation": "#3498db",
}

st.set_page_config(page_title="PCB Defect Analyzer", layout="wide")


def _draw_boxes(
    image: Image.Image,
    detections: list[dict],
    class_to_band: dict[str, str] | None = None,
) -> Image.Image:
    """Draw a box for every raw detection.

    `detections` are raw /detect items ({defect_class, confidence, bbox}) so that
    multiple instances of the same defect class are all drawn — /analyze collapses
    to one result per class and would hide duplicates. Each box is colored by its
    class's retrieval confidence band (from class_to_band, populated from /analyze)
    so the image coloring stays consistent with the result cards and the sidebar
    legend; classes with no band fall back to a neutral blue.
    """
    class_to_band = class_to_band or {}
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        band = class_to_band.get(det["defect_class"])
        color = CONFIDENCE_COLORS.get(band, "#3498db")
        label = f'{det["defect_class"]} {det["confidence"]:.2f}'

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(
            [text_bbox[0], text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 2],
            fill=color,
        )
        draw.text((x1 + 2, y1 - 2), label, fill="black", font=font)

    return annotated


def _render_result_card(res: dict) -> None:
    """Render one RetrievalResult (detection + cases + standards + confidence) as a panel."""
    det = res["detection"]
    meta = res["retrieval_metadata"]
    band = meta["retrieval_confidence"]
    color = CONFIDENCE_COLORS.get(band, "#3498db")

    st.markdown(
        f"### {det['defect_class']}  "
        f"<span style='color:{color}'>● {band.upper()}</span>",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    cols[0].metric("Detection confidence", f"{det['confidence']:.2f}")
    cols[1].metric("Top case similarity", f"{meta['top_case_similarity']:.2f}")
    cols[2].metric("Cases found", meta["cases_found"])
    cols[3].metric("Standards found", meta["standards_found"])

    if meta["flagged_for_human_review"]:
        st.warning("⚠️ Flagged for human review")
    if meta["standards_skipped"]:
        st.info("Standards retrieval was skipped (collection unavailable)")

    tab_cases, tab_standards = st.tabs(["Historical Cases", "Standards"])

    with tab_cases:
        if not res["retrieved_cases"]:
            st.write("No matching historical cases found.")
        for case in res["retrieved_cases"]:
            with st.expander(
                f"{case['case_id']} — similarity {case['similarity_score']:.2f} "
                f"— severity {case['severity']}/5"
            ):
                st.write(f"**Root cause:** {case['root_cause']}")
                st.write(f"**Corrective action:** {case['corrective_action']}")
                st.write(f"**Outcome notes:** {case['outcome_notes']}")
                citation = (
                    f"{case['case_id']}"
                    + (f", {case['component_type']}" if case.get("component_type") else "")
                    + (f", recorded {case['date_recorded']}" if case.get("date_recorded") else "")
                    + " — Synthetic Historical PCB Defect Case Database v1.0"
                )
                st.caption(f"📖 Citation: {citation}")

    with tab_standards:
        if not res["retrieved_standards"]:
            st.write("No matching standards excerpts found.")
        for std in res["retrieved_standards"]:
            with st.expander(
                f"[{std['section_id']}] {std['source_doc']} "
                f"— relevance {std['relevance_score']:.2f}"
            ):
                st.write(std["excerpt"])

    st.divider()


def _render_defect_report(dr: dict) -> None:
    """Render one Phase 4 DefectReport: narrative, severity, grounded root cause, actions."""
    sev = dr["severity"]
    sev_color = SEVERITY_COLORS.get(sev["level"], "#3498db")
    mode = dr["generated_by"]
    mode_badge = "🤖 LLM" if mode == "llm" else "📋 fallback"

    st.markdown(
        f"#### {dr['defect_class']} — "
        f"<span style='color:{sev_color}'>{sev['level'].upper()} ({sev['score']}/5)</span> "
        f"<span style='opacity:0.6;font-size:0.8em'>· {mode_badge} · {dr['location']}</span>",
        unsafe_allow_html=True,
    )
    st.write(dr["narrative"])

    rc = dr["root_cause"]
    if rc["unsupported"]:
        st.warning("⚠️ Root cause not supported by historical data — treat as provisional.")
    st.markdown(f"**Root cause** ({rc['confidence']} confidence): {rc['primary_cause']}")
    if rc["contributing_factors"]:
        st.markdown("Contributing factors: " + ", ".join(rc["contributing_factors"]))
    if rc["evidence_basis"]:
        st.caption("📖 Evidence: " + ", ".join(rc["evidence_basis"]))

    ca = dr["corrective_action"]
    with st.expander("Corrective action"):
        st.markdown(f"**Immediate:** {ca['immediate']}")
        st.markdown(f"**Process adjustment:** {ca['process_adjustment']}")
        st.markdown(f"**Re-inspection:** {ca['re_inspection']}")
        if ca.get("ipc_reference"):
            st.caption(f"IPC reference: {ca['ipc_reference']}")
    st.divider()


def _render_inspection_report(report: dict) -> None:
    """Render a full Phase 4 InspectionReport with headline metrics and per-defect reports."""
    meta = report["generation_metadata"]
    overall = report["overall_severity"]
    sev_color = SEVERITY_COLORS.get(overall, "#3498db")

    st.markdown(
        f"### Inspection Report "
        f"<span style='color:{sev_color}'>● {overall.upper()}</span>",
        unsafe_allow_html=True,
    )
    if meta["generation_mode"] == "fallback":
        st.info(
            "Running in **fallback mode** (no LLM). Reports use static templates. "
            "Install Ollama and `ollama pull llama3.2`, then restart the API, for generated narratives."
        )
    if report["requires_human_review"]:
        st.warning("⚠️ This board requires human review (one or more defects were flagged).")

    cols = st.columns(4)
    cols[0].metric("Total defects", report["total_defects"])
    cols[1].metric("Overall severity", overall)
    cols[2].metric("Mode", meta["generation_mode"])
    cols[3].metric("Fallback count", meta["fallback_count"])

    for dr in report["defect_reports"]:
        _render_defect_report(dr)

    st.caption(f"Report ID: {report['report_id']} · generated {report['generated_at']}")


def main() -> None:
    st.title("PCB Defect Analyzer")
    st.caption("Upload a PCB image to detect defects and retrieve matching historical cases and standards.")

    with st.sidebar:
        st.header("Settings")
        api_url = st.text_input("API base URL", value=API_URL_DEFAULT)
        try:
            health = requests.get(f"{api_url}/health", timeout=3).json()
            if health.get("model_loaded"):
                st.success("Model loaded ✓")
            else:
                st.error("Model not loaded — train a model first (see README).")
        except requests.exceptions.RequestException:
            st.error(f"Cannot reach API at {api_url}. Is uvicorn running?")

        # Phase 4 report-generator status (LLM vs fallback).
        try:
            rh = requests.get(f"{api_url}/report/health", timeout=3).json()
            if rh.get("llm_available"):
                st.success(f"Report LLM ✓ ({rh.get('model')})")
            else:
                st.warning("Report: fallback mode (no LLM)")
        except requests.exceptions.RequestException:
            pass

        st.divider()
        conf_threshold = st.slider(
            "Detection confidence threshold",
            min_value=0.05,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Lower values detect more defects but increase false positives. "
                 "Default 0.25 is tuned for DeepPCB dataset images.",
        )
        iou_threshold = st.slider(
            "NMS IoU threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.45,
            step=0.05,
            help="Controls how aggressively overlapping boxes are merged. "
                 "RAISE it (e.g. 0.6-0.7) to keep more nearby small defects that "
                 "would otherwise be suppressed as duplicates — a recall lever. "
                 "Lower it back toward 0.45 if you start seeing duplicate boxes.",
        )

        st.divider()
        st.markdown("**Confidence legend**")
        for band, color in CONFIDENCE_COLORS.items():
            st.markdown(f"<span style='color:{color}'>●</span> {band}", unsafe_allow_html=True)

        st.divider()
        st.info(
            "ℹ️ **Model domain:** This model was trained on DeepPCB dataset images "
            "(640×640 grayscale template-difference images). Real-world PCB photographs "
            "may produce fewer or no detections due to domain gap. For best results, "
            "use images from the DeepPCB dataset or retrain the model on your own data."
        )

    source = st.radio("Image source", ["Upload file", "Webcam snapshot"], horizontal=True)
    image_bytes: bytes | None = None
    filename = "upload.jpg"

    if source == "Upload file":
        uploaded = st.file_uploader("Choose a PCB image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            filename = uploaded.name
    else:
        snapshot = st.camera_input("Take a snapshot")
        if snapshot is not None:
            image_bytes = snapshot.getvalue()
            filename = "snapshot.jpg"

    if image_bytes is None:
        st.info("Upload an image or take a snapshot to begin.")
        return

    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if st.button("Analyze", type="primary"):
        content_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
        params = {"conf": conf_threshold, "iou": iou_threshold}

        def _post(endpoint: str):
            return requests.post(
                f"{api_url}/{endpoint}",
                params=params,
                files={"file": (filename, image_bytes, content_type)},
                timeout=60,
            )

        with st.spinner("Detecting defects and retrieving matches..."):
            try:
                # /detect -> every raw detection (used for drawing all boxes).
                # /analyze -> one result per defect class (used for the cards below).
                detect_resp = _post("detect")
                analyze_resp = _post("analyze")
            except requests.exceptions.RequestException as exc:
                st.error(f"Request failed: {exc}")
                return

        for label, resp in (("detect", detect_resp), ("analyze", analyze_resp)):
            if resp.status_code != 200:
                try:
                    detail = resp.json().get("detail", resp.text)
                except ValueError:
                    detail = resp.text
                st.error(f"API error on /{label} ({resp.status_code}): {detail}")
                return

        st.session_state["last_detections"] = detect_resp.json().get("detections", [])
        st.session_state["last_result"] = analyze_resp.json()
        st.session_state["last_image"] = original_image
        # Stash the request payload so the report can be generated on demand without re-uploading.
        st.session_state["last_payload"] = {
            "filename": filename,
            "content_type": content_type,
            "image_bytes": image_bytes,
            "params": params,
        }
        st.session_state.pop("last_report", None)  # invalidate any stale report

    if "last_result" in st.session_state:
        body = st.session_state["last_result"]
        detections = st.session_state.get("last_detections", [])
        image = st.session_state["last_image"]

        # Color every raw box by its class's retrieval band, kept consistent with the cards.
        class_to_band = {
            r["detection"]["defect_class"]: r["retrieval_metadata"]["retrieval_confidence"]
            for r in body["results"]
        }

        col_img, col_summary = st.columns([2, 1])
        with col_img:
            if not detections:
                st.image(image, caption="No defects detected", use_container_width=True)
            else:
                annotated = _draw_boxes(image, detections, class_to_band)
                n = len(detections)
                classes = body["total_detections"]
                st.image(
                    annotated,
                    caption=f"{n} defect{'s' if n != 1 else ''} found across {classes} class(es)",
                    use_container_width=True,
                )

        with col_summary:
            st.metric("Defects detected", len(detections))
            st.metric("Unique defect classes", body["total_detections"])
            if body["results"]:
                flagged = sum(1 for r in body["results"] if r["retrieval_metadata"]["flagged_for_human_review"])
                st.metric("Flagged for review", flagged)

        st.divider()
        for res in body["results"]:
            _render_result_card(res)

        # --- Phase 4: inspection report ---
        st.divider()
        st.subheader("📝 Inspection Report")
        payload = st.session_state.get("last_payload")
        if body["total_detections"] == 0:
            st.info("No defects detected — no report to generate.")
        elif payload is not None:
            if st.button("Generate inspection report"):
                with st.spinner("Generating inspection report..."):
                    try:
                        report_resp = requests.post(
                            f"{api_url}/report",
                            params=payload["params"],
                            files={"file": (payload["filename"], payload["image_bytes"], payload["content_type"])},
                            timeout=180,  # allow headroom for a slow local LLM
                        )
                    except requests.exceptions.RequestException as exc:
                        st.error(f"Report request failed: {exc}")
                        report_resp = None
                if report_resp is not None:
                    if report_resp.status_code == 200:
                        st.session_state["last_report"] = report_resp.json()
                    else:
                        try:
                            detail = report_resp.json().get("detail", report_resp.text)
                        except ValueError:
                            detail = report_resp.text
                        st.error(f"API error on /report ({report_resp.status_code}): {detail}")

            if "last_report" in st.session_state:
                _render_inspection_report(st.session_state["last_report"])


if __name__ == "__main__":
    main()
