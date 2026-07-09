"""
Phase 4 prompt templates and grounding-critical helpers.

Everything the LLM sees is defined here. The single most important design goal is
GROUNDING: the model must produce root causes, corrective actions, and statistics
ONLY from the retrieved cases/standards passed in the prompt. A report that invents
a root cause not present in the retrieved context is worse than no report.

Also here (per the Phase 4 spec) is the bbox -> human-readable location translation,
so the prompt builder and the fallback generator share one implementation.

FALLBACK_TEMPLATES vocabulary is pulled directly from data/cases/historical_cases.json
so fallback reports read like the LLM ones, not like generic boilerplate.
"""

from __future__ import annotations

from src.retrieval.schemas import CaseResult, StandardResult
from src.vision.constants import CLASS_NAMES, IMAGE_HEIGHT, IMAGE_WIDTH

# Imported (not hardcoded) so report tone can never drift from the retrieval pipeline.
from src.retrieval.confidence import HUMAN_REVIEW_SIMILARITY_THRESHOLD

# --------------------------------------------------------------------------------------
# Location translation (bbox -> 3x3 grid label)
# --------------------------------------------------------------------------------------

_ROW_LABELS = ("top", "centre", "bottom")
_COL_LABELS = ("left", "centre", "right")


def bbox_to_location(
    bbox: list[float] | None,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
) -> str:
    """Translate a [x1,y1,x2,y2] pixel bbox to a 3x3-grid location label (e.g. 'top-left')."""
    if not bbox or len(bbox) != 4 or image_width <= 0 or image_height <= 0:
        return "location unknown"
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    # Clamp the centre into [0, dim) so out-of-range boxes still map to an edge cell.
    col = min(int(max(cx, 0.0) / image_width * 3), 2)
    row = min(int(max(cy, 0.0) / image_height * 3), 2)
    if row == 1 and col == 1:
        return "centre"
    return f"{_ROW_LABELS[row]}-{_COL_LABELS[col]}"


# --------------------------------------------------------------------------------------
# Prompt context builders
# --------------------------------------------------------------------------------------


def format_cases_block(cases: list[CaseResult]) -> str:
    """Render retrieved cases as a numbered context block for the prompt."""
    if not cases:
        return "(no historical cases were retrieved for this defect)"
    lines: list[str] = []
    for c in cases:
        lines.append(
            f"- case_id: {c.case_id} (similarity {c.similarity_score:.2f}, severity {c.severity}/5)\n"
            f"    component_type: {c.component_type or 'unspecified'}\n"
            f"    root_cause: {c.root_cause}\n"
            f"    corrective_action: {c.corrective_action}\n"
            f"    outcome_notes: {c.outcome_notes}"
        )
    return "\n".join(lines)


def format_standards_block(standards: list[StandardResult]) -> str:
    """Render retrieved standards as a numbered context block for the prompt."""
    if not standards:
        return "(no IPC standards excerpts were retrieved for this defect)"
    lines: list[str] = []
    for s in standards:
        lines.append(
            f"- section_id: {s.section_id} (source: {s.source_doc}, relevance {s.relevance_score:.2f})\n"
            f"    excerpt: {s.excerpt}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# System + user prompt templates
# --------------------------------------------------------------------------------------

SYSTEM_PROMPT = f"""\
You are a PCB quality-control inspector with expertise in IPC-A-610 acceptance criteria \
and bare-board fabrication defect analysis (open, short, mousebite, spur, copper, pin-hole).

Your job is to write ONE structured defect report as strict JSON.

GROUNDING RULES — these override everything else:
- Use ONLY the retrieved historical cases and IPC standards provided in the user message. \
Do NOT introduce root causes, corrective actions, statistics, or IPC section numbers that \
are not present in that provided context.
- Set root_cause.evidence_basis to the case_id values you actually relied on, and ONLY \
case_ids that appear in the provided cases list. Never cite a case_id that is not listed.
- If no cases were retrieved, or every retrieved case has a similarity score below \
{HUMAN_REVIEW_SIMILARITY_THRESHOLD:.2f}, you MUST set root_cause.unsupported = true, state \
plainly in primary_cause that root-cause analysis is not supported by historical data, and \
leave evidence_basis as an empty list.
- ipc_reference fields must be null unless a standards excerpt with that section_id was \
provided; never invent an IPC section number.

WRITING RULES:
- Write the narrative for a factory-floor technician: plain English, standard IPC \
terminology only, no academic hedging, no marketing language. 2-3 sentences.
- Keep contributing_factors to at most 3, and only if supported by the context.

OUTPUT RULES:
- Respond with a SINGLE valid JSON object and nothing else — no preamble, no markdown \
fences, no commentary outside the JSON.
- The JSON must have exactly these keys:
  defect_class (string),
  location (string),
  severity: {{ level: one of "critical"|"major"|"minor"|"observation", score: integer 1-5, \
rationale: string, ipc_reference: string or null }},
  root_cause: {{ primary_cause: string, contributing_factors: array of strings (max 3), \
confidence: one of "high"|"medium"|"low"|"uncertain", evidence_basis: array of strings, \
unsupported: boolean }},
  corrective_action: {{ immediate: string, process_adjustment: string, re_inspection: string, \
ipc_reference: string or null }},
  narrative: string
- Do NOT include a generated_by key; it is set by the system.
- If a value cannot be determined from the provided context, use null (or an empty array \
for list fields) — never invent a value.
"""

# Placeholders are filled with str.format(**kwargs). No literal { } braces appear below,
# so formatting is safe; the JSON shape is described in SYSTEM_PROMPT instead.
DEFECT_REPORT_PROMPT = """\
Detected defect class: {defect_class}
Detection confidence band (from the vision + retrieval pipeline): {confidence_band}
Flagged for human review by the pipeline: {flagged}
Defect location on the board: {location}

RETRIEVED HISTORICAL CASES (this is your only permitted evidence for root cause):
{cases_block}

RETRIEVED IPC STANDARDS EXCERPTS (your only permitted source for ipc_reference values):
{standards_block}

Instructions:
- Base root_cause.primary_cause and root_cause.contributing_factors only on the retrieved \
cases above. Base root_cause.evidence_basis only on the case_id values listed above; do not \
reference any case not in this list.
- Set root_cause.confidence to "{confidence_band}" unless the evidence clearly contradicts it.
- Populate corrective_action from the corrective_action fields of the retrieved cases.
- Only set an ipc_reference (in severity or corrective_action) to a section_id that appears \
in the standards excerpts above; otherwise use null.
- Set defect_class to "{defect_class}" and location to "{location}" exactly as given.

Respond with the JSON object now.
"""


def build_defect_report_prompt(
    defect_class: str,
    confidence_band: str,
    flagged_for_human_review: bool,
    location: str,
    cases: list[CaseResult],
    standards: list[StandardResult],
) -> str:
    """Fill DEFECT_REPORT_PROMPT with the retrieval context for one defect."""
    return DEFECT_REPORT_PROMPT.format(
        defect_class=defect_class,
        confidence_band=confidence_band,
        flagged=str(flagged_for_human_review),
        location=location,
        cases_block=format_cases_block(cases),
        standards_block=format_standards_block(standards),
    )


# --------------------------------------------------------------------------------------
# Fallback templates — one per defect class, vocabulary grounded in historical_cases.json.
# Used when the LLM is unavailable, the call fails, or its output fails validation.
# The generator fills defect_class, location, confidence, evidence_basis (empty),
# unsupported (True), and generated_by ("fallback") at runtime; these provide the prose.
# --------------------------------------------------------------------------------------

FALLBACK_TEMPLATES: dict[str, dict] = {
    "open": {
        "severity_level": "major",
        "severity_score": 4,
        "severity_rationale": "An open trace is a conductor discontinuity that typically breaks the affected net.",
        "primary_cause": "Conductor discontinuity consistent with over-etch or a mechanical nick during depanelization; not confirmed against historical data.",
        "contributing_factors": ["photoresist lift-off during development", "panel-edge handling damage"],
        "immediate": "Quarantine the board and run an electrical continuity test on the affected net.",
        "process_adjustment": "Tighten etch time and rinse control; consider laser depanelization to avoid edge nicks.",
        "re_inspection": "Re-run AOI plus electrical continuity on the affected net and adjacent panels.",
        "narrative": "An open ({defect_class}) was detected at the {location} of the board. Likely causes are over-etch or handling damage, but this could not be confirmed against historical cases. Quarantine the board and confirm continuity before release.",
    },
    "short": {
        "severity_level": "major",
        "severity_score": 4,
        "severity_rationale": "An unintended copper bridge shorts separate nets and can cause functional failure.",
        "primary_cause": "Unintended copper bridge between nets consistent with an etch boundary failure or ionic contamination; not confirmed against historical data.",
        "contributing_factors": ["excess solder paste from worn stencil apertures", "inadequate DI rinse leaving ionic residue"],
        "immediate": "Quarantine the board and isolate the shorted nets from power-on testing.",
        "process_adjustment": "Inspect stencil apertures and improve DI rinse to remove ionic contamination.",
        "re_inspection": "AOI clearance check plus insulation-resistance / hi-pot test on the affected nets.",
        "narrative": "A short ({defect_class}) was detected at the {location} of the board. It resembles an etch-boundary bridge or contamination short, but this was not confirmed against historical cases. Isolate the nets and verify insulation resistance before release.",
    },
    "mousebite": {
        "severity_level": "minor",
        "severity_score": 3,
        "severity_rationale": "Edge notching reduces conductor width and can approach the minimum trace-width limit.",
        "primary_cause": "Lateral undercut (edge notch) consistent with aggressive or uneven etching or photoresist edge damage; not confirmed against historical data.",
        "contributing_factors": ["unstable etchant ORP/pH", "particulate on the exposure vacuum frame"],
        "immediate": "Measure remaining conductor width at the notch against the minimum trace-width spec.",
        "process_adjustment": "Stabilize etchant chemistry and spray pattern; clean the exposure frame between panels.",
        "re_inspection": "Cross-section or trace-width AOI on audit panels from the same lot.",
        "narrative": "A mousebite ({defect_class}) edge notch was detected at the {location} of the board. It is consistent with over-etch or resist damage, though not confirmed against historical cases. Verify the remaining trace width still meets the minimum before release.",
    },
    "spur": {
        "severity_level": "minor",
        "severity_score": 2,
        "severity_rationale": "A copper protrusion into a clearance zone reduces spacing but often remains within tolerance.",
        "primary_cause": "Copper protrusion into a clearance zone consistent with photoresist scumming or artwork misalignment; not confirmed against historical data.",
        "contributing_factors": ["incomplete etch from resist scumming", "second-side artwork misalignment"],
        "immediate": "Verify the clearance from the spur to the nearest conductor still meets the minimum spacing.",
        "process_adjustment": "Add a post-develop descum plasma step and verify double-sided artwork alignment.",
        "re_inspection": "Optical clearance-gap inspection on audit panels.",
        "narrative": "A spur ({defect_class}) was detected at the {location} of the board. It is consistent with incomplete etch or artwork misalignment, but not confirmed against historical cases. Confirm the clearance still meets the minimum spacing before release.",
    },
    "copper": {
        "severity_level": "minor",
        "severity_score": 3,
        "severity_rationale": "Spurious copper in a mask-free area risks bridging if it is not isolated.",
        "primary_cause": "Spurious copper island/nodule in a mask-free area consistent with a foil burr or a dry-film pinhole allowing stray plating; not confirmed against historical data.",
        "contributing_factors": ["routing-bit burr pressed into laminate", "dry-film photoresist pinhole"],
        "immediate": "Confirm the copper island is isolated and poses no bridging risk to adjacent conductors.",
        "process_adjustment": "Shorten routing-bit change interval, increase lamination pressure, and enforce film expiry tracking.",
        "re_inspection": "AOI of tooling and mask-free zones on subsequent panels.",
        "narrative": "Spurious copper ({defect_class}) was detected at the {location} of the board. It resembles a foil burr or plating nodule, though not confirmed against historical cases. Confirm it is isolated with no bridging risk before release.",
    },
    "pin-hole": {
        "severity_level": "minor",
        "severity_score": 2,
        "severity_rationale": "A plating void can expose base copper but is often localized and repairable.",
        "primary_cause": "Plating void consistent with micro-etch residue or organic contamination blocking plating; not confirmed against historical data.",
        "contributing_factors": ["insufficient cleaning after micro-etch", "low rinse conductivity setpoint"],
        "immediate": "Inspect for exposed base copper or a barrel void at the affected feature.",
        "process_adjustment": "Add an alkaline cleaner stage and raise the rinse conductivity setpoint.",
        "re_inspection": "Plating-thickness measurement or peel test per IPC criteria on audit coupons.",
        "narrative": "A pin-hole ({defect_class}) plating void was detected at the {location} of the board. It is consistent with residue blocking plating, but not confirmed against historical cases. Check for exposed base copper before release.",
    },
}

# Guard against silent taxonomy drift between constants.py and the fallback templates.
assert set(FALLBACK_TEMPLATES) == set(CLASS_NAMES), (
    f"FALLBACK_TEMPLATES keys {sorted(FALLBACK_TEMPLATES)} do not match "
    f"CLASS_NAMES {sorted(CLASS_NAMES)}"
)
