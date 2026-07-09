"""
Phase 3 — Step 3: Ingest PCB inspection standards into ChromaDB.

Creates the "pcb_standards" collection with 18 realistic SOP/standards
excerpts grounded in IPC-A-610 terminology, covering all 6 DeepPCB defect
classes. Uses the same embedding model as Phase 2 case ingestion.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.data.ingest_cases import DEFAULT_CHROMA_DIR, EMBEDDING_MODEL
from src.retrieval.retriever import STANDARDS_COLLECTION

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standards documents
#
# Each entry mirrors a real IPC-A-610 / IPC-6012 inspection clause in
# structure: defect description → acceptance criteria → corrective action →
# re-inspection requirement. Severity threshold maps to the HistoricalDefectCase
# severity scale (1=minor, 5=critical).
# ---------------------------------------------------------------------------

STANDARDS_DOCUMENTS: list[dict[str, Any]] = [
    # ── OPEN ─────────────────────────────────────────────────────────────────
    {
        "section_id": "IPC-A610-S3-01",
        "source_doc": "IPC-A-610 Rev H, Section 6.2.1 — Conductor Continuity",
        "defect_class": "open",
        "severity_threshold": 4,
        "excerpt": (
            "Open Defect — Conductor Continuity: An open circuit condition exists when "
            "a conductive path is interrupted, reducing measurable continuity below the "
            "design network requirement. Class 3 (high reliability) assemblies shall have "
            "zero open defects on functional nets. Detection method: 100% electrical "
            "continuity test per IPC-9252. Disposition: boards with open conditions are "
            "non-conforming and must be rejected or reworked. Re-inspection: after rework, "
            "full electrical test must be repeated on all nets associated with the affected layer."
        ),
    },
    {
        "section_id": "IPC-6012-S3-02",
        "source_doc": "IPC-6012 Rev E, Section 3.6.2 — Conductor Width Reduction",
        "defect_class": "open",
        "severity_threshold": 3,
        "excerpt": (
            "Conductor Width Reduction (Potential Open Precursor): Etching process variation "
            "causing conductor width reduction greater than 20% of the minimum design width "
            "is a Class 3 reject condition. Root causes include etch factor drift, spray "
            "pressure variation, or photoresist adhesion loss. Corrective action: adjust "
            "etch chemistry ORP within ±5 mV of target, recalibrate spray nozzle pressure, "
            "and perform micro-etch coupon verification. Sampling frequency: measure width "
            "on 3 coupons per production lot using calibrated cross-sectioning. "
            "Re-inspection: AOI re-run on any panel from the affected lot."
        ),
    },
    {
        "section_id": "IPC-6012-S3-03",
        "source_doc": "IPC-6012 Rev E, Section 3.7.4 — Via Barrel Plating Voids",
        "defect_class": "open",
        "severity_threshold": 4,
        "excerpt": (
            "Via Barrel Plating Void (Open Risk): A void in the electroplated copper barrel "
            "of a plated through-hole (PTH) is a non-conformance when the void exceeds 5% "
            "of the barrel circumference for Class 3. Voids arise from air entrapment during "
            "panel wet-in, inadequate agitation, or depleted plating chemistry. "
            "Corrective action: enable cathode rod oscillation, reduce panel rack density, "
            "and replenish plating bath within established SPC limits. "
            "Verification: cross-section 5 representative vias and inspect at 200× for "
            "continuous copper coverage. Disposition: panels with barrel voids > 5% must "
            "be scrapped or subject to MRB review."
        ),
    },

    # ── SHORT ─────────────────────────────────────────────────────────────────
    {
        "section_id": "IPC-A610-S4-01",
        "source_doc": "IPC-A-610 Rev H, Section 6.2.2 — Short Circuit",
        "defect_class": "short",
        "severity_threshold": 5,
        "excerpt": (
            "Short Circuit Defect: A short circuit exists when an unintended conductive "
            "path is present between two electrically separate conductors. Any short between "
            "functionally different nets is a Class 3 reject — zero acceptance. "
            "Common fabrication causes: solder mask misregistration exposing copper between "
            "fine-pitch pads, copper sliver from CAM cleanup gap, ionic contamination from "
            "inadequate final rinse. Corrective action for mask-related shorts: recalibrate "
            "LDI exposure energy and verify mask develop time on test coupon before "
            "production restart. For ionic contamination: verify DI rinse resistivity "
            "above 10 MΩ·cm and perform ROSE test per IPC-TM-650 2.3.25. "
            "Re-inspection: 100% electrical test on all nets; re-run AOI at 2× resolution."
        ),
    },
    {
        "section_id": "IPC-6012-S4-02",
        "source_doc": "IPC-6012 Rev E, Section 3.5.3 — Solder Mask Integrity",
        "defect_class": "short",
        "severity_threshold": 3,
        "excerpt": (
            "Solder Mask Dam Integrity (Short Precursor): The minimum solder mask dam "
            "between adjacent pads must be ≥ 0.075 mm for Class 3 assemblies. "
            "Insufficient dam from mask misregistration or over-development leaves exposed "
            "copper susceptible to dendritic growth under humidity, creating latent shorts. "
            "Inspection: measure mask dam width on 5 representative locations per panel "
            "using calibrated optical measurement. "
            "Corrective action: adjust LDI exposure dose by ±2%, recalibrate develop time, "
            "and re-verify on test panel before production restart. "
            "Disposition: panels with mask dam < 0.05 mm are Class 3 non-conformances."
        ),
    },
    {
        "section_id": "IPC-6012-S4-03",
        "source_doc": "IPC-6012 Rev E, Section 3.4.1 — Conductor Spacing",
        "defect_class": "short",
        "severity_threshold": 4,
        "excerpt": (
            "Conductor Spacing Violation (Copper Bridge / Short): Minimum conductor-to-"
            "conductor spacing must be maintained as specified in the design file. "
            "Any copper bridge spanning the full spacing gap between isolated nets is a "
            "Class 2/3 reject. Root cause: photoresist scumming blocks etch between fine "
            "traces; electroless copper over-deposition bridges gap during seed layer step; "
            "or CAM artwork cleanup leaves copper sliver. "
            "Corrective action: add post-develop plasma descum step; enforce CAM DRC for "
            "minimum sliver width ≥ 0.15 mm; adjust electroless bath loading. "
            "Verification: electrical isolation test plus optical inspection at ≥ 40× "
            "magnification on sampled panels. Failed panels: 100% electrical test required."
        ),
    },

    # ── MOUSEBITE ─────────────────────────────────────────────────────────────
    {
        "section_id": "IPC-A610-S5-01",
        "source_doc": "IPC-A-610 Rev H, Section 6.2.3 — Conductor Edge Quality",
        "defect_class": "mousebite",
        "severity_threshold": 2,
        "excerpt": (
            "Mouse-Bite Defect — Conductor Edge Quality: A mouse-bite is a localised "
            "edge notch on a conductor caused by lateral under-etch or photoresist damage, "
            "reducing the effective conductor width at that point. "
            "Acceptance criterion: the conductor width at any mouse-bite must not be "
            "reduced below the minimum width specified in the design file. For Class 3, "
            "conductor width reduction > 20% from nominal is reject. "
            "Root causes: pH oscillation in etchant causing lateral undercut; particulate "
            "damage to photoresist edge. Corrective action: stabilise etchant ORP within "
            "±5 mV; HEPA-filter clean room; clean exposure vacuum frame between panels. "
            "Re-inspection: AOI trace width verification on next 3 production panels."
        ),
    },
    {
        "section_id": "IPC-6012-S5-02",
        "source_doc": "IPC-6012 Rev E, Section 3.6.3 — Etch Factor Uniformity",
        "defect_class": "mousebite",
        "severity_threshold": 3,
        "excerpt": (
            "Etch Factor Non-Uniformity (Mouse-Bite Root Cause): Variation in etch factor "
            "across a panel — particularly at high-current-density areas — causes uneven "
            "lateral undercut, producing mouse-bite profiles on conductor edges. "
            "Acceptable etch factor variation: ≤ 15% across panel for Class 3. "
            "Corrective action: rebalance cathode/anode current density; throttle local "
            "current by adjusting bus bar geometry; add dummy copper features to equalise "
            "current distribution in sparse areas. "
            "Process verification: measure etch factor on cross-section coupons at panel "
            "corners and centre; document in process control chart. "
            "Frequency: one cross-section set per production lot."
        ),
    },
    {
        "section_id": "IPC-6012-S5-03",
        "source_doc": "IPC-6012 Rev E, Section 3.8.1 — Photoresist Process Control",
        "defect_class": "mousebite",
        "severity_threshold": 2,
        "excerpt": (
            "Photoresist Edge Adhesion (Mouse-Bite Precursor): Poor photoresist edge "
            "adhesion caused by particulate contamination, worn squeegee, or sub-optimal "
            "vacuum contact during exposure produces notch defects (mouse bites) at "
            "conductor edges after etch. "
            "Control requirements: exposure vacuum frame cleanliness inspected before "
            "each shift; squeegee hardness verified every 200 panels; clean room particle "
            "count < 10,000/m³ (ISO Class 7). "
            "Corrective action on exceedance: clean frame, replace squeegee if hardness "
            "out of specification, run test panel before production restart. "
            "Disposition: panels produced during a known contamination event require "
            "100% AOI re-inspection before release."
        ),
    },

    # ── SPUR ──────────────────────────────────────────────────────────────────
    {
        "section_id": "IPC-A610-S6-01",
        "source_doc": "IPC-A-610 Rev H, Section 6.2.4 — Copper Protrusions (Spurs)",
        "defect_class": "spur",
        "severity_threshold": 3,
        "excerpt": (
            "Copper Spur Defect: A spur is an unwanted copper protrusion extending from "
            "a conductor into an adjacent clearance zone. A spur that reduces the "
            "conductor-to-conductor spacing below the design minimum is a reject condition. "
            "For Class 3, zero spurs bridging isolation gaps on functional nets. "
            "Root causes: photoresist scumming during develop leaves resist island that "
            "plates into spur; acid trap in CAD artwork creates etch shadow; foreign copper "
            "fragment from handling laminated onto panel. "
            "Corrective action: add post-develop plasma descum before etch; update CAM "
            "DRC to remove acid trap angles < 45°; add FOD control (ioniser bars, "
            "post-handling wipe) at panel layup. "
            "Re-inspection: AOI re-run on affected lot; electrical isolation test on "
            "nets adjacent to flagged areas."
        ),
    },
    {
        "section_id": "IPC-6012-S6-02",
        "source_doc": "IPC-6012 Rev E, Section 3.6.4 — CAM Design Rule Compliance",
        "defect_class": "spur",
        "severity_threshold": 2,
        "excerpt": (
            "CAM Artwork — Acid Trap and Re-entrant Angle Control (Spur Prevention): "
            "Re-entrant angles and acid traps in artwork create differential etch rates "
            "that leave copper spurs at acute corners. "
            "Design rule: all interior copper corners must be ≥ 45°; all clearance gaps "
            "must be verifiably connected to etchant flow paths. "
            "CAM verification: DRC check for acid traps must be run on all Gerber files "
            "before photoplot; results reviewed by CAM engineer before release to production. "
            "Corrective action for spur found in production: trace Gerber file for acid "
            "trap; update artwork and re-verify before next production run. "
            "Affected lot disposition: 100% AOI on all panels from suspect period."
        ),
    },

    # ── COPPER (spurious island / nodule) ────────────────────────────────────
    {
        "section_id": "IPC-A610-S7-01",
        "source_doc": "IPC-A-610 Rev H, Section 6.2.5 — Spurious Copper",
        "defect_class": "copper",
        "severity_threshold": 3,
        "excerpt": (
            "Spurious Copper Defect (Island / Nodule): Spurious copper is any copper "
            "feature in the finished board that is not part of the design netlist — "
            "including nodules, islands, or residual plating in clearance zones. "
            "Class 3 acceptance: zero spurious copper in any isolation gap between "
            "functionally different nets. Copper in non-functional areas (e.g., panel "
            "tooling zone) is permissible if not within 1 mm of design copper. "
            "Root causes: photoresist pinhole allows copper nodule; dry film expiry "
            "causes lamination defects; copper foil burr from routing deposited in layup. "
            "Corrective action: enforce photoresist film expiry date; increase vacuum "
            "lamination pressure; halve routing bit change interval and add blow-off. "
            "Re-inspection: full AOI re-run plus electrical test on affected lot."
        ),
    },
    {
        "section_id": "IPC-6012-S7-02",
        "source_doc": "IPC-6012 Rev E, Section 3.3.2 — Plating Uniformity",
        "defect_class": "copper",
        "severity_threshold": 3,
        "excerpt": (
            "Plating Non-Uniformity Leading to Copper Nodule: Localised over-plating at "
            "high-current-density corners or near anode geometry irregularities produces "
            "copper nodules that may bridge clearance zones. "
            "Control limit: copper thickness variation ≤ 20% across panel (measured "
            "at 9 points) for Class 3. Nodule height must not exceed 25% of conductor "
            "thickness above the plane level. "
            "Corrective action: redesign anode shielding; adjust brightener/leveler "
            "concentration within bath specification; add thiourea leveler. "
            "Process monitoring: measure plating thickness at 9-point grid once per lot; "
            "plot on SPC chart; trigger corrective action on 2 consecutive out-of-limit "
            "lots. Affected panels: electrical test plus AOI before release."
        ),
    },

    # ── PIN-HOLE ──────────────────────────────────────────────────────────────
    {
        "section_id": "IPC-A610-S8-01",
        "source_doc": "IPC-A-610 Rev H, Section 6.2.6 — Pin-Hole Voids in Copper",
        "defect_class": "pin-hole",
        "severity_threshold": 4,
        "excerpt": (
            "Pin-Hole Defect — Copper Plane Void: A pin-hole is a through-void in a "
            "copper plane, pour, or pad where the substrate is visible through the copper. "
            "For Class 3, no pin-holes are permitted in power or ground planes or in "
            "controlled-impedance reference planes. Pin-holes in non-functional copper "
            "are a Class 2 defect if area < 0.01 mm². "
            "Root causes: void in electroless copper seed layer; entrapped air during "
            "panel wet-in; laser drill debris blocking metallisation. "
            "Corrective action: replenish electroless bath per supplier SDS; add cathode "
            "rod oscillation; perform ultrasonic clean and vacuum desmear after laser drill. "
            "Verification: backlight inspection of copper planes at 10×; cross-section "
            "5 representative areas on sampled panels. Re-inspection: full electrical "
            "test on affected lot before release."
        ),
    },
    {
        "section_id": "IPC-6012-S8-02",
        "source_doc": "IPC-6012 Rev E, Section 3.7.2 — Electroless Copper Coverage",
        "defect_class": "pin-hole",
        "severity_threshold": 4,
        "excerpt": (
            "Electroless Copper Seed Coverage (Pin-Hole Prevention): Electroless copper "
            "must provide continuous, void-free seed coverage on all substrate surfaces "
            "prior to electrolytic plating. Voids in the seed layer propagate as pin-holes "
            "through subsequent plating. "
            "Acceptance: backlight test must show no visible voids (>0 lux transmission "
            "through copper) on a 30-panel sample per lot. "
            "Process controls: electroless bath chemistry replenishment per SDS schedule; "
            "bath contamination monitored by weekly ICP-OES analysis; filter changed per "
            "manufacturer interval. "
            "Corrective action on pin-hole exceedance: replace electroless bath, run "
            "qualification panels, submit 5 cross-sections for microscopy before restarting "
            "production. Affected lot: 100% electrical test; scrapped unless MRB approved."
        ),
    },
    {
        "section_id": "IPC-6012-S8-03",
        "source_doc": "IPC-6012 Rev E, Section 3.7.5 — Plating Tank Contamination",
        "defect_class": "pin-hole",
        "severity_threshold": 3,
        "excerpt": (
            "Plating Tank Contamination Control (Pin-Hole and Void Prevention): Oil, "
            "organic, or particulate contamination in the plating tank inhibits copper "
            "nucleation at localised sites, creating pin-hole voids in the finished plate. "
            "Control requirements: oil content of plating bath < 5 ppm (measured weekly "
            "by FTIR); conveyor rollers inspected and cleaned daily; compressed air lines "
            "fitted with oil-water separators verified monthly. "
            "Corrective action on contamination event: dummy-plate a sacrificial panel "
            "set; carbon treat bath; re-qualify with cross-section before production. "
            "Environmental control: plating area positive-pressure with filtered supply "
            "air; HVAC filter changed per maintenance schedule."
        ),
    },
]


def _build_embedding_text(doc: dict[str, Any]) -> str:
    return (
        f"Defect class: {doc['defect_class']}. "
        f"Source: {doc['source_doc']}. "
        f"Standard excerpt: {doc['excerpt']}"
    )


def ingest_standards(
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = STANDARDS_COLLECTION,
    model_name: str = EMBEDDING_MODEL,
    reset: bool = True,
) -> int:
    """
    Embed the standards documents and store them in ChromaDB.

    Returns the number of documents ingested.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    texts = [_build_embedding_text(d) for d in STANDARDS_DOCUMENTS]
    ids = [d["section_id"] for d in STANDARDS_DOCUMENTS]
    metadatas = [
        {
            "section_id": d["section_id"],
            "source_doc": d["source_doc"],
            "defect_class": d["defect_class"],
            "severity_threshold": d["severity_threshold"],
            "excerpt": d["excerpt"],
        }
        for d in STANDARDS_DOCUMENTS
    ]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    chroma_dir = chroma_dir.resolve()
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if reset:
        try:
            client.delete_collection(collection_name)
            logger.info("Deleted existing collection '%s'", collection_name)
        except Exception as exc:
            exc_str = str(exc).lower()
            if "does not exist" in exc_str or "not found" in exc_str or "no collection" in exc_str:
                logger.debug("Collection '%s' did not exist; nothing to delete", collection_name)
            else:
                logger.error("Unexpected error deleting collection '%s': %s", collection_name, exc)
                raise

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    stored = collection.count()
    if stored != len(STANDARDS_DOCUMENTS):
        raise RuntimeError(
            f"Post-ingest count mismatch: sent {len(STANDARDS_DOCUMENTS)}, "
            f"collection reports {stored}."
        )

    print(
        f"Ingested {stored} standards documents into '{collection_name}' at {chroma_dir}"
    )
    return stored


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Ingest PCB inspection standards into ChromaDB"
    )
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection", default=STANDARDS_COLLECTION)
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()

    ingest_standards(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        model_name=args.model,
        reset=not args.no_reset,
    )


if __name__ == "__main__":
    main()
