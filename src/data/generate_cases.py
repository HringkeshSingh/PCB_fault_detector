"""Generate synthetic historical PCB defect cases grounded in manufacturing causes."""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from src.data.schemas import (
    DEFECT_CLASSES,
    DefectClass,
    HistoricalCaseCollection,
    HistoricalDefectCase,
)
from src.vision.constants import PROJECT_ROOT

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "cases" / "historical_cases.json"
TARGET_COUNT = 180

# ---------------------------------------------------------------------------
# Independent field pools per defect class.
#
# Root causes, corrective actions, and outcome notes are drawn independently
# so that the Cartesian product (8×8×8 = 512 combinations per class) is far
# larger than the ~30 cases needed per class — eliminating near-duplicates.
#
# All causes are grounded in bare-board PCB fabrication (etch, plate, laminate,
# drill, mask) — NOT SMT assembly. This keeps cases relevant to DeepPCB AOI.
# ---------------------------------------------------------------------------

ROOT_CAUSES: dict[DefectClass, list[str]] = {
    "open": [
        "Under-etch during subtractive etching left thin copper neck that opened after thermal cycling",
        "Incomplete copper plating in via barrel due to air bubble entrapment during electroplating",
        "Mechanical nick during depanelization severed trace at panel edge",
        "Misregistration during lamination caused drill hit to miss pad center, removing connection",
        "Dry film photoresist lift-off during development exposed trace to over-etch",
        "Excessive alkaline etchant concentration removed copper from thin neck region between pads",
        "Grain boundary corrosion of electroplated copper from inadequate rinsing after acid plating",
        "Laser drill energy overshoot ablated copper beneath dielectric, creating blind-via open",
    ],
    "short": [
        "Over-etching combined with solder mask misalignment left insufficient mask dam between pads",
        "Conductive ionic contamination from inadequate DI rinse created dendritic short under humidity",
        "Copper sliver left between pour and signal trace after CAM cleanup gap",
        "Insufficient solder mask tenting over via allowed copper migration to adjacent pad",
        "Electroless copper over-deposition bridged fine-pitch trace spacing during seed-layer step",
        "Photoresist scum between fine traces blocked etch, leaving residual copper bridge",
        "Copper foil wrinkle during lamination contacted adjacent trace in fine-pitch region",
        "Resin bleed into trace gap from aggressive cure temperature left carbonised conductive path",
    ],
    "mousebite": [
        "Aggressive etching with oscillating pH caused lateral undercut (mouse bite) on trace sidewalls",
        "Photoresist edge damage from particulate on exposure vacuum frame created notch defects",
        "Etch factor mismatch on high-current-density areas at panel center created edge scalloping",
        "Brush plating contact mark combined with etch left edge irregularity mimicking mouse bite",
        "Photomask pinhole allowed overexposure of resist edge, narrowing trace before etch",
        "Worn squeegee left resist thinning at trace edges, allowing lateral etch during subtractive process",
        "Panel bow during spray etch caused uneven nozzle impingement angle, increasing lateral undercut",
        "Copper microstructure coarsening from high-temperature bake weakened grain boundaries at trace edge",
    ],
    "spur": [
        "Incomplete etching due to photoresist scumming left isolated copper spur in clearance zone",
        "Broken trace fragment from prior panel handling adhered and was laminated as foreign copper",
        "Misaligned second-side artwork left copper spur extending into mask opening",
        "Partially developed resist island plated and etched into spur rather than clearing",
        "Etchant pooling in concave geometry caused delayed etch, leaving copper spur at acute corner",
        "Copper foil delamination flap re-adhered to adjacent clear zone during lamination",
        "Acid trap in CAD artwork created differential etch shadow leaving spur at re-entrant angle",
        "Insufficient agitation in etch tank left stagnant zone; copper spur survived in low-flow area",
    ],
    "copper": [
        "Pinhole in dry film photoresist allowed electroplated copper nodule in mask-free zone",
        "Resin smear not fully removed before desmear left conductive copper island after plating",
        "Copper foil burr from routing bit deposited and pressed into laminate during layup",
        "Acid trap in CAD geometry caused localized plating buildup appearing as spurious copper",
        "Foreign metal particle from tooling wear embedded in substrate and plated during copper seed step",
        "Over-plating in high-current-density corner regions from non-uniform current distribution",
        "Copper splash from mis-aimed plating bar strike landed and adhered in clear zone",
        "Residual copper from incompletely stripped test coupon contaminated adjacent panel area",
    ],
    "pin-hole": [
        "Void in electroless copper seed layer created pinhole that propagated through subsequent plate",
        "Entrapped air during panel entry into plating bath caused non-coating bubble defect",
        "Insufficient cleaning after micro-etch left organic residue blocking plating at thermal relief spoke",
        "Laser drill debris not fully removed before plating seeded pinhole at capture pad center",
        "Copper foil micro-pit from supplier quality issue propagated through subsequent plating steps",
        "Hydrogen evolution during acid copper plating created gas bubble adhering to surface, masking deposition",
        "Oil contamination from conveyor roller transferred to panel surface, inhibiting copper nucleation",
        "Over-etched micro-roughness on inner-layer copper caused laminate resin intrusion, blocking seed adhesion",
    ],
}

CORRECTIVE_ACTIONS: dict[DefectClass, list[str]] = {
    "open": [
        "Increased etch time monitoring; recalibrated spray pressure on etch line",
        "Enabled pulse plating and reduced panel rack density in plating tank",
        "Switched to laser depanelization and added keep-out routing near break tabs",
        "Tightened lamination press temperature profile and X-ray drill alignment check",
        "Reduced developer concentration and added pre-develop adhesion bake",
        "Installed in-line etch rate coupon monitoring and tightened ORP control band",
        "Extended rinse cascade duration and added DI water resistivity alarm above 1 MΩ·cm",
        "Recalibrated laser drill energy profile with fresh focus verification per panel batch",
    ],
    "short": [
        "Recalibrated LDI exposure dose and tightened solder mask develop time",
        "Extended final rinse cycles and installed in-line resistivity monitoring above 10 MΩ·cm",
        "Updated CAM DRC to enforce minimum sliver removal width of 0.15 mm",
        "Filled vias with epoxy and planarized before mask application",
        "Adjusted electroless copper bath loading and replenishment rate to control deposition thickness",
        "Added post-develop plasma descum step before copper etch",
        "Switched to controlled-tension foil handling and added wrinkle detection camera at layup",
        "Lowered lamination cure peak temperature and extended slow-ramp phase",
    ],
    "mousebite": [
        "Stabilized etchant ORP and switched to controlled spray nozzle pattern",
        "Added frame cleaning between panels and HEPA upgrade on clean room",
        "Rebalanced cathode/anode geometry and throttled current density",
        "Relocated bus bar contact points away from critical edge geometry",
        "Replaced photomask with fresh pellicle-protected set; added pinhole inspection before use",
        "Instituted squeegee hardness checks every 200 panels and reduced coating pressure",
        "Added panel-flatness measurement step before etch and adjusted conveyor tension",
        "Reduced post-plate bake temperature and switched to shorter, lower-temperature cure cycle",
    ],
    "spur": [
        "Added post-develop plasma descum step before etch",
        "Enhanced pre-lamination panel wipe and ionizer bars at layup",
        "Dual-sided alignment pins verified; LDI scaling compensation updated",
        "Developer temperature alarm limits tightened and agitation frequency verified",
        "Updated PCB design guidelines to eliminate acute interior angles below 45°; added DRC check",
        "Added foil adhesion peel test at incoming inspection and rejected warped foil lots",
        "CAM team added acid-trap angle correction rule to post-processing script",
        "Installed additional spray nozzle rows in etch machine low-flow zones; verified flow rate",
    ],
    "copper": [
        "Vacuum lamination pressure increased and film expiry tracking enforced",
        "Extended plasma desmear cycle and added permanganate desmear for high-aspect vias",
        "Routing bit change interval halved and post-route compressed air blow-off added",
        "CAM team revised acid trap angles and added teardrops on acute corners",
        "Installed tooling-wear monitoring via cutting-force sensors; added metal-particle in-line inspection",
        "Redesigned anode shielding to redistribute current density; adjusted brightener concentration",
        "Added plating-bar inspection jig; replaced worn anodes with fresh copper phosphorised sets",
        "Introduced panel isolation barriers between test coupons and production boards in rack",
    ],
    "pin-hole": [
        "Replenished electroless chemistry and reduced bath contamination via filtration",
        "Reduced entry angle and added cathode rod oscillation in plating tank",
        "Added alkaline cleaner stage and raised rinse conductivity setpoint",
        "Ultrasonic clean after laser drill and vacuum desmear before metallisation",
        "Implemented incoming foil inspection for micro-pit density; rejected out-of-spec lots",
        "Added leveler concentration control; installed H₂ bubble sweep agitation bars",
        "Cleaned and replaced conveyor rollers; added oil mist detector above conveyor path",
        "Reduced pre-plate micro-etch dwell time and added copper-foil surface roughness measurement",
    ],
}

OUTCOME_NOTES: dict[DefectClass, list[str]] = {
    "open": [
        "Etch rate verified hourly for one week; no recurrence on same lot style",
        "Cross-section microscopy on next 20 panels showed full barrel coverage",
        "AOI false opens at panel edge dropped to zero over 500 boards",
        "Registration drift held within 50 µm on subsequent inner layers",
        "Micro-etch weight loss returned to nominal range",
        "Continuity yield improved from 97.8% to 99.6% on following production run",
        "Backlight test of copper planes showed full coverage on 30-panel audit",
        "Blind via resistance measured within 5% of nominal across 100 samples",
    ],
    "short": [
        "Electrical test shorts between 0.4 mm pitch pads eliminated on validation lot",
        "HAST chamber retest passed after rinse SOP update",
        "CAM review checklist added; no slivers on next 30 designs",
        "X-ray inspection confirmed no bridging on treated via pairs",
        "Electroless copper bath thickness CV improved from 18% to 7%",
        "AOI short count per panel fell from 4.2 to 0.3 on production run",
        "Foil layup inspection log shows zero wrinkle escapes over 200 panels",
        "Delamination tests passed IPC-TM-650 2.4.8 criteria on 10 sample panels",
    ],
    "mousebite": [
        "Cross-sections showed uniform trace width within ±10% of design",
        "Particle count reduced; edge notch defects not seen in 200-panel run",
        "Impedance TDR sweep within spec on RF test coupons",
        "Gold finger thickness uniform; edge AOI clean after bus bar relocation",
        "Pinhole audit on 50 production masks showed zero escapes",
        "Trace-width SPC chart brought into control within three production shifts",
        "Panel flatness within 0.2 mm across board; edge AOI escapes dropped 90%",
        "Grain size metallography confirmed fine-grained plating on post-fix cross-sections",
    ],
    "spur": [
        "Optical inspection of clearance gaps clear on audit panels",
        "Foreign material FOD checks added to layup checklist; no FOD escapes in 90 days",
        "Registration offset reduced below 25 µm on double-sided lot",
        "Developer bath chemistry logged twice per shift; no spur recurrence",
        "Acute-angle spur escapes dropped to zero in DFM review on 15 subsequent designs",
        "Incoming foil lot acceptance rate improved; delamination-related spurs eliminated",
        "CAM script regression tests now cover 12 spur-inducing corner patterns",
        "Flow rate measurements in etch tank within ±5% across all zones; spur rate zero",
    ],
    "copper": [
        "Nodule count per panel dropped from 3 avg to 0.1 avg",
        "Via chain resistance measurements within 5% of simulation",
        "Tooling zone AOI clean on 100 consecutive panels",
        "Impedance coupons matched target within 2 ohms",
        "Metal particle audits show zero contamination on 50 panels since tooling upgrade",
        "Current distribution simulation confirmed uniformity within 8% across panel",
        "Plating bar inspection added to setup checklist; no splash events in 30-day audit",
        "Cross-panel copper thickness CV improved from 14% to 5% after rack redesign",
    ],
    "pin-hole": [
        "Backlight inspection of planes showed full coverage on sample set",
        "Plating thickness CV improved from 12% to 6%",
        "Peel strength test on thermal relief spokes passed IPC criteria",
        "Cross-section of blind vias showed continuous barrel without voids",
        "Incoming foil lot acceptance inspection added to supplier quality gate",
        "Pin-hole per-panel rate dropped from 2.1 to 0.05 after agitation upgrade",
        "Conveyor oil contamination monitor shows zero exceedances over 90-day period",
        "Copper surface roughness Ra within supplier spec on incoming inspection",
    ],
}

COMPONENT_TYPES: dict[DefectClass, list[str]] = {
    "open": [
        "signal trace (outer layer)",
        "PTH via barrel",
        "fine-pitch pad",
        "inner-layer trace",
        "BGA escape trace",
        "clock line",
        "USB differential pair",
        "reset net",
        "LED anode trace",
    ],
    "short": [
        "adjacent copper pads",
        "power plane clearance",
        "fine trace pair",
        "ground pour region",
        "via-in-pad",
        "QFN ground pad",
        "LVDS differential pair",
        "battery sense lines",
        "decoupling pad array",
    ],
    "mousebite": [
        "outer-layer trace edge",
        "differential pair",
        "RF microstrip",
        "connector finger",
        "DDR address line",
        "analog input trace",
        "I2C bus trace",
        "shield trace",
    ],
    "spur": [
        "ground isolation gap",
        "high-density routing channel",
        "soldermask-defined pad edge",
        "test pad array",
        "keep-out zone edge",
        "antenna feed gap",
        "high-voltage isolation slot",
        "fiducial vicinity",
    ],
    "copper": [
        "soldermask clearance",
        "inner-layer prepreg window",
        "panel tooling hole vicinity",
        "impedance control stripe",
        "RF ground ring",
        "castellation edge",
        "heatsink attach area",
        "ESD guard ring",
    ],
    "pin-hole": [
        "power plane",
        "large copper pour",
        "thermal relief spoke",
        "blind via capture pad",
        "backplane connector plane",
        "motor driver pour",
        "GND stitch via field",
        "heat spreader plane",
    ],
}


def generate_cases(count: int = TARGET_COUNT, seed: int = 42) -> HistoricalCaseCollection:
    rng = random.Random(seed)
    cases: list[HistoricalDefectCase] = []
    per_class = count // len(DEFECT_CLASSES)
    remainder = count % len(DEFECT_CLASSES)
    start_date = date(2021, 1, 1)

    case_num = 1
    for defect_class in DEFECT_CLASSES:
        n = per_class + (1 if remainder > 0 else 0)
        remainder -= 1 if remainder > 0 else 0

        root_causes = ROOT_CAUSES[defect_class]
        corrective_actions = CORRECTIVE_ACTIONS[defect_class]
        outcome_notes_pool = OUTCOME_NOTES[defect_class]
        component_pool = COMPONENT_TYPES[defect_class]

        # Track (root_cause, corrective_action) pairs to avoid exact duplicates.
        used_pairs: set[tuple[str, str]] = set()

        for _ in range(n):
            # Try to find a unique (root_cause, corrective_action) pair.
            root_cause = corrective_action = ""
            for _attempt in range(20):
                rc = rng.choice(root_causes)
                ca = rng.choice(corrective_actions)
                if (rc, ca) not in used_pairs:
                    root_cause, corrective_action = rc, ca
                    break
            else:
                # Exhausted retries — allow a repeat rather than infinite loop.
                root_cause = rng.choice(root_causes)
                corrective_action = rng.choice(corrective_actions)

            used_pairs.add((root_cause, corrective_action))

            outcome = rng.choice(outcome_notes_pool)
            component = rng.choice(component_pool)
            days_offset = rng.randint(0, 1400)
            severity = rng.choices([1, 2, 3, 4, 5], weights=[10, 25, 35, 20, 10])[0]

            cases.append(
                HistoricalDefectCase(
                    case_id=f"PCB-CASE-{case_num:04d}",
                    defect_class=defect_class,
                    component_type=component,
                    root_cause=root_cause,
                    corrective_action=corrective_action,
                    severity=severity,
                    date_recorded=start_date + timedelta(days=days_offset),
                    outcome_notes=outcome,
                )
            )
            case_num += 1

    rng.shuffle(cases)
    return HistoricalCaseCollection(cases=cases)


def write_cases(output_path: Path, count: int = TARGET_COUNT, seed: int = 42) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collection = generate_cases(count=count, seed=seed)
    payload = collection.model_dump(mode="json")
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(collection.cases)} cases to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic historical PCB defect cases")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=TARGET_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_cases(args.output, count=args.count, seed=args.seed)


if __name__ == "__main__":
    main()
