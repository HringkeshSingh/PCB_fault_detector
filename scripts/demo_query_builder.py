"""Demo: show QueryPlan output for the three Phase 3 specification test cases."""

import sys
sys.path.insert(0, ".")

from src.retrieval.query_builder import build_queries

CASES = [
    (
        "Case 1 — single detection (mousebite @ 0.82)",
        [{"defect_class": "mousebite", "confidence": 0.82}],
    ),
    (
        "Case 2 — multi-defect (open @ 0.71, short @ 0.38)",
        [
            {"defect_class": "open", "confidence": 0.71},
            {"defect_class": "short", "confidence": 0.38},
        ],
    ),
    (
        "Case 3 — zero detections",
        [],
    ),
]

for label, detections in CASES:
    plan = build_queries(detections)
    print("=" * 72)
    print(f"  {label}")
    print("=" * 72)
    print(f"  no_defects : {plan.no_defects}")
    print(f"  queries    : {len(plan.queries)}")
    for q in plan.queries:
        print()
        print(f"  [rank {q.rank}]")
        print(f"    defect_class   : {q.defect_class}")
        print(f"    confidence     : {q.confidence}  "
              f"band={q.confidence_band!r}  uncertain={q.uncertain}")
        print(f"    bbox           : {q.bbox}")
        print(f"    query_text     :")
        print(f"      {q.query_text}")
    print()
