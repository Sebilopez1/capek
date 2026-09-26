"""QA-owned benchmark v2 for phase 2 scoring (plan P2-3): builder, evaluator (the only judge of the DoD 4 bars),
and real-like / corrupt fixtures. Test data only: the scorer never imports this package or reads ``gt.json``.

    python -m robot_report_card.bench build    --out DIR --seeds dev|heldout|file:PATH [--small]
    python -m robot_report_card.bench check    DIR                       # rrc score both sets + evaluate
    python -m robot_report_card.bench evaluate MIXED.json MIXED_GT.json --clean-only CLEAN.json CLEAN_GT.json
    python -m robot_report_card.bench realify  DATASET --out DIR
"""
