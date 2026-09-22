"""Audit who actually produced the existing labels in this project.

READ-ONLY. This script rewrites no dataset and changes no label. Its output is
evidence for the limitations section of the manuscript.

WHY IT EXISTS
Most labels in this repository were produced by a language model, but the
columns and filenames do not say so:

  * `annotator_1_id` is "Gemini_A1" / "Gemini_Auto"; `annotator_2_id` is
    "Codex Annotator 2" / "Codex A2 Product-Only".
  * Two annotation scripts write model output into columns literally named
    `human_risk_detected`, `human_risk_category` and `human_risk_severity`.
  * The frozen "gold test set" records two annotators, "Gemini" and "Gemini 2",
    whose labels are identical on every row of every field - the signature of a
    duplicated column, not of two independent judgements.

None of that is fraud; it is an artefact of moving fast with an LLM in the loop.
But it means accuracy measured on that test set is agreement with Gemini, not
accuracy against ground truth, and the write-up has to say so. This script
produces the numbers that let it say so precisely.

Run:
    .\\.venv\\Scripts\\python.exe .\\research\\audit_legacy_label_provenance.py

Writes:
    outputs/legacy_label_provenance_audit/audit.json
    outputs/legacy_label_provenance_audit/summary.md
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "legacy_label_provenance_audit"

# Substrings that identify a machine annotator. Matched case-insensitively
# against the annotator id recorded in the data.
MACHINE_MARKERS = (
    "gemini", "codex", "gpt", "claude", "llm", "_auto", "auto_",
    "heuristic", "ai_", "model", "first_pass",
)

# Datasets to inspect: (label, relative path, columns holding an annotator id).
DATASETS = [
    (
        "FiReCS risk round 1",
        "FiReCS_Risk_Annotation_Round_1_Annotated.csv",
        ["annotator_1_id", "annotator_2_id", "adjudicator_id"],
    ),
    (
        "FiReCS risk round 1 + annotator 2",
        "FiReCS_Risk_Annotation_Round_1_Annotator_2.xlsx",
        ["annotator_1_id", "annotator_2_id", "adjudicator_id"],
    ),
    (
        "FiReCS product-only final",
        "outputs/firecs_risk_annotation_final/FiReCS_Risk_Annotation_Round_1_Product_Only_Final.xlsx",
        ["annotator_1_id", "annotator_2_id", "adjudicator_id"],
    ),
    (
        "Lazada round 1 (Gemini pass)",
        "Lazada_Annotation_Round_1_Complete.xlsx",
        ["annotator_1_id", "annotator_2_id"],
    ),
    (
        "Lazada round 1 (fully annotated)",
        "Gemini_Lazada_Annotation_Round_1_Fully_Annotated.xlsx",
        ["annotator_1_id", "annotator_2_id"],
    ),
    (
        "Lazada round 1 adjudicated (canonical)",
        "outputs/lazada_gold_test_v1/lazada_round1_all_adjudicated.csv",
        ["annotator_id"],
    ),
    (
        "Lazada gold test v1 (FROZEN TEST SET)",
        "outputs/lazada_gold_test_v1/lazada_round1_gold_test_v1.csv",
        ["annotator_id"],
    ),
    (
        "Risk validation AI first pass",
        "outputs/risk_development_round_1/risk_validation_ai_first_pass.csv",
        ["status"],
    ),
    (
        "Manual marketplace reviews",
        "complete_manual_marketplace_reviews.csv",
        ["annotator_id"],
    ),
]

# Pairs of reviewer columns that are supposed to be independent judgements.
# Column pairs are named explicitly rather than guessed, so the check cannot
# quietly compare the wrong things and report a reassuring answer.
INDEPENDENCE_CHECKS = [
    {
        "label": "Gold test: completed adjudication (78 rows)",
        "path": "outputs/annotation_quality_round_1/Completed_annotate.csv",
        "reviewer_columns": ("Reviewer 1", "Reviewer 2"),
        "field_pairs": [
            ("R1 sentiment", "R2 sentiment"),
            ("R1 risk", "R2 risk"),
            ("R1 category", "R2 category"),
            ("R1 severity", "R2 severity"),
        ],
    },
    {
        "label": "Gold test: blind review (55 rows)",
        "path": "outputs/annotation_quality_round_1/lazada_round1_blind_review_adjudicated.csv",
        "reviewer_columns": ("reviewer", "reviewer_2"),
        "field_pairs": [
            ("human_sentiment", "human_sentiment_2"),
            ("human_risk_detected", "human_risk_detected_2"),
            ("human_risk_category", "human_risk_category_2"),
            ("human_risk_severity", "human_risk_severity_2"),
        ],
    },
]

# Annotator ids that read as human but whose rows come from a machine pass.
# This is the check that a plain value_counts cannot make: the frozen test set
# records "human_adjudicated_round1", which sounds like a person, but those rows
# are the Gemini/Gemini 2 reviews listed in INDEPENDENCE_CHECKS above.
MISLEADING_IDS = {
    "human_adjudicated_round1": "outputs/annotation_quality_round_1/Completed_annotate.csv",
    "blind_adjudicated_round1": "outputs/annotation_quality_round_1/lazada_round1_blind_review_adjudicated.csv",
    "human_audit": "outputs/risk_development_round_1/risk_validation_human_audit.csv",
    "first_pass_no_risk": "outputs/risk_development_round_1/risk_validation_ai_first_pass.csv",
    "reviewed_risk_positive": "outputs/risk_development_round_1/risk_validation_ai_first_pass.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        return pd.read_csv(path, skipinitialspace=True, low_memory=False)
    except Exception as error:  # pragma: no cover - reported, not raised
        print(f"  ! could not read {path.name}: {error}")
        return None


def _is_machine(value: str) -> bool:
    lowered = str(value).casefold()
    return any(marker in lowered for marker in MACHINE_MARKERS)


def audit_datasets() -> list[dict]:
    results = []
    for label, relative, columns in DATASETS:
        path = ROOT / relative
        entry: dict = {"dataset": label, "path": relative, "exists": path.is_file()}
        if not path.is_file():
            print(f"[skip] {label}: not found")
            results.append(entry)
            continue

        frame = _read(path)
        if frame is None:
            entry["error"] = "unreadable"
            results.append(entry)
            continue

        entry["sha256"] = _sha256(path)
        entry["rows"] = int(len(frame))
        entry["annotator_columns"] = {}
        machine_rows = 0
        for column in columns:
            if column not in frame.columns:
                continue
            counts = frame[column].value_counts(dropna=False)
            rendered = {str(key): int(value) for key, value in counts.items()}
            entry["annotator_columns"][column] = rendered
            machine_rows = max(
                machine_rows,
                sum(count for name, count in rendered.items() if _is_machine(name)),
            )
        entry["rows_with_machine_annotator"] = int(machine_rows)
        entry["machine_share"] = (
            round(machine_rows / len(frame), 4) if len(frame) else 0.0
        )
        print(
            f"[ok]   {label}: {entry['rows']} rows, "
            f"{machine_rows} with a machine annotator id"
        )
        results.append(entry)
    return results


def audit_independence() -> list[dict]:
    """Check whether two 'independent' reviewers ever actually disagreed.

    Two genuinely independent annotators always disagree somewhere. Perfect
    agreement across every field of every row means one opinion was recorded
    twice, which makes the resulting kappa meaningless.
    """
    results = []
    for check in INDEPENDENCE_CHECKS:
        label, relative = check["label"], check["path"]
        path = ROOT / relative
        entry: dict = {"check": label, "path": relative, "exists": path.is_file()}
        if not path.is_file():
            print(f"[skip] {label}: not found")
            results.append(entry)
            continue

        frame = _read(path)
        if frame is None:
            entry["error"] = "unreadable"
            results.append(entry)
            continue

        entry["sha256"] = _sha256(path)
        entry["rows"] = int(len(frame))
        entry["reviewers"] = {}
        for column in check["reviewer_columns"]:
            if column in frame.columns:
                entry["reviewers"][column] = {
                    str(key): int(value)
                    for key, value in frame[column].value_counts(dropna=False).items()
                }
        entry["both_reviewers_are_machines"] = bool(
            entry["reviewers"]
            and all(
                all(_is_machine(name) for name in values)
                for values in entry["reviewers"].values()
            )
        )

        agreement: dict[str, dict] = {}
        for left_column, right_column in check["field_pairs"]:
            if left_column not in frame.columns or right_column not in frame.columns:
                continue
            left = frame[left_column].astype(str).fillna("")
            right = frame[right_column].astype(str).fillna("")
            identical = int((left == right).sum())
            agreement[left_column] = {
                "compared_with": right_column,
                "identical_rows": identical,
                "total_rows": int(len(frame)),
                "identical_share": round(identical / len(frame), 4) if len(frame) else 0.0,
            }
        entry["field_agreement"] = agreement
        entry["perfect_agreement_on_every_field"] = bool(
            agreement
            and all(item["identical_share"] == 1.0 for item in agreement.values())
        )
        if entry["perfect_agreement_on_every_field"]:
            flag = "PERFECT AGREEMENT on every field - not independent"
        else:
            worst = min(item["identical_share"] for item in agreement.values()) if agreement else 0
            flag = f"labels vary (lowest field agreement {worst:.0%})"
        print(f"[ok]   {label}: {flag}")
        results.append(entry)
    return results


def audit_misleading_ids(datasets: list[dict]) -> list[dict]:
    """Flag annotator ids that read as human but come from a machine pass.

    A plain count of annotator ids makes the frozen test set look clean: its
    rows say "human_adjudicated_round1". Following that id back to the file it
    came from shows two machine reviewers. This is the check that catches it.
    """
    findings = []
    for entry in datasets:
        for column, counts in (entry.get("annotator_columns") or {}).items():
            for name, count in counts.items():
                if name in MISLEADING_IDS and not _is_machine(name):
                    findings.append(
                        {
                            "dataset": entry["dataset"],
                            "column": column,
                            "annotator_id": name,
                            "rows": count,
                            "actually_produced_by": MISLEADING_IDS[name],
                            "note": (
                                f"{count} rows record {name!r}, which reads as a human "
                                f"annotator. Those rows originate in "
                                f"{MISLEADING_IDS[name]}, whose reviewers are machine passes."
                            ),
                        }
                    )
                    print(
                        f"[flag] {entry['dataset']}: {count} rows labelled {name!r} "
                        f"trace back to a machine pass"
                    )
    return findings


def audit_scripts() -> list[dict]:
    """Find scripts that write model output into human-named columns."""
    # Two shapes occur in this repo: assignment into a row mapping
    # (row["human_risk_detected"] = label) and a dict literal
    # ({"human_risk_detected": label}). Catch both.
    human_write = re.compile(r"""["'](human_[a-z_]+)["']\s*(?:\]\s*=|:)""")
    source_marker = re.compile(r"""["'](codex_[a-z_]+|AI_First_Pass|gemini[a-z_]*)["']""", re.I)

    findings = []
    for path in sorted((ROOT / "research").glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        writes = human_write.findall(text)
        markers = source_marker.findall(text)
        if writes and markers:
            findings.append(
                {
                    "script": f"research/{path.name}",
                    "sha256": _sha256(path),
                    "writes_human_columns": sorted(set(writes)),
                    "machine_source_markers": sorted({marker.lower() for marker in markers}),
                }
            )
            print(f"[flag] research/{path.name}: writes human_* columns from a machine pass")
    return findings


def write_summary(report: dict) -> None:
    lines = [
        "# Legacy label provenance audit",
        "",
        f"Generated {report['generated_at']} by `research/audit_legacy_label_provenance.py`.",
        "This script is read-only; no dataset was modified.",
        "",
        "## Why this matters",
        "",
        "Accuracy measured against labels a language model produced is agreement",
        "with that model, not accuracy against ground truth. The numbers below",
        "quantify how much of this project's labelled data is in that position, so",
        "the limitations section can state it precisely instead of vaguely.",
        "",
        "## Machine-labelled share by dataset",
        "",
        "A 0% share here does NOT mean the labels are human. It means the recorded",
        "annotator id does not name a machine. See the next two sections - the",
        "frozen test set reads 0% here and is entirely machine-labelled.",
        "",
        "| Dataset | Rows | Rows with a machine annotator id | Share |",
        "| --- | ---: | ---: | ---: |",
    ]
    for entry in report["datasets"]:
        if not entry.get("exists") or "rows" not in entry:
            lines.append(f"| {entry['dataset']} | - | not found | - |")
            continue
        lines.append(
            f"| {entry['dataset']} | {entry['rows']} | "
            f"{entry['rows_with_machine_annotator']} | "
            f"{entry['machine_share']:.0%} |"
        )

    lines += [
        "",
        "## Were the two annotators independent?",
        "",
        "Two genuinely independent annotators always disagree somewhere. Perfect",
        "agreement on every field of every row means one opinion was recorded",
        "twice, and any inter-annotator statistic computed from it is meaningless.",
        "",
    ]
    for entry in report["independence"]:
        if not entry.get("exists"):
            lines.append(f"- **{entry['check']}**: file not found.")
            continue
        verdict = (
            "**identical on every field of every row - not independent**"
            if entry.get("perfect_agreement_on_every_field")
            else "labels vary between reviewers"
        )
        lines.append(f"- **{entry['check']}** ({entry.get('rows', 0)} rows): {verdict}")
        for field, item in (entry.get("field_agreement") or {}).items():
            lines.append(
                f"  - `{field}` vs `{item['compared_with']}`: "
                f"{item['identical_rows']}/{item['total_rows']} identical "
                f"({item['identical_share']:.0%})"
            )

    lines += [
        "",
        "## Annotator ids that read as human but are not",
        "",
        "A count of annotator ids alone makes the frozen test set look clean: its",
        "rows say `human_adjudicated_round1`. Following the id back to its source",
        "file is what reveals the problem.",
        "",
    ]
    if report["misleading_annotator_ids"]:
        for entry in report["misleading_annotator_ids"]:
            lines.append(
                f"- **{entry['dataset']}**: {entry['rows']} rows labelled "
                f"`{entry['annotator_id']}` originate in `{entry['actually_produced_by']}`, "
                "whose reviewers are machine passes."
            )
    else:
        lines.append("None found.")

    lines += [
        "",
        "## Scripts that write model output into human-named columns",
        "",
    ]
    if report["scripts"]:
        for entry in report["scripts"]:
            columns = ", ".join(f"`{name}`" for name in entry["writes_human_columns"])
            markers = ", ".join(f"`{name}`" for name in entry["machine_source_markers"])
            lines.append(
                f"- `{entry['script']}` writes {columns} and carries the "
                f"machine source markers {markers}."
            )
    else:
        lines.append("None found.")

    if report["unreadable"]:
        lines += [
            "",
            "## Files this audit could not read",
            "",
            "These are Excel workbooks. The project's Excel handling was written in",
            "Node, so the Python environment has no `openpyxl` and these rows are",
            "NOT included in the counts above. Install it and re-run for a complete",
            "audit: `.\\.venv\\Scripts\\python.exe -m pip install openpyxl`",
            "",
        ]
        for path in report["unreadable"]:
            lines.append(f"- `{path}`")

    lines += [
        "",
        "## What this does not say",
        "",
        "The partitioning discipline in this project is sound and was verified",
        "separately: the frozen test set has zero review-id, text and product",
        "overlap with the development set, and the training scripts assert that.",
        "The problem is the origin of the labels, not the split.",
        "",
        "## Recommended wording for the limitations section",
        "",
        "> Labels used during development were produced by a language model with",
        "> limited human adjudication. The frozen test set recorded two annotators,",
        "> but both were language-model passes whose labels are identical on every",
        "> row, so it does not provide independent ground truth. Results measured",
        "> against it are reported as agreement with that model rather than as",
        "> accuracy, and a human-verified test set was constructed for the badge",
        "> model evaluation.",
        "",
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Auditing label provenance (read-only)\n")

    datasets = audit_datasets()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "research/audit_legacy_label_provenance.py",
        "read_only": True,
        "machine_markers": list(MACHINE_MARKERS),
        "datasets": datasets,
        "independence": audit_independence(),
        "misleading_annotator_ids": audit_misleading_ids(datasets),
        "scripts": audit_scripts(),
        "unreadable": [
            entry["path"] for entry in datasets if entry.get("error") == "unreadable"
        ],
    }

    (OUTPUT_DIR / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_summary(report)

    readable = [item for item in report["datasets"] if "rows" in item]
    total_rows = sum(item["rows"] for item in readable)
    total_machine = sum(item["rows_with_machine_annotator"] for item in readable)
    print(
        f"\n{total_machine} of {total_rows} inspected rows carry a machine annotator id."
    )
    print(f"Wrote {OUTPUT_DIR / 'audit.json'}")
    print(f"Wrote {OUTPUT_DIR / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
