"""Turn completed hazard labelling worksheets into reportable statistics.

Each sentence is labelled independently by two of three annotators, so this
script joins the worksheets on row_id and reports:

  * Cohen's kappa per category, the standard measure of whether two people
    agreed beyond what chance alone would produce.
  * The precision of the keyword screen that pre-labelled the queue. This is
    the number the live lexicon tagger currently cannot claim.
  * Hazard prevalence, computed on the random_all stratum only and weighted by
    the sampling weights, because the queue is deliberately enriched and a raw
    rate across it would overstate prevalence several-fold.
  * An anchoring check comparing blind rows against pre-labelled rows, to test
    whether seeing the keyword guess pulled annotators toward it.
  * The rows where the two annotators disagreed, which is the adjudication
    queue for the third annotator.

Recall cannot be computed from the enriched strata and is reported only when
random_all rows are present.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_QUEUE = ROOT / "outputs" / "labelling_queue_round_1"
DEFAULT_OUTPUT = ROOT / "research" / "labelling_round_1_analysis.json"
CATEGORIES = ("safety", "defect", "counterfeit")
PREVALENCE_STRATUM = "random_all"
YES_VALUES = {"yes", "y", "1", "true"}
NO_VALUES = {"no", "n", "0", "false"}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a worksheet, accepting the .xlsx files annotators return from Excel.

    Empty spreadsheet cells arrive as NaN, so they are normalized to the empty
    string that parse_flag already treats as "not labelled".
    """
    if path.suffix.casefold() in {".xlsx", ".xlsm"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise SystemExit(
                f"Reading {path.name} needs pandas and openpyxl:\n"
                "  python -m pip install pandas openpyxl"
            ) from exc
        frame = pd.read_excel(path, sheet_name=0, dtype=str).fillna("")
        return frame.to_dict("records")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def parse_flag(value: str | None) -> bool | None:
    """Return True/False for a filled cell, or None when the annotator left it blank."""
    text = str(value or "").strip().casefold()
    if text in YES_VALUES:
        return True
    if text in NO_VALUES:
        return False
    return None


def load_worksheets(paths: list[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    """Return {row_id: {verifier_id: labels}} for every row an annotator filled in."""
    labelled: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in paths:
        for row in read_csv(path):
            row_id = str(row.get("row_id") or "").strip()
            if not row_id:
                continue
            flags = {category: parse_flag(row.get(f"human_{category}")) for category in CATEGORIES}
            if all(value is None for value in flags.values()):
                continue  # Untouched row.
            verifier = str(row.get("verifier_id") or path.stem).strip() or path.stem
            labelled[row_id][verifier] = {
                **flags,
                "action": str(row.get("verifier_action") or "").strip().casefold(),
                "subtype": str(row.get("human_hazard_subtype") or "").strip(),
                "negated": parse_flag(row.get("human_negated")),
                "notes": str(row.get("human_notes") or "").strip(),
            }
    return labelled


def agreement_statistics(
    labelled: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Cohen's kappa and raw agreement per category, over doubly-labelled rows."""
    from sklearn.metrics import cohen_kappa_score

    per_category: dict[str, Any] = {}
    disagreements: dict[str, list[str]] = {category: [] for category in CATEGORIES}

    for category in CATEGORIES:
        first: list[int] = []
        second: list[int] = []
        for row_id, verifiers in sorted(labelled.items()):
            names = sorted(verifiers)
            if len(names) < 2:
                continue
            a_value = verifiers[names[0]][category]
            b_value = verifiers[names[1]][category]
            if a_value is None or b_value is None:
                continue
            first.append(int(a_value))
            second.append(int(b_value))
            if a_value != b_value:
                disagreements[category].append(row_id)

        if not first:
            per_category[category] = {"doubly_labelled_rows": 0, "note": "no doubly-labelled rows"}
            continue

        matches = sum(1 for a, b in zip(first, second) if a == b)
        # Kappa is undefined when both annotators used a single class throughout.
        if len(set(first)) < 2 and len(set(second)) < 2:
            kappa: float | None = None
            kappa_note = "undefined: both annotators used one class only"
        else:
            kappa = round(float(cohen_kappa_score(first, second)), 4)
            kappa_note = None

        per_category[category] = {
            "doubly_labelled_rows": len(first),
            "raw_agreement": round(matches / len(first), 4),
            "cohens_kappa": kappa,
            "kappa_note": kappa_note,
            "positives_annotator_a": sum(first),
            "positives_annotator_b": sum(second),
            "disagreement_rows": len(disagreements[category]),
        }

    return {"per_category": per_category, "adjudication_queue": disagreements}


def consensus_labels(
    labelled: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, bool]]:
    """Keep only rows where both annotators agreed; disagreements need adjudication."""
    consensus: dict[str, dict[str, bool]] = {}
    for row_id, verifiers in labelled.items():
        agreed: dict[str, bool] = {}
        for category in CATEGORIES:
            values = [
                payload[category]
                for payload in verifiers.values()
                if payload[category] is not None
            ]
            if values and all(value == values[0] for value in values):
                agreed[category] = bool(values[0])
        if agreed:
            consensus[row_id] = agreed
    return consensus


def fully_confirmed_rows(labelled: dict[str, dict[str, dict[str, Any]]]) -> int:
    """Count rows where two annotators both labelled all three categories and agreed.

    Stricter than consensus_labels, which accepts a category labelled by only one
    annotator. This is the figure to quote as double-confirmed coverage.
    """
    total = 0
    for verifiers in labelled.values():
        names = sorted(verifiers)
        if len(names) < 2:
            continue
        first, second = verifiers[names[0]], verifiers[names[1]]
        if all(
            first[category] is not None
            and second[category] is not None
            and first[category] == second[category]
            for category in CATEGORIES
        ):
            total += 1
    return total


def keyword_screen_precision(
    master_rows: list[dict[str, str]], consensus: dict[str, dict[str, bool]]
) -> dict[str, Any]:
    """Measure how often the keyword screen's positives were confirmed by people."""
    per_category: dict[str, Any] = {}
    for category in CATEGORIES:
        flagged = 0
        confirmed = 0
        for row in master_rows:
            row_id = str(row.get("row_id") or "").strip()
            agreed = consensus.get(row_id)
            if agreed is None or category not in agreed:
                continue
            if parse_flag(row.get(f"llm_{category}")):
                flagged += 1
                confirmed += int(agreed[category])
        per_category[category] = {
            "screen_positives_with_consensus": flagged,
            "human_confirmed": confirmed,
            "precision": round(confirmed / flagged, 4) if flagged else None,
            "note": None if flagged else "no screened positives have consensus labels yet",
        }
    return per_category


def weighted_prevalence(
    master_rows: list[dict[str, str]], consensus: dict[str, dict[str, bool]]
) -> dict[str, Any]:
    """Estimate prevalence from the random_all stratum, weighted by sampling weight."""
    per_category: dict[str, Any] = {}
    for category in CATEGORIES:
        weight_total = 0.0
        weight_positive = 0.0
        rows = 0
        positives = 0
        for row in master_rows:
            if str(row.get("stratum") or "").strip() != PREVALENCE_STRATUM:
                continue
            agreed = consensus.get(str(row.get("row_id") or "").strip())
            if agreed is None or category not in agreed:
                continue
            try:
                weight = float(row.get("sampling_weight") or 0.0)
            except ValueError:
                weight = 0.0
            rows += 1
            weight_total += weight
            if agreed[category]:
                positives += 1
                weight_positive += weight
        per_category[category] = {
            "labelled_rows": rows,
            "positive_rows": positives,
            "weighted_prevalence": (
                round(weight_positive / weight_total, 6) if weight_total else None
            ),
            "note": (
                None
                if rows
                else f"no labelled rows in the {PREVALENCE_STRATUM} stratum; prevalence and recall are not estimable"
            ),
        }
    return per_category


def anchoring_check(
    master_rows: list[dict[str, str]], consensus: dict[str, dict[str, bool]]
) -> dict[str, Any]:
    """Compare blind rows against pre-labelled rows to detect anchoring bias."""
    buckets: dict[str, dict[str, list[int]]] = {
        "blind": {category: [] for category in CATEGORIES},
        "pre_labelled": {category: [] for category in CATEGORIES},
    }
    for row in master_rows:
        agreed = consensus.get(str(row.get("row_id") or "").strip())
        if agreed is None:
            continue
        bucket = "blind" if parse_flag(row.get("is_blind")) else "pre_labelled"
        for category in CATEGORIES:
            if category in agreed:
                buckets[bucket][category].append(int(agreed[category]))

    return {
        bucket: {
            category: {
                "labelled_rows": len(values),
                "positive_rate": round(sum(values) / len(values), 4) if values else None,
            }
            for category, values in categories.items()
        }
        for bucket, categories in buckets.items()
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    worksheet_paths = args.worksheets or sorted(args.queue.glob("worksheet_*_done.csv"))
    if not worksheet_paths:
        worksheet_paths = sorted(
            path for path in args.queue.glob("worksheet_*.csv") if "_master" not in path.name
        )
    if not worksheet_paths:
        raise SystemExit(f"No worksheets found in {args.queue}")

    master_rows = [row for row in read_csv(args.master) if not parse_flag(row.get("is_practice"))]
    labelled = load_worksheets(worksheet_paths)

    if not labelled:
        raise SystemExit(
            "No labelled rows found. The worksheets are still empty, so there is\n"
            "nothing to analyse yet. Fill in human_safety / human_defect /\n"
            "human_counterfeit following research/HAZARD_SENTENCE_TAGGING_GUIDE.md,\n"
            f"then re-run. Worksheets checked: {', '.join(p.name for p in worksheet_paths)}"
        )

    consensus = consensus_labels(labelled)
    agreement = agreement_statistics(labelled)
    stratum_counts: dict[str, int] = defaultdict(int)
    for row in master_rows:
        if str(row.get("row_id") or "").strip() in consensus:
            stratum_counts[str(row.get("stratum") or "unknown").strip()] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "worksheets": [path.name for path in worksheet_paths],
        "coverage": {
            "rows_with_any_label": len(labelled),
            "rows_with_two_labels": sum(1 for values in labelled.values() if len(values) >= 2),
            # A row counts here once the two annotators agree on ANY category, so
            # this is an upper bound. rows_agreed_on_all_categories is the figure
            # to quote: nearly every row agrees that some category is absent.
            "rows_with_consensus_on_any_category": len(consensus),
            "rows_agreed_on_all_categories": fully_confirmed_rows(labelled),
            "consensus_rows_by_stratum": dict(sorted(stratum_counts.items())),
            "queue_rows_total": len(master_rows),
        },
        "inter_annotator_agreement": agreement["per_category"],
        "adjudication_queue": {
            category: rows for category, rows in agreement["adjudication_queue"].items() if rows
        },
        "keyword_screen_precision": keyword_screen_precision(master_rows, consensus),
        "prevalence_random_all_only": weighted_prevalence(master_rows, consensus),
        "anchoring_check": anchoring_check(master_rows, consensus),
        "claim_boundary": (
            "Precision is measured on screened positives only. Recall and prevalence "
            "are valid only for the random_all stratum. Nothing here qualifies the "
            "hazard tagger for integration."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--master", type=Path)
    parser.add_argument("--worksheets", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.master is None:
        args.master = args.queue / "queue_master.csv"
    report = analyze(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Bytes, not write_text: on Windows text mode turns every LF into CRLF.
        args.output.write_bytes((rendered + "\n").encode("utf-8"))
        print(f"Wrote {args.output}")
    coverage = report["coverage"]
    print(
        f"{coverage['rows_agreed_on_all_categories']} of {coverage['rows_with_two_labels']} "
        f"doubly-labelled rows agree on all three categories."
    )
    for category, stats in report["inter_annotator_agreement"].items():
        if stats.get("doubly_labelled_rows"):
            print(
                f"  {category}: kappa={stats['cohens_kappa']}, "
                f"agreement={stats['raw_agreement']}, n={stats['doubly_labelled_rows']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
