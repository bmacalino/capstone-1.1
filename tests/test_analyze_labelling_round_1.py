import csv
import json
import tempfile
import unittest
from pathlib import Path

from research.analyze_labelling_round_1 import analyze, build_parser

MASTER_FIELDS = [
    "row_id",
    "sentence_text",
    "is_blind",
    "llm_safety",
    "llm_defect",
    "llm_counterfeit",
    "stratum",
    "sampling_weight",
    "is_practice",
]
WORKSHEET_FIELDS = [
    "row_id",
    "sentence_text",
    "verifier_id",
    "verifier_action",
    "human_safety",
    "human_defect",
    "human_counterfeit",
    "human_hazard_subtype",
    "human_negated",
    "human_notes",
]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


class LabellingAnalysisTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.queue = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)

    def build(self, master_rows, a_rows, b_rows) -> dict:
        write_csv(self.queue / "queue_master.csv", MASTER_FIELDS, master_rows)
        write_csv(self.queue / "worksheet_A1.csv", WORKSHEET_FIELDS, a_rows)
        write_csv(self.queue / "worksheet_A2.csv", WORKSHEET_FIELDS, b_rows)
        args = build_parser().parse_args([])
        args.queue = self.queue
        args.master = self.queue / "queue_master.csv"
        args.worksheets = None
        args.output = None
        return analyze(args)

    def master(self, count: int, **overrides) -> list[dict[str, str]]:
        rows = []
        for index in range(1, count + 1):
            row = {
                "row_id": f"r{index}",
                "sentence_text": f"sentence {index}",
                "is_blind": "no",
                "llm_safety": "no",
                "llm_defect": "no",
                "llm_counterfeit": "no",
                "stratum": "seed_hit",
                "sampling_weight": "8.156",
                "is_practice": "no",
            }
            row.update({key: value[index - 1] for key, value in overrides.items()})
            rows.append(row)
        return rows

    def worksheet(self, verifier: str, safety: list[str]) -> list[dict[str, str]]:
        return [
            {
                "row_id": f"r{index}",
                "verifier_id": verifier,
                "verifier_action": "accept",
                "human_safety": value,
                "human_defect": "no",
                "human_counterfeit": "no",
            }
            for index, value in enumerate(safety, start=1)
        ]

    def test_perfect_agreement_gives_kappa_of_one(self):
        labels = ["yes", "yes", "no", "no"]
        report = self.build(
            self.master(4), self.worksheet("A1", labels), self.worksheet("A2", labels)
        )
        safety = report["inter_annotator_agreement"]["safety"]

        self.assertEqual(safety["doubly_labelled_rows"], 4)
        self.assertEqual(safety["raw_agreement"], 1.0)
        self.assertEqual(safety["cohens_kappa"], 1.0)
        self.assertEqual(safety["disagreement_rows"], 0)

    def test_chance_level_agreement_gives_kappa_of_zero(self):
        # A: yes yes no no / B: yes no yes no -> observed 0.5, expected 0.5, kappa 0.
        report = self.build(
            self.master(4),
            self.worksheet("A1", ["yes", "yes", "no", "no"]),
            self.worksheet("A2", ["yes", "no", "yes", "no"]),
        )
        safety = report["inter_annotator_agreement"]["safety"]

        self.assertEqual(safety["raw_agreement"], 0.5)
        self.assertEqual(safety["cohens_kappa"], 0.0)
        self.assertEqual(safety["disagreement_rows"], 2)
        self.assertEqual(sorted(report["adjudication_queue"]["safety"]), ["r2", "r3"])

    def test_kappa_is_undefined_when_both_annotators_use_one_class(self):
        labels = ["no", "no", "no", "no"]
        report = self.build(
            self.master(4), self.worksheet("A1", labels), self.worksheet("A2", labels)
        )
        safety = report["inter_annotator_agreement"]["safety"]

        self.assertIsNone(safety["cohens_kappa"])
        self.assertIn("one class", safety["kappa_note"])

    def test_only_agreed_rows_reach_consensus(self):
        report = self.build(
            self.master(4),
            self.worksheet("A1", ["yes", "yes", "no", "no"]),
            self.worksheet("A2", ["yes", "no", "yes", "no"]),
        )

        self.assertEqual(report["coverage"]["rows_with_two_labels"], 4)
        # r2 and r3 disagree on safety but agree on defect and counterfeit, so all
        # four rows carry consensus for some category...
        self.assertEqual(report["coverage"]["rows_with_consensus_on_any_category"], 4)
        # ...but only r1 and r4 agree on all three, which is the figure to quote.
        self.assertEqual(report["coverage"]["rows_agreed_on_all_categories"], 2)

    def test_keyword_screen_precision_counts_only_screened_positives(self):
        # The screen flags r1 and r2; people confirm r1 only -> precision 0.5.
        master = self.master(4, llm_safety=["yes", "yes", "no", "no"])
        labels = ["yes", "no", "no", "yes"]
        report = self.build(master, self.worksheet("A1", labels), self.worksheet("A2", labels))
        precision = report["keyword_screen_precision"]["safety"]

        self.assertEqual(precision["screen_positives_with_consensus"], 2)
        self.assertEqual(precision["human_confirmed"], 1)
        self.assertEqual(precision["precision"], 0.5)

    def test_prevalence_uses_only_the_random_all_stratum(self):
        master = self.master(
            4,
            stratum=["seed_hit", "seed_hit", "random_all", "random_all"],
            sampling_weight=["8.0", "8.0", "40.0", "40.0"],
        )
        labels = ["yes", "yes", "yes", "no"]
        report = self.build(master, self.worksheet("A1", labels), self.worksheet("A2", labels))
        prevalence = report["prevalence_random_all_only"]["safety"]

        # The two enriched positives are excluded; 1 of 2 random_all rows is positive.
        self.assertEqual(prevalence["labelled_rows"], 2)
        self.assertEqual(prevalence["positive_rows"], 1)
        self.assertEqual(prevalence["weighted_prevalence"], 0.5)

    def test_prevalence_reports_a_note_when_no_random_rows_are_labelled(self):
        report = self.build(
            self.master(2),
            self.worksheet("A1", ["yes", "no"]),
            self.worksheet("A2", ["yes", "no"]),
        )
        prevalence = report["prevalence_random_all_only"]["safety"]

        self.assertIsNone(prevalence["weighted_prevalence"])
        self.assertIn("not estimable", prevalence["note"])

    def test_anchoring_check_separates_blind_from_prelabelled_rows(self):
        master = self.master(4, is_blind=["yes", "yes", "no", "no"])
        labels = ["yes", "no", "yes", "yes"]
        report = self.build(master, self.worksheet("A1", labels), self.worksheet("A2", labels))

        self.assertEqual(report["anchoring_check"]["blind"]["safety"]["positive_rate"], 0.5)
        self.assertEqual(report["anchoring_check"]["pre_labelled"]["safety"]["positive_rate"], 1.0)

    def test_practice_rows_are_excluded(self):
        master = self.master(4, is_practice=["yes", "yes", "no", "no"])
        labels = ["yes", "yes", "yes", "yes"]
        report = self.build(master, self.worksheet("A1", labels), self.worksheet("A2", labels))

        self.assertEqual(report["coverage"]["queue_rows_total"], 2)

    def test_empty_worksheets_explain_what_to_do(self):
        # An untouched worksheet: every human_* cell blank, as generated.
        def blank(verifier: str) -> list[dict[str, str]]:
            return [
                {"row_id": f"r{index}", "verifier_id": verifier} for index in range(1, 5)
            ]

        with self.assertRaises(SystemExit) as caught:
            self.build(self.master(4), blank("A1"), blank("A2"))

        self.assertIn("still empty", str(caught.exception))

    def test_report_is_json_serializable(self):
        labels = ["yes", "no", "yes", "no"]
        report = self.build(
            self.master(4), self.worksheet("A1", labels), self.worksheet("A2", labels)
        )

        self.assertIsInstance(json.dumps(report), str)


if __name__ == "__main__":
    unittest.main()
