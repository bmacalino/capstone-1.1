"""Final retraining (v6): queue, features and scoring pieces."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from research.build_labelling_queue import (
    COUNTERFEIT_FOCUS,
    MODEL_DISAGREEMENT,
    ROUND_DEFAULTS,
    SeedMatcher,
    build_pool,
    load_seeds,
    load_shopee_hf,
    round_strata,
)
from research.measure_hazard_tagger import build_truth, read_adjudication, review_scores
from research.train_final_v6 import filter_threshold
from research.train_hazard_classifier import featurize


def sheet(folder: Path, name: str, fields: list[str], rows: list[dict[str, str]]) -> Path:
    path = folder / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


class RoundFiveQueueTests(unittest.TestCase):
    def test_round_five_draws_only_from_the_new_set(self) -> None:
        self.assertEqual(ROUND_DEFAULTS[5]["cohorts"], ("shopee_hf",))
        self.assertEqual(ROUND_DEFAULTS[5]["blind_rate"], 1.0)
        self.assertEqual(round_strata(5)[:2], (COUNTERFEIT_FOCUS, MODEL_DISAGREEMENT))

    def test_earlier_rounds_keep_their_strata(self) -> None:
        self.assertEqual(round_strata(3)[0], "defect_disagreement")
        self.assertNotIn(COUNTERFEIT_FOCUS, round_strata(3))

    def test_a_folder_of_splits_is_read_as_one_set(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            for name, text in (("train.csv", "sira agad ang charger"), ("test.csv", "ok naman po")):
                sheet(Path(folder), name, ["text", "label"], [{"text": text, "label": "0"}])
            rows, files = load_shopee_hf(Path(folder))
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(files), 2)
        self.assertEqual({row["rating"] for row in rows}, {1})

    def test_special_strata_are_tried_in_order(self) -> None:
        seeds, blocks = load_seeds()
        rows = [
            {"cohort": "shopee_hf", "review_text": "hindi original ang charger", "rating": 1},
            {"cohort": "shopee_hf", "review_text": "mabagal mag charge ang phone", "rating": 1},
        ]
        firsts = [("first", lambda text: "original" in text), ("second", lambda text: True)]
        pool, _ = build_pool(rows, SeedMatcher(seeds, blocks), set(), firsts)
        self.assertEqual([row["stratum"] for row in pool], ["first", "second"])


class TrainingFeatureTests(unittest.TestCase):
    def test_features_are_taken_before_the_counterfeit_rules(self) -> None:
        # "copy lang" is flagged only by rules c1, which run after the features.
        feats = featurize([{"text": "pero copy lang sya"}, {"text": "Definitely scam."}], need_embeddings=False)
        self.assertEqual(feats["kw"][0].tolist()[2], 0.0)
        self.assertEqual(feats["kw"][1].tolist()[2], 1.0)

    def test_the_safety_filter_prefers_the_lower_threshold_on_a_tie(self) -> None:
        import numpy as np

        y = np.array([1, 1, 0, 0])
        keyword = np.array([1, 1, 1, 0])
        probability = np.array([0.9, 0.8, 0.2, 0.95])
        threshold, stats = filter_threshold(y, keyword, probability)
        self.assertEqual(threshold, 0.8)
        self.assertEqual((stats["tp"], stats["fp"]), (2, 0))


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        folder = Path(self.folder.name)
        fields = ["row_id", "verifier_id", "human_safety", "human_defect", "human_counterfeit"]

        def answers(who: str, fake: list[str]) -> list[dict[str, str]]:
            return [
                {"row_id": f"r-{i}", "verifier_id": who, "human_safety": "no", "human_defect": "no", "human_counterfeit": fake[i]}
                for i in range(len(fake))
            ]

        self.sheets = [
            sheet(folder, "a.csv", fields, answers("A1", ["yes", "yes", "no", "no"])),
            sheet(folder, "b.csv", fields, answers("A2", ["yes", "no", "no", "no"])),
        ]
        self.decided = sheet(
            folder, "adj.csv", ["row_id", "adjudicated_counterfeit"],
            [{"row_id": "r-1", "adjudicated_counterfeit": "yes"}, {"row_id": "r-2", "adjudicated_counterfeit": "yes"}],
        )

    def test_a_disagreement_is_left_out_without_adjudication(self) -> None:
        truth = build_truth(self.sheets)
        self.assertNotIn("counterfeit", truth["r-1"])
        self.assertTrue(truth["r-0"]["counterfeit"])

    def test_adjudication_fills_only_the_disagreements(self) -> None:
        truth = build_truth(self.sheets, read_adjudication(self.decided))
        self.assertTrue(truth["r-1"]["counterfeit"])
        # r-2 was agreed "no"; an adjudication row for it changes nothing.
        self.assertFalse(truth["r-2"]["counterfeit"])

    def test_a_review_counts_when_any_sentence_does(self) -> None:
        truth = build_truth(self.sheets, read_adjudication(self.decided))
        master = {
            "r-0": {"review_id": "one"}, "r-1": {"review_id": "one"},
            "r-2": {"review_id": "two"}, "r-3": {"review_id": "two"},
        }
        predicted = {"r-0": set(), "r-1": {"counterfeit"}, "r-2": set(), "r-3": {"counterfeit"}}
        stats = review_scores(truth, predicted, master)["counterfeit"]
        self.assertEqual((stats["reviews"], stats["true_positives"], stats["false_positives"]), (2, 1, 1))

    def test_a_review_with_an_unsettled_sentence_is_left_out(self) -> None:
        truth = build_truth(self.sheets)
        master = {"r-1": {"review_id": "one"}, "r-2": {"review_id": "one"}}
        self.assertEqual(review_scores(truth, {}, master)["counterfeit"]["reviews"], 0)


if __name__ == "__main__":
    unittest.main()
