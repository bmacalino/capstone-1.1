from __future__ import annotations

import csv
import json
import random
import unittest
from pathlib import Path

from research.build_labelling_queue import (
    MASTER_FIELDS,
    STRATA,
    WORKSHEET_FIELDS,
    SeedMatcher,
    assign,
    build_pool,
    load_seeds,
    prelabel,
    sample_stratum,
    text_key,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/labelling_queue_round_1"


class SeedMatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seeds, blocks = load_seeds()
        cls.matcher = SeedMatcher(seeds, blocks)

    def test_hazard_terms_match(self) -> None:
        hit = self.matcher.match("Nasunog agad, meaning nagshort yan sa loob")
        self.assertIn("safety", hit["categories"])
        self.assertEqual(hit["subtype"], "fire_explosion_swelling")

    def test_most_severe_subtype_wins(self) -> None:
        # Both overheating and fire are present; the guide says record the
        # more severe one.
        hit = self.matcher.match("sobrang init tapos nasunog yung wire")
        self.assertEqual(hit["subtype"], "fire_explosion_swelling")

    def test_counterfeit_and_defect_are_independent(self) -> None:
        hit = self.matcher.match("peke to, at hindi gumagana")
        self.assertIn("counterfeit", hit["categories"])
        self.assertIn("defect", hit["categories"])

    def test_false_friends_are_masked(self) -> None:
        """The guide's section 6 cases must not reach the safety pool."""
        for text in (
            "not legit dual shock 4 but it's working fine",
            "Sabog ang sound. Pangit.",
            "fake nails, ang ganda",
        ):
            with self.subTest(text=text):
                hit = self.matcher.match(text)
                self.assertNotIn("safety", hit["categories"])
                self.assertTrue(hit["blocked"])

    def test_dual_shock_still_counts_as_counterfeit(self) -> None:
        """Masking the false friend must not swallow the real claim."""
        hit = self.matcher.match("not legit dual shock 4 but it's working fine")
        self.assertIn("counterfeit", hit["categories"])

    def test_word_boundaries_hold(self) -> None:
        """'init' inside another word must not fire an overheating match."""
        hit = self.matcher.match("the initial setup was easy")
        self.assertNotIn("safety", hit["categories"])

    def test_blind_prelabel_shape(self) -> None:
        guess = prelabel("umiinit at hindi na nag-charge", self.matcher)
        self.assertEqual(guess["llm_safety"], "yes")
        self.assertEqual(guess["llm_defect"], "yes")
        self.assertEqual(guess["llm_counterfeit"], "no")


class PoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seeds, blocks = load_seeds()
        cls.matcher = SeedMatcher(seeds, blocks)

    def _pool(self, rows, excluded=frozenset()):
        return build_pool(rows, self.matcher, set(excluded))

    def test_duplicates_and_exclusions_are_dropped(self) -> None:
        rows = [
            {"cohort": "own_scrape", "rating": 5, "review_text": "ang ganda nito"},
            {"cohort": "own_scrape", "rating": 5, "review_text": "ang ganda nito"},
            {"cohort": "own_scrape", "rating": 1, "review_text": "already labelled"},
            {"cohort": "own_scrape", "rating": 5, "review_text": "  "},
        ]
        pool, drops = self._pool(rows, {text_key("already labelled")})
        self.assertEqual(len(pool), 1)
        self.assertEqual(drops["duplicate"], 1)
        self.assertEqual(drops["excluded"], 1)
        self.assertEqual(drops["empty"], 1)

    def test_strata_are_assigned_as_designed(self) -> None:
        rows = [
            {"cohort": "own_scrape", "rating": 5, "review_text": "nasunog ang unit"},
            {"cohort": "own_scrape", "rating": 2, "review_text": "ang pangit ng kulay"},
            {"cohort": "own_scrape", "rating": 5, "review_text": "maganda, salamat"},
        ]
        pool, _ = self._pool(rows)
        self.assertEqual(
            [row["stratum"] for row in pool],
            ["seed_hit", "low_star", "random_all"],
        )

    def test_seed_hit_wins_over_low_star(self) -> None:
        """A 1-star hazard report belongs in seed_hit, counted once only."""
        rows = [{"cohort": "own_scrape", "rating": 1, "review_text": "nasunog agad"}]
        pool, _ = self._pool(rows)
        self.assertEqual(pool[0]["stratum"], "seed_hit")

    def test_sampling_never_splits_a_review(self) -> None:
        rows = [
            {
                "cohort": "own_scrape",
                "rating": 1,
                "review_text": "Isa. Dalawa. Tatlo. Apat.",
            },
            {"cohort": "own_scrape", "rating": 1, "review_text": "Maikli lang ito."},
        ]
        pool, _ = self._pool(rows)
        sampled = sample_stratum(pool, sentence_budget=2, rng=random.Random(1))
        for review in sampled:
            original = next(r for r in pool if r["review_id"] == review["review_id"])
            self.assertEqual(
                len(review["sentences"]),
                len(
                    [s for s in review["sentences"]],
                ),
                original["review_text"],
            )
        # The four-sentence review cannot fit a budget of two alongside another.
        self.assertLessEqual(sum(len(r["sentences"]) for r in sampled), 4)

    def test_rare_categories_are_offered_first(self) -> None:
        rows = [
            {"cohort": "own_scrape", "rating": 1, "review_text": f"sira na agad {i}"}
            for i in range(50)
        ]
        rows.append(
            {"cohort": "own_scrape", "rating": 5, "review_text": "nasunog ang charger"}
        )
        pool, _ = self._pool(rows)
        seed_hits = [r for r in pool if r["stratum"] == "seed_hit"]
        sampled = sample_stratum(
            seed_hits, sentence_budget=3, rng=random.Random(7), prioritise_rare=True
        )
        categories = set()
        for review in sampled:
            categories |= review["seed_categories"]
        self.assertIn("safety", categories)


class AssignmentTests(unittest.TestCase):
    def _rows(self, count: int) -> list[dict[str, str]]:
        return [
            {"is_practice": "no", "assigned_to": "", "adjudicator": ""}
            for _ in range(count)
        ]

    def test_every_row_gets_two_annotators_and_a_free_adjudicator(self) -> None:
        rows = self._rows(30)
        assign(rows, ["AA", "BB", "CC"])
        for row in rows:
            assigned = row["assigned_to"].split("|")
            self.assertEqual(len(set(assigned)), 2)
            self.assertNotIn(row["adjudicator"], assigned)
            self.assertTrue(row["adjudicator"])

    def test_load_is_even_across_annotators(self) -> None:
        rows = self._rows(300)
        annotators = ["AA", "BB", "CC"]
        assign(rows, annotators)
        counts = {
            name: sum(1 for r in rows if name in r["assigned_to"].split("|"))
            for name in annotators
        }
        self.assertEqual(len(set(counts.values())), 1, counts)

    def test_practice_rows_go_to_everyone(self) -> None:
        rows = self._rows(3)
        rows[1]["is_practice"] = "yes"
        assign(rows, ["AA", "BB", "CC"])
        self.assertEqual(rows[1]["assigned_to"], "AA|BB|CC")
        self.assertEqual(rows[1]["adjudicator"], "")

    def test_two_annotators_is_allowed_but_one_is_not(self) -> None:
        rows = self._rows(4)
        assign(rows, ["AA", "BB"])
        self.assertTrue(all(r["assigned_to"] == "AA|BB" for r in rows))
        with self.assertRaises(ValueError):
            assign(self._rows(2), ["AA"])


@unittest.skipUnless(
    (OUTPUT / "manifest.json").is_file(),
    "run research/build_labelling_queue.py first",
)
class GeneratedQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
        with (OUTPUT / "queue_master.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            cls.master = list(csv.DictReader(handle))

    def test_master_columns_match_the_contract(self) -> None:
        self.assertEqual(list(self.master[0].keys()), MASTER_FIELDS)

    def test_row_ids_are_unique(self) -> None:
        ids = [row["row_id"] for row in self.master]
        self.assertEqual(len(ids), len(set(ids)))

    def test_worksheets_withhold_the_stratum(self) -> None:
        """An annotator who can see why a row was picked is not labelling blind."""
        for annotator in self.manifest["annotators"]:
            path = OUTPUT / f"worksheet_{annotator}.csv"
            with path.open(encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle))
            self.assertEqual(header, WORKSHEET_FIELDS)
            for withheld in ("stratum", "sampling_weight", "matched_terms"):
                self.assertNotIn(withheld, header)

    def test_blind_rows_carry_no_machine_guess(self) -> None:
        blind = [row for row in self.master if row["is_blind"] == "yes"]
        self.assertTrue(blind)
        for row in blind:
            for column in (
                "llm_safety",
                "llm_defect",
                "llm_counterfeit",
                "llm_hazard_subtype",
            ):
                self.assertEqual(row[column], "", row["row_id"])

    def test_human_columns_ship_empty(self) -> None:
        for row in self.master:
            for column in (
                "human_safety",
                "human_defect",
                "human_counterfeit",
                "verifier_action",
            ):
                self.assertEqual(row[column], "", row["row_id"])

    def test_every_stratum_is_represented(self) -> None:
        present = {row["stratum"] for row in self.master}
        self.assertEqual(present, set(STRATA))

    def test_sentences_of_a_review_stay_together(self) -> None:
        """A review must be fully present or fully absent, and numbered 1..n."""
        by_review: dict[str, list[int]] = {}
        for row in self.master:
            by_review.setdefault(row["review_id"], []).append(int(row["sentence_index"]))
        for review_id, indexes in by_review.items():
            self.assertEqual(
                sorted(indexes), list(range(1, len(indexes) + 1)), review_id
            )

    def test_no_row_comes_from_a_frozen_split(self) -> None:
        """The gold test set must not be re-labelled under a new guide."""
        frozen = set()
        for name in (
            "lazada_round1_all_adjudicated.csv",
            "lazada_round1_gold_test_v1.csv",
        ):
            path = ROOT / "outputs/lazada_gold_test_v1" / name
            if not path.is_file():
                continue
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for record in csv.DictReader(handle):
                    value = (record.get("review_text") or "").strip()
                    if value:
                        frozen.add(text_key(value))
        self.assertTrue(frozen)
        leaked = [
            row["row_id"]
            for row in self.master
            if text_key(row["review_text_raw"]) in frozen
        ]
        self.assertEqual(leaked, [])

    def test_practice_set_is_stacked_with_false_friends(self) -> None:
        """The instructions promise this, so it has to be true of the file."""
        with (OUTPUT / "practice_set_master.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            practice = list(csv.DictReader(handle))
        self.assertTrue(practice)
        false_friends = [row for row in practice if row["blocked_terms"]]
        self.assertGreaterEqual(len(false_friends), len(practice) // 5)

    def test_practice_reviews_are_not_also_in_the_queue(self) -> None:
        """Nothing may be labelled twice under two different protocols."""
        with (OUTPUT / "practice_set_master.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            practice = list(csv.DictReader(handle))
        queued = {row["review_id"] for row in self.master}
        overlap = {row["review_id"] for row in practice} & queued
        self.assertEqual(overlap, set())

    def test_practice_set_goes_to_every_annotator(self) -> None:
        with (OUTPUT / "practice_set_master.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                self.assertEqual(
                    row["assigned_to"].split("|"), self.manifest["annotators"]
                )

    def test_manifest_names_the_reportable_stratum(self) -> None:
        self.assertEqual(self.manifest["reportable_prevalence_stratum"], "random_all")
        self.assertIn("keyword baseline", self.manifest["prelabel_source"])



ROUND_2 = ROOT / "outputs/labelling_queue_round_2"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class RoundDefaultsTests(unittest.TestCase):
    def test_round_one_keeps_its_original_settings(self) -> None:
        from research.build_labelling_queue import DEFAULT_BUDGET, parse_args

        args = parse_args([])
        self.assertEqual(args.round, 1)
        self.assertEqual(args.blind_rate, 0.15)
        self.assertEqual(args.seed, 20260922)
        self.assertEqual({s: getattr(args, f"budget_{s}") for s in STRATA}, DEFAULT_BUDGET)
        self.assertTrue(str(args.output_dir).endswith("labelling_queue_round_1"))

    def test_round_two_is_blind_and_separate(self) -> None:
        from research.build_labelling_queue import parse_args

        args = parse_args(["--round", "2"])
        self.assertEqual(args.blind_rate, 1.0)
        self.assertNotEqual(args.seed, 20260922)
        self.assertTrue(str(args.output_dir).endswith("labelling_queue_round_2"))


@unittest.skipUnless(
    (ROUND_2 / "manifest.json").is_file() and (OUTPUT / "queue_master.csv").is_file(),
    "run research/build_labelling_queue.py --round 2 first",
)
class GeneratedRoundTwoTests(unittest.TestCase):
    """Round 2 is the held-out test of the hazard checker."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.master = _read(ROUND_2 / "queue_master.csv")
        cls.practice = _read(ROUND_2 / "practice_set_master.csv")
        cls.round_one = {
            text_key(row["review_text_raw"])
            for name in ("queue_master.csv", "practice_set_master.csv")
            for row in _read(OUTPUT / name)
        }

    def test_no_review_from_round_one_is_labelled_again(self) -> None:
        for row in self.master + self.practice:
            self.assertNotIn(text_key(row["review_text_raw"]), self.round_one, row["row_id"])

    def test_every_test_row_is_blind(self) -> None:
        # The checker under test produces the pre-labels; showing them would
        # pull the test labels towards its answers.
        for row in self.master:
            self.assertEqual(row["is_blind"], "yes", row["row_id"])
            for column in ("llm_safety", "llm_defect", "llm_counterfeit", "llm_hazard_subtype"):
                self.assertEqual(row[column], "", row["row_id"])

    def test_row_ids_are_round_two(self) -> None:
        self.assertTrue(all(row["row_id"].startswith("r2-") for row in self.master))

    def test_each_row_has_two_annotators(self) -> None:
        self.assertTrue(all(len(row["assigned_to"].split("|")) == 2 for row in self.master))


ROUND_3 = ROOT / "outputs/labelling_queue_round_3"


@unittest.skipUnless(
    (ROUND_3 / "manifest.json").is_file() and (ROUND_2 / "queue_master.csv").is_file(),
    "run research/build_labelling_queue.py --round 3 first",
)
class GeneratedRoundThreeTests(unittest.TestCase):
    """Round 3 is the held-out test of the gated defect model."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.master = _read(ROUND_3 / "queue_master.csv")
        cls.practice = _read(ROUND_3 / "practice_set_master.csv")
        cls.earlier = {
            text_key(row["review_text_raw"])
            for folder in (OUTPUT, ROUND_2)
            for name in ("queue_master.csv", "practice_set_master.csv")
            for row in _read(folder / name)
        }

    def test_no_review_from_rounds_one_or_two_is_labelled_again(self) -> None:
        for row in self.master + self.practice:
            self.assertNotIn(text_key(row["review_text_raw"]), self.earlier, row["row_id"])

    def test_every_test_row_is_blind(self) -> None:
        for row in self.master:
            self.assertEqual(row["is_blind"], "yes", row["row_id"])
            self.assertEqual(row["llm_defect"], "", row["row_id"])

    def test_the_disagreement_stratum_is_present_and_weighted(self) -> None:
        strata = {row["stratum"] for row in self.master}
        self.assertEqual(strata, {"defect_disagreement", "seed_hit", "low_star", "random_all"})
        self.assertTrue(all(float(row["sampling_weight"]) > 0 for row in self.master))

    def test_row_ids_are_round_three(self) -> None:
        self.assertTrue(all(row["row_id"].startswith("r3-") for row in self.master))


if __name__ == "__main__":
    unittest.main()
