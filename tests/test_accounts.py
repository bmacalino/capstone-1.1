import sqlite3
import tempfile
from contextlib import closing
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import accounts
import webapp
from tests.test_webapp import SAMPLE_PRODUCT, fire_product


def clean_product(link="https://shopee.ph/clean-i.1.4", reviews=40):
    """Enough reviews and nothing flagged."""
    return {
        **SAMPLE_PRODUCT,
        "link": link,
        "name": "Calm earbuds",
        "sentiment_summary": {
            **SAMPLE_PRODUCT["sentiment_summary"],
            "verdict": {"state": "no_warning_signs", "detail": "No flags in 40 reviews"},
            "usable_review_count": reviews,
        },
    }


SIGNUP = {
    "full_name": "Juan Dela Cruz",
    "email": "Juan@Example.com",
    "password": "correct horse",
    "confirm": "correct horse",
    "agree": True,
}
JOB = {"id": "job-1", "state": "running", "stage": 1, "platform": "shopee"}


class AccountsTestCase(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.db = str(Path(folder.name) / "accounts.db")
        webapp._catalogs.clear()
        webapp._jobs.clear()
        webapp._login_failures.clear()
        self.app = webapp.create_app({"TESTING": True, "SECRET_KEY": "test", "ACCOUNTS_DB": self.db})
        self.client = self.app.test_client()
        with self.client.session_transaction() as state:
            state["catalog_key"] = "accounts-test"
        webapp._catalogs["accounts-test"] = [SAMPLE_PRODUCT]

    def signup(self, client=None, **changes):
        return (client or self.client).post("/api/auth/signup", json={**SIGNUP, **changes})

    def live_scan(self, client=None):
        return (client or self.client).post(
            "/api/search", json={"query": "https://shopee.ph/x-i.1.9", "platform": "shopee", "live": True}
        )


class TestSignUpAndSignIn(AccountsTestCase):
    def test_sign_up_signs_you_in(self):
        response = self.signup()
        self.assertEqual(response.status_code, 201)
        me = self.client.get("/api/auth/me").json
        self.assertEqual(me["user"], {"name": "Juan Dela Cruz", "email": "juan@example.com"})
        self.assertIsNone(me["guest"])

    def test_duplicate_email_in_any_capitals_is_refused(self):
        self.signup()
        other = self.app.test_client()
        response = self.signup(other, email="JUAN@example.COM")
        self.assertEqual(response.status_code, 409)
        self.assertIn("email", response.json["fields"])

    def test_each_form_problem_is_named_by_field(self):
        response = self.signup(full_name=" ", email="not-an-email", password="short", confirm="short", agree=False)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(set(response.json["fields"]), {"full_name", "email", "password", "agree"})
        mismatch = self.signup(confirm="something else")
        self.assertEqual(mismatch.json["fields"], {"confirm": "The passwords do not match."})

    def test_form_requests_must_be_json(self):
        response = self.client.post("/api/auth/login", data={"email": "a@b.co", "password": "x"})
        self.assertEqual(response.status_code, 415)

    def test_password_is_stored_only_as_a_hash(self):
        self.signup()
        with closing(sqlite3.connect(self.db)) as connection:
            stored = connection.execute("SELECT password_hash FROM users").fetchone()[0]
            dump = "\n".join(connection.iterdump())
        self.assertNotIn("correct horse", dump)
        self.assertTrue(stored.startswith(("scrypt:", "pbkdf2:")))

    def test_wrong_password_and_unknown_email_get_the_same_answer(self):
        self.signup()
        self.client.post("/api/auth/logout", json={})
        wrong = self.client.post("/api/auth/login", json={"email": SIGNUP["email"], "password": "nope nope"})
        unknown = self.client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "nope nope"})
        self.assertEqual((wrong.status_code, unknown.status_code), (401, 401))
        self.assertEqual(wrong.json, unknown.json)

    def test_correct_password_signs_in_and_logout_signs_out(self):
        self.signup()
        self.client.post("/api/auth/logout", json={})
        self.assertIsNone(self.client.get("/api/auth/me").json["user"])
        response = self.client.post("/api/auth/login", json={"email": "juan@example.com", "password": "correct horse"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["user"]["name"], "Juan Dela Cruz")

    def test_five_failures_pause_sign_in_for_that_email(self):
        self.signup()
        for _ in range(webapp.LOGIN_FAILURE_LIMIT):
            self.client.post("/api/auth/login", json={"email": "juan@example.com", "password": "bad password"})
        paused = self.client.post("/api/auth/login", json={"email": "juan@example.com", "password": "correct horse"})
        self.assertEqual(paused.status_code, 429)


class TestGuestScanLimit(AccountsTestCase):
    def test_guest_gets_three_new_scans_a_day(self):
        with patch("webapp._start_job", return_value=JOB) as start:
            statuses = [self.live_scan().status_code for _ in range(4)]
        self.assertEqual(statuses, [202, 202, 202, 429])
        self.assertEqual(start.call_count, 3)
        refused = self.live_scan()
        self.assertTrue(refused.json["limit_reached"])
        self.assertEqual(refused.json["guest"]["left"], 0)
        self.assertEqual(self.client.get("/api/auth/me").json["guest"], {"limit": 3, "left": 0})

    def test_looking_up_checked_products_never_counts(self):
        for _ in range(5):
            response = self.client.post("/api/search", json={"query": "Sample gadget"})
            self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me").json["guest"]["left"], 3)

    def test_signed_in_users_are_not_limited(self):
        self.signup()
        with patch("webapp._start_job", return_value=JOB):
            statuses = {self.live_scan().status_code for _ in range(5)}
        self.assertEqual(statuses, {202})

    def test_a_scan_that_fails_to_start_is_given_back(self):
        with patch("webapp._start_job", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self.live_scan()
        self.assertEqual(self.client.get("/api/auth/me").json["guest"]["left"], 3)

    def test_a_new_day_in_manila_resets_the_count(self):
        # 23:30 UTC on 1 Oct is already 07:30 on 2 Oct in Manila.
        late = datetime(2026, 10, 1, 23, 30, tzinfo=timezone.utc)
        self.assertEqual(accounts.manila_day(late), "2026-10-02")
        for _ in range(3):
            self.assertTrue(accounts.reserve_guest_scan(self.db, "guest", "2026-10-02"))
        self.assertFalse(accounts.reserve_guest_scan(self.db, "guest", "2026-10-02"))
        self.assertTrue(accounts.reserve_guest_scan(self.db, "guest", "2026-10-03"))


class TestAccountChanges(AccountsTestCase):
    def setUp(self):
        super().setUp()
        self.signup()

    def test_changes_need_the_current_password(self):
        refused = self.client.patch("/api/auth/account", json={"full_name": "New", "current_password": "wrong pw"})
        self.assertEqual(refused.status_code, 403)
        changed = self.client.patch(
            "/api/auth/account",
            json={"full_name": "Juana", "email": "juana@example.com", "new_password": "new password", "current_password": "correct horse"},
        )
        self.assertEqual(changed.json["user"], {"name": "Juana", "email": "juana@example.com"})
        self.client.post("/api/auth/logout", json={})
        login = self.client.post("/api/auth/login", json={"email": "juana@example.com", "password": "new password"})
        self.assertEqual(login.status_code, 200)

    def test_cannot_move_to_an_email_another_account_uses(self):
        self.signup(self.app.test_client(), email="other@example.com")
        response = self.client.patch("/api/auth/account", json={"email": "Other@example.com", "current_password": "correct horse"})
        self.assertEqual(response.status_code, 409)

    def test_delete_account_removes_everything_and_signs_out(self):
        self.client.put("/api/me/activity", json={"link": SAMPLE_PRODUCT["link"], "product": SAMPLE_PRODUCT, "openedAt": 5})
        self.client.put("/api/me/wishlist", json={"link": SAMPLE_PRODUCT["link"]})
        response = self.client.delete("/api/auth/account", json={"current_password": "correct horse"})
        self.assertIsNone(response.json["user"])
        with closing(sqlite3.connect(self.db)) as connection:
            for table in ("users", "user_activity", "user_wishlist"):
                self.assertEqual(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0, table)
        again = self.client.post("/api/auth/login", json={"email": "juan@example.com", "password": "correct horse"})
        self.assertEqual(again.status_code, 401)


class TestLibrary(AccountsTestCase):
    def test_signed_out_calls_are_refused(self):
        self.assertEqual(self.client.get("/api/me/library").status_code, 401)
        self.assertEqual(self.client.put("/api/me/wishlist", json={"link": SAMPLE_PRODUCT["link"]}).status_code, 401)

    def test_history_keeps_a_summary_without_the_reviews(self):
        self.signup()
        self.client.put("/api/me/activity", json={"link": SAMPLE_PRODUCT["link"], "product": SAMPLE_PRODUCT, "openedAt": 1700})
        entry = self.client.get("/api/me/library").json["activity"][SAMPLE_PRODUCT["link"]]
        self.assertEqual(entry["openedAt"], 1700)
        self.assertEqual(entry["product"]["name"], "Sample gadget")
        self.assertNotIn("comments", entry["product"])

    def test_wishlist_add_and_remove(self):
        self.signup()
        self.client.put("/api/me/wishlist", json={"link": SAMPLE_PRODUCT["link"]})
        self.assertEqual(self.client.get("/api/me/library").json["wishlist"], [SAMPLE_PRODUCT["link"]])
        self.client.delete("/api/me/wishlist", json={"link": SAMPLE_PRODUCT["link"]})
        self.assertEqual(self.client.get("/api/me/library").json["wishlist"], [])

    def test_only_the_newest_hundred_are_kept_plus_the_wishlist(self):
        user = accounts.create_user(self.db, "A", "a@example.com", "password1")
        oldest = "https://shopee.ph/p-0"
        accounts.set_wishlist(self.db, user["id"], oldest, True)
        for index in range(accounts.ACTIVITY_LIMIT + 5):
            link = f"https://shopee.ph/p-{index}"
            accounts.record_activity(self.db, user["id"], link, {**SAMPLE_PRODUCT, "link": link}, index)
        kept = accounts.library(self.db, user["id"])["activity"]
        self.assertEqual(len(kept), accounts.ACTIVITY_LIMIT + 1)
        self.assertIn(oldest, kept)
        self.assertNotIn("https://shopee.ph/p-1", kept)

    def test_import_merges_and_the_newer_open_wins(self):
        self.signup()
        link = SAMPLE_PRODUCT["link"]
        self.client.put("/api/me/activity", json={"link": link, "product": SAMPLE_PRODUCT, "openedAt": 50})
        response = self.client.post(
            "/api/me/import",
            json={
                "activity": {
                    link: {"product": SAMPLE_PRODUCT, "openedAt": 10},
                    "javascript:alert(1)": {"product": SAMPLE_PRODUCT, "openedAt": 99},
                },
                "wishlist": [link, "ftp://nope"],
            },
        )
        self.assertEqual(response.json["imported"], {"activity": 1, "wishlist": 1})
        self.assertEqual(response.json["activity"][link]["openedAt"], 50)
        self.assertEqual(response.json["wishlist"], [link])

    def test_unsafe_links_are_refused(self):
        self.signup()
        response = self.client.put("/api/me/wishlist", json={"link": "javascript:alert(1)"})
        self.assertEqual(response.status_code, 400)

    def test_users_never_see_each_other(self):
        self.signup()
        self.client.put("/api/me/wishlist", json={"link": SAMPLE_PRODUCT["link"]})
        other = self.app.test_client()
        self.signup(other, email="other@example.com")
        self.assertEqual(other.get("/api/me/library").json, {"activity": {}, "wishlist": []})

    def test_clearing_history_keeps_the_wishlist(self):
        self.signup()
        for link in (SAMPLE_PRODUCT["link"], "https://shopee.ph/two"):
            self.client.put("/api/me/activity", json={"link": link, "product": {**SAMPLE_PRODUCT, "link": link}, "openedAt": 9})
        self.client.put("/api/me/wishlist", json={"link": "https://shopee.ph/two"})
        cleared = self.client.delete("/api/me/activity", json={}).json
        self.assertEqual(list(cleared["activity"]), ["https://shopee.ph/two"])
        self.assertEqual(cleared["activity"]["https://shopee.ph/two"]["openedAt"], 0)
        self.assertEqual(cleared["wishlist"], ["https://shopee.ph/two"])


class TestPublicNumbers(AccountsTestCase):
    def test_public_stats_are_the_real_counts(self):
        with patch("webapp.database_counts", return_value={"products": 12, "reviews": 3456}):
            stats = self.client.get("/api/stats/public").json
        self.assertEqual(stats, {"products": 12, "reviews": 3456, "marketplaces": 3})

    def test_picks_are_only_products_with_no_warning_signs(self):
        webapp._catalogs["accounts-test"] = [SAMPLE_PRODUCT, fire_product(), clean_product()]
        with patch("webapp.load_saved_products", return_value=[]):
            picks = self.client.get("/api/picks").json["products"]
        self.assertEqual([item["name"] for item in picks], ["Calm earbuds"])
        self.assertNotIn("comments", picks[0])

    def test_popular_needs_two_savers_and_never_names_them(self):
        webapp._catalogs["accounts-test"] = [clean_product(), fire_product()]
        clean_link, fire_link = clean_product()["link"], fire_product()["link"]
        for email in ("a@example.com", "b@example.com"):
            user = accounts.create_user(self.db, "Saver", email, "password1")
            accounts.set_wishlist(self.db, user["id"], clean_link, True)
        lone = accounts.create_user(self.db, "Lone", "c@example.com", "password1")
        accounts.set_wishlist(self.db, lone["id"], fire_link, True)
        with patch("webapp.load_saved_products", return_value=[]):
            response = self.client.get("/api/wishlist/popular")
        popular = response.json["products"]
        self.assertEqual([(item["product"]["link"], item["saves"]) for item in popular], [(clean_link, 2)])
        body = response.get_data(as_text=True)
        for private in ("a@example.com", "b@example.com", "user_id", "Saver"):
            self.assertNotIn(private, body)


if __name__ == "__main__":
    unittest.main()
