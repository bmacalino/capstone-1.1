"""User accounts, the guest scan limit, and each user's history and wishlist.

Kept in its own SQLite file (instance/accounts.db by default), separate from
defaketive.db, so the research database never holds names, emails or password
hashes. Passwords are stored only as Werkzeug hashes; the plain password is
never written anywhere.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from werkzeug.security import check_password_hash, generate_password_hash


GUEST_DAILY_SCANS = 3
ACTIVITY_LIMIT = 100
MAX_PRODUCT_BYTES = 200_000
MIN_PASSWORD_LENGTH = 8
MAX_NAME_LENGTH = 100
# Guest scans reset at midnight in the Philippines, not at midnight UTC.
MANILA = timezone(timedelta(hours=8))
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# Checked against when an email is unknown, so a wrong email takes as long as
# a wrong password and the response time does not reveal which accounts exist.
_DUMMY_HASH = generate_password_hash("defaketive-no-such-account")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS guest_scans (
    guest_key TEXT NOT NULL, day TEXT NOT NULL, used INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (guest_key, day)
);
CREATE TABLE IF NOT EXISTS user_activity (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    link TEXT NOT NULL, product_json TEXT NOT NULL, opened_at INTEGER NOT NULL,
    PRIMARY KEY (user_id, link)
);
CREATE TABLE IF NOT EXISTS user_wishlist (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    link TEXT NOT NULL, saved_at TEXT NOT NULL,
    PRIMARY KEY (user_id, link)
);
"""


class AccountError(ValueError):
    """A request the user can fix, with a message for one form field."""

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field
        self.message = message


class EmailTaken(AccountError):
    def __init__(self) -> None:
        super().__init__("email", "An account with that email already exists. Sign in instead.")


class WrongPassword(AccountError):
    def __init__(self) -> None:
        super().__init__("current_password", "That password is not correct.")


@contextmanager
def _connect(path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)
        _migrate(connection)
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _migrate(connection: sqlite3.Connection) -> None:
    """Add columns that newer versions need to a database made by an older one."""
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(users)").fetchall()}
    if "history_cleared_at" not in columns:
        # When the user last pressed "Clear scan history", in milliseconds.
        connection.execute("ALTER TABLE users ADD COLUMN history_cleared_at INTEGER NOT NULL DEFAULT 0")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def manila_day(now: datetime | None = None) -> str:
    """The calendar date in the Philippines, which is when guest scans reset."""
    return (now or datetime.now(timezone.utc)).astimezone(MANILA).date().isoformat()


def safe_link(value: Any) -> str:
    text = str(value or "").strip()
    return text if re.match(r"^https?://", text, re.IGNORECASE) and len(text) <= 2048 else ""


def _public(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {"id": int(row["id"]), "name": row["full_name"], "email": row["email"]}


def _clean_name(full_name: Any) -> str:
    name = " ".join(str(full_name or "").split())
    if not name:
        raise AccountError("full_name", "Enter your name.")
    if len(name) > MAX_NAME_LENGTH:
        raise AccountError("full_name", f"Use {MAX_NAME_LENGTH} characters or fewer.")
    return name


def _clean_email(email: Any) -> str:
    address = str(email or "").strip().lower()
    if not EMAIL_PATTERN.match(address) or len(address) > 254:
        raise AccountError("email", "Enter a valid email address, like you@example.com.")
    return address


def _check_new_password(password: Any, field: str = "password") -> str:
    text = str(password or "")
    if len(text) < MIN_PASSWORD_LENGTH:
        raise AccountError(field, f"Use at least {MIN_PASSWORD_LENGTH} characters.")
    return text


def signup_errors(full_name: Any, email: Any, password: Any, confirm: Any, agree: Any) -> dict[str, str]:
    """Every problem with a sign-up form at once, keyed by field."""
    errors: dict[str, str] = {}
    for field, check in (
        ("full_name", lambda: _clean_name(full_name)),
        ("email", lambda: _clean_email(email)),
        ("password", lambda: _check_new_password(password)),
    ):
        try:
            check()
        except AccountError as error:
            errors[field] = error.message
    if "password" not in errors and str(confirm or "") != str(password or ""):
        errors["confirm"] = "The passwords do not match."
    if agree is not True:
        errors["agree"] = "Please agree to the Terms of Use and Privacy Policy."
    return errors


def create_user(path: str | Path, full_name: Any, email: Any, password: Any) -> dict[str, Any]:
    name, address, secret = _clean_name(full_name), _clean_email(email), _check_new_password(password)
    with _connect(path) as connection:
        try:
            cursor = connection.execute(
                "INSERT INTO users(full_name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (name, address, generate_password_hash(secret), _now()),
            )
        except sqlite3.IntegrityError as error:
            raise EmailTaken() from error
        row = connection.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return _public(row)


def authenticate(path: str | Path, email: Any, password: Any) -> dict[str, Any] | None:
    """The user for this email and password, or None. Never says which one was wrong."""
    address = str(email or "").strip().lower()
    with _connect(path) as connection:
        row = connection.execute("SELECT * FROM users WHERE email = ?", (address,)).fetchone()
    stored = row["password_hash"] if row else _DUMMY_HASH
    matches = check_password_hash(stored, str(password or ""))
    return _public(row) if row and matches else None


def get_user(path: str | Path, user_id: Any) -> dict[str, Any] | None:
    try:
        key = int(user_id)
    except (TypeError, ValueError):
        return None
    with _connect(path) as connection:
        return _public(connection.execute("SELECT * FROM users WHERE id = ?", (key,)).fetchone())


def _require_password(connection: sqlite3.Connection, user_id: int, current_password: Any) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None or not check_password_hash(row["password_hash"], str(current_password or "")):
        raise WrongPassword()
    return row


def update_user(
    path: str | Path,
    user_id: int,
    current_password: Any,
    full_name: Any = None,
    email: Any = None,
    new_password: Any = None,
) -> dict[str, Any]:
    """Change name, email and/or password. The current password is always required."""
    with _connect(path) as connection:
        _require_password(connection, user_id, current_password)
        if full_name is not None:
            connection.execute("UPDATE users SET full_name = ? WHERE id = ?", (_clean_name(full_name), user_id))
        if email is not None:
            try:
                connection.execute("UPDATE users SET email = ? WHERE id = ?", (_clean_email(email), user_id))
            except sqlite3.IntegrityError as error:
                raise EmailTaken() from error
        if new_password not in (None, ""):
            secret = _check_new_password(new_password, "new_password")
            connection.execute("UPDATE users SET password_hash = ? WHERE id = ?", (generate_password_hash(secret), user_id))
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _public(row)


def delete_user(path: str | Path, user_id: int, current_password: Any) -> None:
    """Delete the account, its history and its wishlist (foreign keys cascade)."""
    with _connect(path) as connection:
        _require_password(connection, user_id, current_password)
        connection.execute("DELETE FROM users WHERE id = ?", (user_id,))


# --- Guest scan limit -------------------------------------------------------

def guest_scans_used(path: str | Path, guest_key: str, day: str) -> int:
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT used FROM guest_scans WHERE guest_key = ? AND day = ?", (guest_key, day)
        ).fetchone()
    return int(row["used"]) if row else 0


def reserve_guest_scan(path: str | Path, guest_key: str, day: str, limit: int = GUEST_DAILY_SCANS) -> bool:
    """Count one scan if the guest has one left today. One statement, so two
    scans started at the same moment cannot both take the last one."""
    with _connect(path) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO guest_scans(guest_key, day, used) VALUES (?, ?, 0)", (guest_key, day)
        )
        cursor = connection.execute(
            "UPDATE guest_scans SET used = used + 1 WHERE guest_key = ? AND day = ? AND used < ?",
            (guest_key, day, limit),
        )
        return cursor.rowcount == 1


def release_guest_scan(path: str | Path, guest_key: str, day: str) -> None:
    """Give a scan back when the job could not start."""
    with _connect(path) as connection:
        connection.execute(
            "UPDATE guest_scans SET used = used - 1 WHERE guest_key = ? AND day = ? AND used > 0",
            (guest_key, day),
        )


# --- History and wishlist ---------------------------------------------------

def _summary_json(link: str, product: Any) -> str:
    if not isinstance(product, dict) or safe_link(product.get("link")) != link:
        raise AccountError("product", "The product summary does not match its link.")
    # The full review list stays in the analysis store; the account keeps the
    # same compact summary the browser kept before accounts existed.
    summary = {key: value for key, value in product.items() if key != "comments"}
    payload = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    if len(payload.encode("utf-8")) > MAX_PRODUCT_BYTES:
        raise AccountError("product", "That product summary is too large to save.")
    return payload


def _prune(connection: sqlite3.Connection, user_id: int) -> None:
    connection.execute(
        """DELETE FROM user_activity
           WHERE user_id = ?
             AND link NOT IN (SELECT link FROM user_wishlist WHERE user_id = ?)
             AND link NOT IN (
                 SELECT link FROM user_activity WHERE user_id = ?
                 ORDER BY opened_at DESC LIMIT ?
             )""",
        (user_id, user_id, user_id, ACTIVITY_LIMIT),
    )


def _opened_at(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def record_activity(path: str | Path, user_id: int, link: Any, product: Any, opened_at: Any) -> None:
    """Save a product summary; a later open replaces an earlier one."""
    address = safe_link(link)
    if not address:
        raise AccountError("link", "That is not a valid product link.")
    payload, when = _summary_json(address, product), _opened_at(opened_at)
    with _connect(path) as connection:
        connection.execute(
            """INSERT INTO user_activity(user_id, link, product_json, opened_at) VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id, link) DO UPDATE SET
                   product_json = excluded.product_json,
                   opened_at = MAX(user_activity.opened_at, excluded.opened_at)""",
            (user_id, address, payload, when),
        )
        _prune(connection, user_id)


def clear_activity(path: str | Path, user_id: int, cleared_at: Any = None) -> None:
    """Settings -> Clear scan history. Wishlisted products keep their summary.

    The moment is remembered, so products checked before it stay out of this
    user's history on every browser, even though the app still holds them.
    """
    when = _opened_at(cleared_at) or int(datetime.now(timezone.utc).timestamp() * 1000)
    with _connect(path) as connection:
        connection.execute("UPDATE users SET history_cleared_at = ? WHERE id = ?", (when, user_id))
        connection.execute(
            """DELETE FROM user_activity WHERE user_id = ?
               AND link NOT IN (SELECT link FROM user_wishlist WHERE user_id = ?)""",
            (user_id, user_id),
        )
        connection.execute("UPDATE user_activity SET opened_at = 0 WHERE user_id = ?", (user_id,))


def set_wishlist(path: str | Path, user_id: int, link: Any, saved: bool) -> None:
    address = safe_link(link)
    if not address:
        raise AccountError("link", "That is not a valid product link.")
    with _connect(path) as connection:
        if saved:
            connection.execute(
                "INSERT OR IGNORE INTO user_wishlist(user_id, link, saved_at) VALUES (?, ?, ?)",
                (user_id, address, _now()),
            )
        else:
            connection.execute("DELETE FROM user_wishlist WHERE user_id = ? AND link = ?", (user_id, address))
            _prune(connection, user_id)


def library(path: str | Path, user_id: int) -> dict[str, Any]:
    with _connect(path) as connection:
        activity_rows = connection.execute(
            "SELECT link, product_json, opened_at FROM user_activity WHERE user_id = ? ORDER BY opened_at DESC",
            (user_id,),
        ).fetchall()
        wishlist_rows = connection.execute(
            "SELECT link FROM user_wishlist WHERE user_id = ? ORDER BY saved_at, rowid", (user_id,)
        ).fetchall()
        cleared = connection.execute("SELECT history_cleared_at FROM users WHERE id = ?", (user_id,)).fetchone()
    activity: dict[str, Any] = {}
    for row in activity_rows:
        try:
            product = json.loads(row["product_json"])
        except json.JSONDecodeError:
            continue
        activity[row["link"]] = {"product": product, "openedAt": int(row["opened_at"])}
    return {
        "activity": activity,
        "wishlist": [row["link"] for row in wishlist_rows],
        "clearedAt": int(cleared["history_cleared_at"]) if cleared else 0,
    }


def import_library(path: str | Path, user_id: int, activity: Any, wishlist: Any) -> dict[str, int]:
    """Merge a browser's guest history into the account. Bad entries are skipped,
    not fatal: an old browser cache must never block signing in."""
    imported = {"activity": 0, "wishlist": 0}
    entries = activity.items() if isinstance(activity, dict) else []
    for link, entry in list(entries)[: ACTIVITY_LIMIT * 2]:
        if not isinstance(entry, dict):
            continue
        try:
            record_activity(path, user_id, link, entry.get("product"), entry.get("openedAt"))
            imported["activity"] += 1
        except AccountError:
            continue
    for link in (wishlist if isinstance(wishlist, list) else [])[: ACTIVITY_LIMIT * 2]:
        try:
            set_wishlist(path, user_id, link, True)
            imported["wishlist"] += 1
        except AccountError:
            continue
    return imported


def popular_wishlist(path: str | Path, minimum: int = 2, limit: int = 8) -> list[tuple[str, int]]:
    """Links saved by at least `minimum` users, most saved first. Counts only:
    who saved them is never returned."""
    with _connect(path) as connection:
        rows = connection.execute(
            """SELECT link, COUNT(*) AS saves FROM user_wishlist
               GROUP BY link HAVING COUNT(*) >= ?
               ORDER BY saves DESC, MAX(saved_at) DESC LIMIT ?""",
            (minimum, limit),
        ).fetchall()
    return [(row["link"], int(row["saves"])) for row in rows]
