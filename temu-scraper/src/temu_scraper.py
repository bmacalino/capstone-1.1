"""Visible-browser scraper for public Temu Philippines products and reviews.

Temu renders most catalog and review content in JavaScript and frequently
changes generated CSS class names. This scraper therefore prefers canonical
product IDs, JSON-LD, semantic attributes, accessible labels, and text-based
fallbacks. It never substitutes sample data when extraction fails.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import logging
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit

# Python 3.12+ compatibility for undetected-chromedriver 3.x.
try:  # pragma: no cover - host-version dependent
    import distutils  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    import setuptools._distutils as _distutils

    sys.modules["distutils"] = _distutils

import undetected_chromedriver as uc
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

DEFAULT_REVIEW_LIMIT = 100


class SafeChrome(uc.Chrome):
    """Suppress undetected-chromedriver's harmless Windows double-quit error."""

    def __del__(self):
        try:
            super().__del__()
        except OSError:
            pass


class TemuScraper:
    """Scrape public Temu product metadata and written customer reviews."""

    REVIEW_EXTRACTOR_VERSION = 9
    BASE_URL = "https://www.temu.com"
    MARKET_HOME = "https://www.temu.com/ph-en/"
    PRODUCT_LINK_SELECTORS = (
        'a[href*="goods.html"][href*="goods_id="]',
        'a[href*="-g-"][href*=".html"]',
        '[data-product-id] a[href]',
        '[data-goods-id] a[href]',
    )
    REVIEW_ROOT_SELECTORS = (
        '#reviewContent',
        '[data-testid*="review" i]',
        '[data-uniqid*="review" i]',
        'section[id*="review" i]',
        '[id*="review" i]',
    )

    def __init__(
        self,
        keyword: str = "",
        max_products: int = 10,
        review_limit: int = DEFAULT_REVIEW_LIMIT,
        pages: int = 1,
        product_urls: list[str] | None = None,
        all_reviews: bool = False,
        max_review_batches: int = 100,
        min_rating: float = 0.0,
        min_reviews: int = 0,
        include_empty: bool = False,
        chrome_user_data_dir: str | None = None,
        cookies_file: str | None = None,
        output: str | None = None,
        pause_for_verification: bool = True,
        verification_timeout: int = 0,
        pagination_retries: int = 3,
        pagination_timeout: float = 15.0,
        checkpoint_every: int = 5,
    ):
        raw_urls = product_urls or []
        normalized_urls = [self._normalize_product_url(url) for url in raw_urls]
        normalized_urls = [url for url in normalized_urls if url]
        if raw_urls and not normalized_urls:
            raise ValueError("No valid Temu product URLs were provided")
        if not keyword.strip() and not normalized_urls:
            raise ValueError("Provide a keyword or at least one Temu product URL")
        if not 0 <= min_rating <= 5:
            raise ValueError("min_rating must be between 0 and 5")

        self.driver = None
        self.keyword = keyword.strip()
        self.max_products = max(0, max_products)
        self.review_limit = max(0, review_limit)
        self.pages = max(1, pages)
        self.product_urls = normalized_urls
        self.product_navigation_urls = {}
        for raw_url in raw_urls:
            canonical_url = self._normalize_product_url(raw_url)
            if canonical_url:
                self.product_navigation_urls.setdefault(
                    canonical_url,
                    self._product_navigation_url(raw_url, canonical_url),
                )
        self.all_reviews = all_reviews
        self.max_review_batches = max(1, max_review_batches)
        self.min_rating = float(min_rating)
        self.min_reviews = max(0, min_reviews)
        self.include_empty = include_empty
        self.chrome_user_data_dir = chrome_user_data_dir
        self.pause_for_verification = pause_for_verification
        self.verification_timeout = max(0, int(verification_timeout))
        self.pagination_retries = max(1, pagination_retries)
        self.pagination_timeout = max(1.0, pagination_timeout)
        self.checkpoint_every = max(1, checkpoint_every)
        self._last_pagination_stop_reason = ""
        self.cookies_file = cookies_file or "cookies_temu_ph.dat"
        self._cookies_loaded = False
        self._network_reviews_by_product: dict[str, list[dict[str, Any]]] = {}
        self._review_request_context: dict[str, tuple[str, str]] = {}
        self._performance_logging_available = True

        slug_source = self.keyword or "products"
        slug = re.sub(r"[^a-z0-9_]+", "_", slug_source.casefold()).strip("_")
        self.out_file = output or f"temu_ph_{slug or 'products'}.json"
        self.output_data: dict[str, dict[str, Any]] = {}
        self._setup_logging()
        self.options = uc.ChromeOptions()
        self._configure_options()
        self._load_existing_data()

    @staticmethod
    def _product_navigation_url(raw_url, canonical_url):
        """Keep a localized Temu product path while discarding tracking data."""
        try:
            parts = urlsplit(str(raw_url or "").strip())
        except ValueError:
            return canonical_url
        host = (parts.hostname or "").casefold()
        if host in {"temu.com", "www.temu.com", "m.temu.com"} and re.search(
            r"-g-\d+\.html$", parts.path, re.I
        ):
            return urlunsplit(("https", "www.temu.com", parts.path, "", ""))
        return canonical_url

    def _setup_logging(self):
        os.makedirs("logs", exist_ok=True)
        if logging.getLogger().handlers:
            return
        stamp = datetime.datetime.now().strftime("temu_ph_%d_%m_%H_%M_%S.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(Path("logs") / stamp),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def _configure_options(self):
        if self.chrome_user_data_dir:
            profile = os.path.abspath(os.path.expanduser(self.chrome_user_data_dir))
            self.options.add_argument(f"--user-data-dir={profile}")
        if sys.platform.startswith("linux"):
            self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--start-maximized")
        self.options.add_argument("--lang=en-PH")
        self.options.add_argument("--no-first-run")
        self.options.add_argument("--no-default-browser-check")
        self.options.add_argument("--disable-default-apps")
        self.options.add_argument("--disable-search-engine-choice-screen")
        self.options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    @staticmethod
    def _detect_chrome_major():
        if not sys.platform.startswith("win"):
            return None
        roots = [
            os.path.join(os.environ.get("PROGRAMFILES", ""), "Google", "Chrome", "Application"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Google", "Chrome", "Application"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application"),
        ]
        versions = []
        for root in roots:
            if not os.path.isdir(root):
                continue
            for entry in os.listdir(root):
                if re.fullmatch(r"\d+(?:\.\d+){1,3}", entry):
                    versions.append(tuple(int(part) for part in entry.split(".")))
        return max(versions)[0] if versions else None

    @staticmethod
    def _cached_driver_for_major(chrome_major):
        if not chrome_major or not sys.platform.startswith("win"):
            return None
        path = os.path.join(
            os.environ.get("APPDATA", ""),
            "undetected_chromedriver",
            "undetected_chromedriver.exe",
        )
        if not os.path.isfile(path):
            return None
        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
            match = re.search(r"ChromeDriver\s+(\d+)", result.stdout)
            return path if match and int(match.group(1)) == chrome_major else None
        except (OSError, subprocess.SubprocessError):
            return None

    @classmethod
    def _normalize_product_url(cls, href):
        if not href:
            return ""
        absolute = urljoin(f"{cls.BASE_URL}/", str(href).strip())
        parts = urlsplit(absolute)
        host = parts.netloc.casefold().split(":")[0]
        if host not in {"temu.com", "www.temu.com", "m.temu.com"}:
            return ""
        query = parse_qs(parts.query)
        goods_id = next(iter(query.get("goods_id", [])), "")
        if not goods_id:
            match = re.search(r"-g-(\d+)\.html(?:$|[/?#])", parts.path, re.I)
            goods_id = match.group(1) if match else ""
        if not re.fullmatch(r"\d{6,}", str(goods_id)):
            return ""
        return urlunsplit(
            ("https", "www.temu.com", "/goods.html", urlencode({"goods_id": goods_id}), "")
        )

    def _build_search_url(self, page=1):
        query = {
            "search_key": self.keyword,
            "search_method": "user",
            "refer_page_el_sn": "200010",
        }
        if page > 1:
            query["page"] = str(page)
        return f"{self.BASE_URL}/ph-en/search_result.html?{urlencode(query)}"

    @staticmethod
    def _parse_compact_number(text):
        if text is None:
            return 0
        match = re.search(r"(\d[\d,.]*)\s*([kmb])?", str(text), re.I)
        if not match:
            return 0
        value, suffix = match.group(1), (match.group(2) or "").casefold()
        multiplier = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(suffix, 1)
        if suffix and "," in value and "." not in value:
            value = value.replace(",", ".")
        else:
            value = value.replace(",", "")
        try:
            return int(float(value) * multiplier)
        except ValueError:
            return 0

    @staticmethod
    def _parse_rating(text):
        match = re.search(
            r"(?<!\d)([0-5](?:\.\d+)?)\s*(?:/\s*5|out\s+of\s+(?:five|5)|stars?)?",
            str(text or ""),
            re.I,
        )
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def _clean_text(text):
        value = str(text or "")
        if any(marker in value for marker in ("Ã¢", "Ã‚")):
            for source_encoding in ("cp1252", "latin-1"):
                try:
                    value = value.encode(source_encoding).decode("utf-8")
                    break
                except (UnicodeEncodeError, UnicodeDecodeError):
                    continue
        return re.sub(r"\s+", " ", value.replace("\u00a0", " ")).strip()

    @classmethod
    def _normalize_product_name(cls, value):
        name = cls._clean_text(value)
        return re.sub(r"\s*(?:\||-|–)\s*Temu(?:\s+Philippines)?.*$", "", name, flags=re.I).strip()

    @classmethod
    def _price_from_page_source(cls, page_source):
        """Read Temu's structured price fallback when no semantic DOM price exists."""
        source = str(page_source or "")
        for field in (
            "minToMaxPriceStr",
            "minOnSalePriceStr",
            "normalPriceStr",
        ):
            match = re.search(rf'"{field}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', source)
            if not match:
                continue
            try:
                value = json.loads(f'"{match.group(1)}"')
            except json.JSONDecodeError:
                value = match.group(1)
            value = cls._clean_text(value)
            if len(value) < 80 and re.search(r"\d", value):
                return value
        return ""

    @classmethod
    def _first_text(cls, root, selectors):
        for selector in selectors:
            try:
                for element in root.find_elements(By.CSS_SELECTOR, selector):
                    text = cls._clean_text(element.text)
                    if text:
                        return text
            except (NoSuchElementException, AttributeError):
                continue
        return ""

    @staticmethod
    def _first_attribute(root, selectors, attribute):
        for selector in selectors:
            try:
                for element in root.find_elements(By.CSS_SELECTOR, selector):
                    value = element.get_attribute(attribute)
                    if value:
                        return value
            except (NoSuchElementException, AttributeError):
                continue
        return ""

    def _save_debug_artifacts(self, kind):
        if not self.driver:
            return
        try:
            os.makedirs("debug", exist_ok=True)
            Path("debug", f"temu_ph_{kind}.html").write_text(
                self.driver.page_source, encoding="utf-8"
            )
            self.driver.save_screenshot(str(Path("debug", f"temu_ph_{kind}.png")))
            logging.warning("Saved %s diagnostics for %s", kind, self.driver.current_url)
        except Exception as exc:
            logging.warning("Could not save diagnostics: %s", exc)

    def _save_cookies(self):
        if not self.driver:
            return
        temporary_path = f"{self.cookies_file}.tmp"
        try:
            cookies = self.driver.get_cookies()
            if not cookies:
                logging.warning("Temu returned no cookies; keeping the previous session file")
                return
            with open(temporary_path, "wb") as file:
                pickle.dump(cookies, file)
            os.replace(temporary_path, self.cookies_file)
            logging.info("Saved %s Temu session cookie(s)", len(cookies))
        except Exception as exc:
            logging.warning("Could not save Temu cookies: %s", exc)
        finally:
            if os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass

    def _load_cookies(self):
        if self._cookies_loaded:
            return True
        if not os.path.exists(self.cookies_file):
            return False
        if os.path.getsize(self.cookies_file) == 0:
            logging.warning("The Temu session cookie file is empty; manual login is required")
            return False
        try:
            with open(self.cookies_file, "rb") as file:
                cookies = pickle.load(file)
            loaded = 0
            for cookie in cookies:
                cookie.pop("sameSite", None)
                try:
                    self.driver.add_cookie(cookie)
                    loaded += 1
                except Exception:
                    continue
            self._cookies_loaded = loaded > 0
            if self._cookies_loaded:
                logging.info("Restored %s Temu session cookie(s)", loaded)
            return self._cookies_loaded
        except Exception as exc:
            logging.warning("Could not load Temu cookies: %s", exc)
            return False

    @staticmethod
    def _network_field(record, names):
        normalized = {
            re.sub(r"[^a-z0-9]", "", str(key).casefold()): value
            for key, value in record.items()
        }
        for name in names:
            value = normalized.get(re.sub(r"[^a-z0-9]", "", name.casefold()))
            if value not in (None, "", [], {}):
                return value
        return ""

    @classmethod
    def _network_review_time(cls, value):
        """Convert Temu epoch timestamps to the same readable date used by the DOM."""
        cleaned = cls._clean_text(value)
        if not re.fullmatch(r"\d{10,13}", cleaned):
            return cleaned
        try:
            timestamp = int(cleaned)
            if timestamp >= 10_000_000_000:
                timestamp /= 1000
            parsed = datetime.datetime.fromtimestamp(
                timestamp, tz=datetime.timezone.utc
            )
        except (OSError, OverflowError, ValueError):
            return cleaned
        return f"on {parsed.strftime('%b')} {parsed.day}, {parsed.year}"

    @classmethod
    def _network_record_product_ids(cls, record):
        ids = set()
        for key, value in record.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if normalized not in {"goodsid", "itemid", "productid"}:
                continue
            match = re.search(r"\d{6,}", str(value or ""))
            if match:
                ids.add(match.group(0))
        return ids

    @classmethod
    def _review_from_network_record(cls, record, expected_goods_id=""):
        """Normalize one JSON review object without accepting product cards."""
        if not isinstance(record, dict):
            return None
        normalized_keys = {
            re.sub(r"[^a-z0-9]", "", str(key).casefold()) for key in record
        }
        explicit_review_content_keys = {
            "commentcontent",
            "reviewcontent",
            "reviewtext",
            "comment",
        }
        review_metadata_keys = {
            "commentid",
            "reviewid",
            "reviewtime",
            "createtime",
            "createdat",
            "username",
            "nickname",
            "author",
        }
        has_explicit_review_content = bool(
            normalized_keys & explicit_review_content_keys
        )
        has_contextual_review_content = (
            "content" in normalized_keys
            and bool(normalized_keys & review_metadata_keys)
        )
        if not has_explicit_review_content and not has_contextual_review_content:
            return None
        record_ids = cls._network_record_product_ids(record)
        if expected_goods_id and record_ids and expected_goods_id not in record_ids:
            return None

        content = cls._network_field(
            record,
            (
                "comment_content",
                "commentContent",
                "review_content",
                "reviewContent",
                "review_text",
                "reviewText",
                "comment",
                "content",
            ),
        )
        review_language = cls._network_field(record, ("review_lang", "reviewLang"))
        if isinstance(review_language, dict):
            content = cls._network_field(
                review_language,
                ("translate_comment", "translateComment", "translated_comment"),
            ) or content
        rating_value = cls._network_field(
            record,
            (
                "star_rating",
                "starRating",
                "goods_score",
                "goodsScore",
                "rating",
                "stars",
                "star",
                "score",
            ),
        )
        rating = cls._individual_review_rating(cls._parse_rating(rating_value))
        content = cls._written_review_content(content)
        if not rating or not content:
            return None

        author = cls._network_field(
            record,
            (
                "user_name",
                "userName",
                "username",
                "nickname",
                "nick_name",
                "author",
                "name",
                "user_info",
                "userInfo",
                "user",
            ),
        )
        if isinstance(author, dict):
            author = cls._network_field(author, ("name", "nickname", "username"))
        review_time = cls._network_field(
            record,
            (
                "review_time",
                "reviewTime",
                "create_time",
                "createTime",
                "created_at",
                "createdAt",
                "concat_time_lang",
                "concatTimeLang",
                "date",
                "time",
            ),
        )
        variation = cls._network_field(
            record,
            ("variation", "sku_info", "skuInfo", "spec", "product_spec"),
        )
        if isinstance(variation, (dict, list)):
            variation = json.dumps(variation, ensure_ascii=False, sort_keys=True)
        helpful = cls._network_field(
            record,
            ("helpful_count", "helpfulCount", "like_count", "likeCount", "likes"),
        )
        comment_id = cls._network_field(
            record,
            ("comment_id", "commentId", "review_id", "reviewId", "id"),
        )
        seller_response = cls._network_field(
            record,
            ("seller_response", "sellerResponse", "reply", "response"),
        )
        if isinstance(seller_response, (dict, list)):
            seller_response = json.dumps(
                seller_response, ensure_ascii=False, sort_keys=True
            )
        return {
            "comment_id": cls._clean_text(comment_id),
            "author": cls._clean_text(author),
            "rating": rating,
            "time": cls._network_review_time(review_time),
            "content": content,
            "variation": cls._clean_text(variation),
            "seller_respond": cls._clean_text(seller_response),
            "like_count": cls._parse_compact_number(helpful),
            "extraction_source": "network",
        }

    @classmethod
    def _reviews_from_network_payload(cls, payload, expected_goods_id=""):
        reviews = []
        stack = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                review = cls._review_from_network_record(item, expected_goods_id)
                if review:
                    reviews.append(review)
                stack.extend(reversed(tuple(item.values())))
            elif isinstance(item, list):
                stack.extend(reversed(item))
        return cls._deduplicate_reviews(reviews)

    def _drain_performance_logs(self):
        if not self._performance_logging_available or not self.driver:
            return []
        try:
            return self.driver.get_log("performance") or []
        except Exception as exc:
            self._performance_logging_available = False
            logging.warning("Chrome performance logging is unavailable: %s", exc)
            return []

    def _reset_product_network_capture(self, product):
        goods_id = str(product.get("goods_id") or "")
        self._network_reviews_by_product[goods_id] = []
        self._review_request_context = {}
        self._drain_performance_logs()

    def _capture_review_network(self, product):
        """Collect valid reviews from Temu's own browser Fetch/XHR responses."""
        goods_id = str(product.get("goods_id") or "")
        if not goods_id:
            return []
        captured = list(self._network_reviews_by_product.get(goods_id, []))
        for entry in self._drain_performance_logs():
            try:
                message = json.loads(entry.get("message", "{}"))["message"]
                method = message.get("method")
                if method == "Network.requestWillBeSent":
                    params = message.get("params", {})
                    request = params.get("request", {})
                    request_url = str(request.get("url") or "")
                    if re.search(r"review|comment", request_url, re.I):
                        self._review_request_context[params.get("requestId", "")] = (
                            request_url,
                            str(request.get("postData") or ""),
                        )
                    continue
                if method != "Network.responseReceived":
                    continue
                params = message.get("params", {})
                if params.get("type") not in {"Fetch", "XHR"}:
                    continue
                response = params.get("response", {})
                response_url = str(response.get("url") or "")
                if not re.search(r"review|comment", response_url, re.I):
                    continue
                request_url, post_data = self._review_request_context.get(
                    params.get("requestId", ""), (response_url, "")
                )
                request_goods_ids = set(
                    re.findall(
                        r"(?i)(?:goods|item|product)(?:_|%5f)?id\D{0,12}(\d{6,})",
                        f"{request_url} {post_data}",
                    )
                )
                if request_goods_ids and goods_id not in request_goods_ids:
                    continue
                body_result = self.driver.execute_cdp_cmd(
                    "Network.getResponseBody",
                    {"requestId": params["requestId"]},
                )
                raw_body = body_result.get("body", "")
                if body_result.get("base64Encoded"):
                    raw_body = base64.b64decode(raw_body).decode(
                        "utf-8", errors="replace"
                    )
                payload = json.loads(raw_body)
                response_reviews = self._reviews_from_network_payload(
                    payload, expected_goods_id=goods_id
                )
                if response_reviews:
                    logging.info(
                        "Captured %s valid Temu review(s) from %s",
                        len(response_reviews),
                        urlsplit(response_url).path,
                    )
                    captured.extend(response_reviews)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            except Exception:
                continue
        captured = self._deduplicate_reviews(captured)
        self._network_reviews_by_product[goods_id] = captured
        return list(captured)

    def _load_existing_data(self):
        if not os.path.exists(self.out_file):
            return
        try:
            with open(self.out_file, "r", encoding="utf-8") as file:
                records = json.load(file)
            for item in records:
                if not isinstance(item, dict):
                    continue
                item["name"] = self._normalize_product_name(item.get("name"))
                item["description"] = self._clean_text(item.get("description"))
                comments = self._reviews_for_output(item.get("comments", []))
                item["comments"] = comments
                checkpoint = item.get("review_checkpoint")
                if isinstance(checkpoint, dict):
                    checkpoint["collected_reviews"] = len(comments)
                if comments:
                    item.pop("no_reviews_confirmed", None)
                    item.pop("no_reviews_evidence", None)
                    if item.get("review_collection_warning") == "This product has no written reviews":
                        item.pop("review_collection_warning", None)
                    total = int(item.get("total_rating", 0) or 0)
                    stop_reason = (item.get("review_checkpoint") or {}).get("stop_reason")
                    if total > len(comments) and stop_reason == "no_additional_reviews":
                        item["review_collection_warning"] = (
                            f"Temu displayed {total} ratings but exposed only "
                            f"{len(comments)} unique written review(s) in this browser session"
                        )
            self.output_data = {
                item["link"]: item
                for item in records
                if isinstance(item, dict) and item.get("link")
            }
        except (OSError, ValueError, TypeError) as exc:
            logging.warning("Could not load %s: %s", self.out_file, exc)

    def _periodic_save(self):
        output_path = Path(self.out_file)
        temporary_path = output_path.with_name(f"{output_path.name}.tmp")
        records = list(self.output_data.values())
        for item in records:
            if isinstance(item, dict):
                item["comments"] = self._reviews_for_output(
                    item.get("comments", [])
                )
                checkpoint = item.get("review_checkpoint")
                if isinstance(checkpoint, dict):
                    checkpoint["collected_reviews"] = len(item["comments"])
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, output_path)
        logging.info("Saved %s product(s) to %s", len(self.output_data), self.out_file)

    @staticmethod
    def _deduplicate_reviews(reviews):
        unique, seen = [], set()
        for review in reviews:
            identity = review.get("comment_id") or (
                review.get("author", ""),
                review.get("time", ""),
                review.get("content", ""),
                review.get("rating", 0),
            )
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(review)
        return unique

    def _reviews_for_output(self, reviews):
        """Keep only the non-identifying review fields required by DeFaketive."""
        minimized = []
        for review in self._deduplicate_reviews(reviews):
            if not isinstance(review, dict):
                continue
            comment_id = self._clean_text(review.get("comment_id"))
            rating = self._individual_review_rating(review.get("rating"))
            content = self._written_review_content(review.get("content"))
            if not comment_id or not rating or (not content and not self.include_empty):
                continue
            minimized.append(
                {
                    "comment_id": comment_id,
                    "rating": rating,
                    "content": content,
                }
            )
        return self._deduplicate_reviews(minimized)

    @classmethod
    def _reviews_probably_match(cls, left, right):
        """Match DOM and network forms of a review, including translated text."""
        left_rating = cls._individual_review_rating(left.get("rating"))
        right_rating = cls._individual_review_rating(right.get("rating"))
        if not left_rating or left_rating != right_rating:
            return False
        left_content = cls._clean_text(left.get("content")).casefold()
        right_content = cls._clean_text(right.get("content")).casefold()
        if left_content and left_content == right_content:
            return True
        left_time = cls._network_review_time(left.get("time")).casefold()
        right_time = cls._network_review_time(right.get("time")).casefold()
        if not left_time or left_time != right_time:
            return False
        left_author = cls._clean_text(left.get("author")).casefold()
        right_author = cls._clean_text(right.get("author")).casefold()
        return not left_author or not right_author or left_author == right_author

    @classmethod
    def _merge_review_sources(cls, dom_reviews, network_reviews):
        """Prefer structured network reviews and append only unmatched DOM cards."""
        network = cls._deduplicate_reviews(network_reviews)
        if not network:
            return cls._deduplicate_reviews(dom_reviews)
        merged = list(network)
        for review in cls._deduplicate_reviews(dom_reviews):
            if any(cls._reviews_probably_match(review, item) for item in network):
                continue
            merged.append(review)
        return cls._deduplicate_reviews(merged)

    @staticmethod
    def _review_matches_source_filter(review, source_filter):
        recorded_filter = review.get("source_rating_filter")
        if recorded_filter is not None:
            return int(recorded_filter) == int(source_filter)
        return int(float(review.get("rating", 0) or 0)) == int(source_filter)

    def _order_reviews_low_to_high(self, reviews):
        return sorted(
            self._deduplicate_reviews(reviews),
            key=lambda review: int(
                float(review.get("source_rating_filter") or review.get("rating", 0) or 6)
            ),
        )

    def _checkpoint_reviews(self, product, reviews, batch_number, stop_reason="", context=None):
        product["comments"] = self._reviews_for_output(reviews)
        product["review_extractor_version"] = self.REVIEW_EXTRACTOR_VERSION
        previous = product.get("review_checkpoint", {})
        product["review_checkpoint"] = {
            "context": context or ("all" if self.all_reviews else "limited"),
            "completed_batches": max(int(previous.get("completed_batches", 0)), batch_number),
            "collected_reviews": len(product["comments"]),
            "stop_reason": stop_reason,
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        if product.get("link"):
            self.output_data[product["link"]] = product
        self._periodic_save()

    @classmethod
    def _explicit_no_review_evidence(cls, body_text):
        """Return the rendered zero-review message, if Temu shows one explicitly."""
        text = cls._clean_text(body_text).casefold()
        patterns = (
            r"\b0\s*(?:customer\s+)?reviews?\b",
            r"\bno\s+(?:customer\s+|written\s+)?reviews?(?:\s+yet)?\b",
            r"\bno\s+ratings?(?:\s+yet)?\b",
            r"\bbe\s+the\s+first\s+to\s+(?:write\s+a\s+)?review\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(0)
        return ""

    def _record_no_reviews(self, product, evidence):
        if not str(evidence or "").strip():
            raise ValueError("Explicit Temu zero-review evidence is required")
        product["comments"] = []
        product["review_status"] = "no_reviews"
        product["review_collection_warning"] = "This product has no written reviews"
        product["no_reviews_confirmed"] = True
        product["no_reviews_evidence"] = self._clean_text(evidence)
        logging.info("No written Temu reviews found for %s", product.get("link"))
        self._checkpoint_reviews(
            product,
            [],
            batch_number=0,
            stop_reason="no_reviews",
            context="all",
        )

    def _record_review_widget_missing(self, product, warning):
        """Save an incomplete, retryable result without claiming zero reviews."""
        product.pop("no_reviews_confirmed", None)
        product.pop("no_reviews_evidence", None)
        product["comments"] = self._reviews_for_output(
            product.get("comments", [])
        )
        product["review_status"] = "review_widget_missing"
        product["review_collection_warning"] = warning
        logging.warning(
            "Temu review extraction is incomplete for %s: %s",
            product.get("link"),
            warning,
        )
        self._checkpoint_reviews(
            product,
            product["comments"],
            batch_number=0,
            stop_reason="review_widget_missing",
            context="all" if self.all_reviews else "limited",
        )

    def _existing_result_complete(self, product):
        """Return whether a checkpoint is both terminal and a usable product result."""
        if not self._clean_text(product.get("name")):
            return False
        try:
            extractor_version = int(product.get("review_extractor_version", 0) or 0)
        except (TypeError, ValueError):
            extractor_version = 0
        if extractor_version < self.REVIEW_EXTRACTOR_VERSION:
            return False
        expected = product.get("total_rating", 0) if self.all_reviews else self.review_limit
        comments = self._deduplicate_reviews(product.get("comments", []))
        checkpoint = product.get("review_checkpoint", {})
        context = checkpoint.get("context")
        stop_reason = checkpoint.get("stop_reason")
        terminal_stop = context in {"all", "limited"} and stop_reason in {
            "target_reached",
            "next_button_disabled",
            "no_additional_reviews",
            "all_rendered_reviews_scanned",
        }
        confirmed_empty = (
            context in {"all", "limited"}
            and stop_reason == "no_reviews"
            and product.get("review_status") == "no_reviews"
            and product.get("no_reviews_confirmed") is True
            and bool(self._clean_text(product.get("no_reviews_evidence")))
        )
        return terminal_stop or confirmed_empty or (
            expected > 0 and len(comments) >= expected
        )

    def _verification_detected(self):
        current_url = self.driver.current_url
        if isinstance(current_url, bytes):
            current_url = current_url.decode("utf-8", errors="replace")
        path = urlsplit(str(current_url or "")).path.casefold()
        title = (self.driver.title or "").casefold()
        body = ""
        try:
            body = self.driver.find_element(By.TAG_NAME, "body").text[:2000].casefold()
        except NoSuchElementException:
            pass
        return any(token in path for token in ("captcha", "verify", "login")) or any(
            token in f"{title} {body}"
            for token in (
                "verify you are human",
                "security verification",
                "unusual activity",
                "access denied",
            )
        )

    def _wait_for_verification_clear(self):
        if self.verification_timeout <= 0:
            return False
        logging.info("Waiting up to %s seconds for Temu verification", self.verification_timeout)
        deadline = time.time() + self.verification_timeout
        while time.time() < deadline:
            if not self._verification_detected():
                self._save_cookies()
                return True
            time.sleep(2)
        logging.warning("Temu verification was not completed within the timeout")
        return False

    def _check_verification(self):
        blocked = self._verification_detected()
        if blocked and self.pause_for_verification:
            logging.info("Temu login or verification detected. Complete it manually in Chrome.")
            input("Press Enter after the Temu page is ready...")
            time.sleep(2)
            blocked = self._verification_detected()
            if blocked:
                logging.warning("Temu is still showing login or verification")
        elif blocked and self._wait_for_verification_clear():
            blocked = False
        return blocked

    def _safe_get(self, url, check_verification=True):
        self.driver.get(url)
        try:
            WebDriverWait(self.driver, 30).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
        except TimeoutException:
            logging.warning("Timed out waiting for %s", url)
        time.sleep(1)
        if check_verification:
            return self._check_verification()
        return False

    def _prepare_session(self):
        """Restore saved authentication before waiting on Temu login gates."""
        self._safe_get(self.MARKET_HOME, check_verification=False)
        restored = self._load_cookies()
        if restored:
            self.driver.refresh()
            try:
                WebDriverWait(self.driver, 30).until(
                    lambda driver: driver.execute_script("return document.readyState") == "complete"
                )
            except TimeoutException:
                logging.warning("Timed out refreshing the restored Temu session")
            time.sleep(1)
        return not self._check_verification()

    def _dismiss_popups(self):
        xpath = (
            "//*[self::button or @role='button']["
            "translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
            "'abcdefghijklmnopqrstuvwxyz')='close' or "
            "translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
            "'abcdefghijklmnopqrstuvwxyz')='not now' or "
            "translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
            "'abcdefghijklmnopqrstuvwxyz')='continue shopping']"
        )
        for button in self.driver.find_elements(By.XPATH, xpath)[:3]:
            if not button.is_displayed():
                continue
            try:
                button.click()
            except Exception:
                continue

    def _card_for_anchor(self, anchor):
        try:
            return self.driver.execute_script(
                """
                return arguments[0].closest(
                  '[data-product-id], [data-goods-id], [data-uniqid*="goods"]'
                ) || arguments[0].parentElement?.parentElement?.parentElement || arguments[0];
                """,
                anchor,
            )
        except Exception:
            return anchor

    def _find_product_cards(self):
        cards, seen = [], set()
        for selector in self.PRODUCT_LINK_SELECTORS:
            for anchor in self.driver.find_elements(By.CSS_SELECTOR, selector):
                link = self._normalize_product_url(anchor.get_attribute("href"))
                if not link or link in seen:
                    continue
                seen.add(link)
                cards.append((self._card_for_anchor(anchor), anchor, link))
        return cards

    def _product_from_card(self, card, anchor, link):
        name = (
            anchor.get_attribute("aria-label")
            or anchor.get_attribute("title")
            or self._first_text(card, ('[data-uniqid*="name" i]', "h2", "h3"))
            or self._first_attribute(card, ("img",), "alt")
        )
        card_text = self._clean_text(getattr(card, "text", ""))
        price = self._first_text(
            card,
            (
                '[data-uniqid*="price" i]',
                '[itemprop="price"]',
                '[aria-label*="price" i]',
            ),
        )
        if not price:
            match = re.search(r"(?:₱|PHP\s*)\s*[\d,]+(?:\.\d{1,2})?", card_text, re.I)
            price = match.group(0) if match else ""
        rating_label = self._first_attribute(
            card,
            ('[aria-label*="out of 5" i]', '[aria-label*="star" i]'),
            "aria-label",
        )
        sold_match = re.search(r"([\d,.]+\s*[kmb]?\+?)\s*(?:sold|bought)", card_text, re.I)
        review_match = re.search(r"([\d,.]+\s*[kmb]?\+?)\s*reviews?", card_text, re.I)
        image = self._first_attribute(card, ("img",), "src")
        if not image or image.startswith("data:image"):
            image = self._first_attribute(card, ("img",), "data-src")
        return {
            "link": link,
            "goods_id": parse_qs(urlsplit(link).query).get("goods_id", [""])[0],
            "name": self._clean_text(name),
            "price": self._clean_text(price),
            "rating": self._parse_rating(rating_label),
            "total_rating": self._parse_compact_number(review_match.group(1)) if review_match else 0,
            "sold_count": self._parse_compact_number(sold_match.group(1)) if sold_match else 0,
            "img": image,
            "platform": "temu",
            "market": "PH",
            "currency": "PHP",
        }

    def _retrieve_search_products(self):
        if self._safe_get(self._build_search_url()):
            self._save_debug_artifacts("search")
            return []
        self._dismiss_popups()
        products: dict[str, dict[str, Any]] = {}
        unchanged = 0
        previous = -1
        for _ in range(self.pages * 8):
            for card, anchor, link in self._find_product_cards():
                if link not in products:
                    products[link] = self._product_from_card(card, anchor, link)
                if len(products) >= self.max_products:
                    return list(products.values())
            unchanged = unchanged + 1 if len(products) == previous else 0
            if unchanged >= 3:
                break
            previous = len(products)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1.2)
        if not products:
            self._save_debug_artifacts("search")
        return list(products.values())[: self.max_products]

    @staticmethod
    def _product_from_url(url):
        return {
            "link": url,
            "goods_id": parse_qs(urlsplit(url).query).get("goods_id", [""])[0],
            "name": "",
            "price": "",
            "rating": 0.0,
            "total_rating": 0,
            "sold_count": 0,
            "img": "",
            "platform": "temu",
            "market": "PH",
            "currency": "PHP",
        }

    def _json_ld_product(self):
        for element in self.driver.find_elements(By.CSS_SELECTOR, 'script[type="application/ld+json"]'):
            raw = element.get_attribute("textContent") or ""
            try:
                value = json.loads(raw)
            except (TypeError, ValueError):
                continue
            stack = value if isinstance(value, list) else [value]
            while stack:
                item = stack.pop()
                if isinstance(item, list):
                    stack.extend(item)
                    continue
                if not isinstance(item, dict):
                    continue
                item_type = item.get("@type", "")
                types = item_type if isinstance(item_type, list) else [item_type]
                if any(str(value).casefold() == "product" for value in types):
                    return item
                stack.extend(value for value in item.values() if isinstance(value, (dict, list)))
        return {}

    def _apply_json_ld(self, product, data):
        if not data:
            return
        product["name"] = self._clean_text(data.get("name")) or product.get("name", "")
        product["description"] = self._clean_text(data.get("description"))
        image = data.get("image", "")
        if isinstance(image, list):
            image = next(iter(image), "")
        if isinstance(image, dict):
            image = image.get("url", "")
        product["img"] = image or product.get("img", "")
        brand = data.get("brand", "")
        product["brand"] = self._clean_text(brand.get("name") if isinstance(brand, dict) else brand)

        offers = data.get("offers", {})
        if isinstance(offers, list):
            offers = next(iter(offers), {})
        if isinstance(offers, dict):
            raw_price = offers.get("price") or offers.get("lowPrice")
            currency = offers.get("priceCurrency")
            if raw_price not in (None, ""):
                prefix = "₱" if currency == "PHP" else f"{currency or ''} "
                product["price"] = f"{prefix}{raw_price}".strip()
            if currency:
                product["currency"] = currency

        aggregate = data.get("aggregateRating", {})
        if isinstance(aggregate, dict):
            product["rating"] = self._parse_rating(aggregate.get("ratingValue")) or product.get(
                "rating", 0.0
            )
            product["total_rating"] = self._parse_compact_number(
                aggregate.get("reviewCount") or aggregate.get("ratingCount")
            ) or product.get("total_rating", 0)

    def _review_root(self):
        candidates = []
        for selector in self.REVIEW_ROOT_SELECTORS:
            try:
                candidates.extend(self.driver.find_elements(By.CSS_SELECTOR, selector))
            except Exception:
                continue
        visible = [
            item
            for item in candidates
            if item.is_displayed() and len(self._clean_text(item.text)) >= 20
        ]
        return max(visible, key=lambda item: len(item.text), default=None)

    def _expand_review_widget(self, root):
        """Open Temu's full review list when the product page shows a preview."""
        try:
            old_count = self._review_card_count(root)
            candidates = root.find_elements(
                By.CSS_SELECTOR, 'button, [role="button"], a'
            )
        except Exception:
            return root
        pattern = re.compile(r"^(?:see|view)\s+all\s+reviews?$", re.I)
        for candidate in candidates:
            try:
                label = self._clean_text(
                    " ".join(
                        filter(
                            None,
                            (
                                candidate.text,
                                candidate.get_attribute("aria-label"),
                                candidate.get_attribute("title"),
                            ),
                        )
                    )
                )
                if not candidate.is_displayed() or not pattern.fullmatch(label):
                    continue
                self.driver.execute_script(
                    "arguments[0].scrollIntoView({block:'center'})", candidate
                )
                try:
                    candidate.click()
                except Exception:
                    self.driver.execute_script("arguments[0].click()", candidate)

                def expanded(_driver):
                    active = self._review_root()
                    if not active:
                        return False
                    if active != root or self._review_card_count(active) > old_count:
                        return active
                    return False

                try:
                    return WebDriverWait(
                        self.driver, min(self.pagination_timeout, 10)
                    ).until(expanded)
                except TimeoutException:
                    return self._review_root() or root
            except Exception:
                continue
        return root

    def _open_review_section(self):
        review_text_xpath = (
            "//*[self::button or self::a or @role='button']["
            "contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
            "'abcdefghijklmnopqrstuvwxyz'), 'see all reviews') or "
            "contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
            "'abcdefghijklmnopqrstuvwxyz'), 'customer reviews')]"
        )
        completed_passes = 0
        for _ in range(80):
            if self._verification_detected():
                logging.info(
                    "Temu verification appeared while opening the review section"
                )
                if self._check_verification():
                    return None
            root = self._review_root()
            if root:
                self.driver.execute_script("arguments[0].scrollIntoView({block:'center'})", root)
                time.sleep(1)
                return self._expand_review_widget(root)
            buttons = [
                item
                for item in self.driver.find_elements(By.XPATH, review_text_xpath)
                if item.is_displayed()
            ]
            if buttons:
                button = buttons[-1]
                self.driver.execute_script("arguments[0].scrollIntoView({block:'center'})", button)
                try:
                    button.click()
                except Exception:
                    self.driver.execute_script("arguments[0].click()", button)
                time.sleep(1.5)
                root = self._review_root()
                if root:
                    return root
            metrics = self.driver.execute_script(
                """
                return {
                  y: window.scrollY,
                  viewport: window.innerHeight,
                  height: document.documentElement.scrollHeight
                };
                """
            ) or {}
            at_bottom = (
                float(metrics.get("y", 0)) + float(metrics.get("viewport", 0))
                >= float(metrics.get("height", 0)) - 10
            )
            if at_bottom:
                if completed_passes >= 1:
                    break
                completed_passes += 1
                self.driver.execute_script("window.scrollTo(0, 0)")
                time.sleep(2)
                continue
            self.driver.execute_script(
                "window.scrollBy(0, Math.max(500, Math.floor(window.innerHeight * 0.7)))"
            )
            time.sleep(0.75)
        return None

    @staticmethod
    def _review_javascript():
        return r"""
        const root = arguments[0] || document;
        const selectors = [
          '[data-review-id]', '[data-comment-id]',
          '[data-testid*="review-item" i]',
          '[data-uniqid*="review-item" i]',
          '[data-uniqid*="review_item" i]'
        ];
        let items = [];
        for (const selector of selectors) {
          const found = Array.from(root.querySelectorAll(selector));
          if (found.length) { items = found; break; }
        }
        if (!items.length) {
          const stars = Array.from(root.querySelectorAll('[aria-label*="star" i]')).filter(star => {
            const label = (star.getAttribute('aria-label') || '').trim();
            return /^(?:rated\s+)?[1-5](?:\.0)?\s*(?:out of (?:five|5)\s*)?stars?$/i.test(label);
          });
          for (const star of stars) {
            let node = star;
            for (let depth = 0; node && depth < 12; depth++, node = node.parentElement) {
              const text = (node.innerText || '').trim();
              const avatar = node.querySelector(
                '[aria-label="avatar" i], img[alt="avatar" i]'
              );
              const datedText = node.querySelector('[role="text"][aria-label*=" on " i], time');
              if (avatar && datedText && text.length >= 10 && text.length <= 5000) {
                items.push(node);
                break;
              }
            }
          }
        }
        items = Array.from(new Set(items));
        const firstText = (item, selectors) => {
          for (const selector of selectors) {
            const node = item.querySelector(selector);
            const text = ((node && (node.innerText || node.textContent)) || '').trim();
            if (text) return text;
          }
          return '';
        };
        return items.map((item, index) => {
          const fullText = (item.innerText || '').trim();
          const lines = fullText.split(/\n+/).map(x => x.trim()).filter(Boolean);
          const ratingNode = item.querySelector('[aria-label*="star" i], [aria-label*="out of 5" i]');
          const ratingLabel = ratingNode ? (ratingNode.getAttribute('aria-label') || '') : '';
          let content = firstText(item, [
            '[data-testid*="review-content" i]', '[data-testid*="review-text" i]',
            '[data-uniqid*="review-content" i]', '[data-uniqid*="review_text" i]',
            '[itemprop="reviewBody"]'
          ]);
          if (!content) {
            const sections = Array.from(item.querySelectorAll('section'))
              .map(section => (section.innerText || section.textContent || '').trim())
              .filter(text =>
                text.length >= 2 &&
                !/^(?:all\s+)?reviews?\s+are\s+from\s+verified\s+purchases?$/i.test(text) &&
                !/^(?:on\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}$/i.test(text) &&
                !/^(?:purchased|size|color|style|model|type)\s*:/i.test(text)
              )
              .sort((a, b) => b.length - a.length);
            content = sections[0] || '';
          }
          if (!content) {
            const candidates = lines.filter(line =>
              line.length >= 5 &&
              !/^\d(?:\.\d)?\s*(?:out of 5|stars?)?$/i.test(line) &&
              !/^(helpful|report|verified purchase|purchased:|size:|color:|style:)/i.test(line) &&
              !/^\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4}$/.test(line)
            );
            content = candidates.sort((a, b) => b.length - a.length)[0] || '';
          }
          let author = firstText(item, [
            '[data-testid*="author" i]', '[data-testid*="user" i]',
            '[data-uniqid*="author" i]', '[data-uniqid*="user" i]',
            '[itemprop="author"]'
          ]);
          if (!author) {
            const authorLink = Array.from(item.querySelectorAll('[role="link"], a')).find(node => {
              const text = (node.innerText || node.textContent || '').trim();
              const label = (node.getAttribute('aria-label') || '').trim();
              return text.length >= 1 && text.length <= 80 &&
                !/^(?:helpful|report|see more|show more|view more)$/i.test(text) &&
                !/\bstars?\b/i.test(label) &&
                !/\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b/i.test(text);
            });
            author = (authorLink && (authorLink.innerText || authorLink.textContent) || '').trim();
          }
          const timeNode = item.querySelector('time');
          let reviewTime = (timeNode && (timeNode.getAttribute('datetime') || timeNode.innerText)) ||
            firstText(item, ['[data-testid*="date" i]', '[data-uniqid*="date" i]']);
          if (!reviewTime) {
            const datedNode = Array.from(item.querySelectorAll('[role="text"][aria-label]')).find(node =>
              /\bon\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b/i.test(node.getAttribute('aria-label') || '')
            );
            const datedLabel = (datedNode && datedNode.getAttribute('aria-label') || '').trim();
            const datedMatch = datedLabel.match(/\bon\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b/i);
            reviewTime = datedMatch ? datedMatch[0] : datedLabel;
          }
          if (!reviewTime) {
            reviewTime = lines.find(line =>
              /\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b/i.test(line) ||
              /\b\d{4}[\/-]\d{1,2}[\/-]\d{1,2}\b/.test(line)
            ) || '';
          }
          const variation = lines.find(line => /^(?:purchased|size|color|style|model|type)\s*:/i.test(line)) || '';
          const helpful = firstText(item, [
            '[data-testid*="helpful" i]', '[data-uniqid*="helpful" i]',
            'button[aria-label*="helpful" i]'
          ]);
          return {
            comment_id: item.dataset.reviewId || item.dataset.commentId || item.id || '',
            author,
            rating_label: ratingLabel,
            time: reviewTime,
            content,
            variation,
            helpful,
            snapshot_index: index
          };
        });
        """

    @staticmethod
    def _individual_review_rating(value):
        try:
            rating = float(value)
        except (TypeError, ValueError):
            return 0
        return int(rating) if rating.is_integer() and 1 <= rating <= 5 else 0

    @classmethod
    def _written_review_content(cls, content, review_time=""):
        content = cls._clean_text(content)
        review_time = cls._clean_text(review_time)
        if not content:
            return ""
        if re.fullmatch(
            r"(?:all\s+)?reviews?\s+are\s+from\s+verified\s+purchases?",
            content,
            re.I,
        ):
            return ""
        if re.fullmatch(r"purchased\s*:\s*.+", content, re.I):
            return ""
        if review_time and content.casefold() == review_time.casefold():
            return ""
        if re.fullmatch(
            r"(?:on\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
            r"\s+\d{1,2},?\s+\d{4}",
            content,
            re.I,
        ):
            return ""
        return content

    def _reviews_from_current_view(self, root):
        records = self.driver.execute_script(self._review_javascript(), root) or []
        reviews = []
        for item in records:
            author = self._clean_text(item.get("author"))
            review_time = self._clean_text(item.get("time"))
            content = self._written_review_content(item.get("content"), review_time)
            if author and content.casefold() == author.casefold():
                content = ""
            rating = self._individual_review_rating(
                self._parse_rating(item.get("rating_label"))
            )
            # Aggregate rating headers and date-only containers are not
            # individual written reviews.
            if not rating:
                continue
            if not content and not self.include_empty:
                continue
            reviews.append(
                {
                    "comment_id": self._clean_text(item.get("comment_id")),
                    "author": author,
                    "rating": rating,
                    "time": review_time,
                    "content": content,
                    "variation": self._clean_text(item.get("variation")),
                    "seller_respond": "",
                    "like_count": self._parse_compact_number(item.get("helpful")),
                }
            )
        return reviews

    def _review_signature(self, root):
        reviews = self._reviews_from_current_view(root)
        return "|".join(
            str(item.get("comment_id") or f"{item['author']}:{item['time']}:{item['content']}")
            for item in reviews[-3:]
        )

    def _review_card_count(self, root):
        """Count rendered review cards, including cards without written text."""
        records = self.driver.execute_script(self._review_javascript(), root) or []
        return len(records)

    @staticmethod
    def _review_scroll_javascript():
        """Scroll Temu's nearest review-modal scroller to its current end."""
        return """
            const root = arguments[0];
            if (!root) return false;
            const stars = Array.from(root.querySelectorAll('[aria-label*="star" i]')).filter(node =>
              /^(?:rated\s+)?[1-5](?:\.0)?\s*(?:out of (?:five|5)\s*)?stars?$/i.test((node.getAttribute('aria-label') || '').trim())
            );
            const last = stars[stars.length - 1] || root.lastElementChild || root;
            let node = last;
            let scroller = null;
            while (node && node !== document.body) {
              const style = window.getComputedStyle(node);
              const canScroll = node.scrollHeight > node.clientHeight + 2;
              const scrollsVertically = /^(?:auto|scroll|overlay)$/.test(style.overflowY);
              if (canScroll && scrollsVertically) {
                scroller = node;
                break;
              }
              node = node.parentElement;
            }
            if (last && last.scrollIntoView) last.scrollIntoView({block: 'end'});
            if (scroller) {
              scroller.scrollTop = scroller.scrollHeight;
              scroller.dispatchEvent(new Event('scroll', {bubbles: true}));
              return true;
            }
            if (root.scrollHeight > root.clientHeight + 2) {
              root.scrollTop = root.scrollHeight;
              root.dispatchEvent(new Event('scroll', {bubbles: true}));
              return true;
            }
            const bottom = root.getBoundingClientRect().bottom;
            window.scrollBy(0, Math.max(300, Math.min(window.innerHeight, bottom)));
            return false;
        """

    def _click_lowest_rating_sort(self, root, product=None):
        """Select Temu's exact Lowest rating review-sort option."""
        selector = 'button, [role="button"], [role="option"]'

        def label_for(element):
            return next(
                (
                    value
                    for value in (
                        self._clean_text(element.text),
                        self._clean_text(element.get_attribute("aria-label")),
                        self._clean_text(element.get_attribute("title")),
                    )
                    if value
                ),
                "",
            )

        def visible_exact(label):
            for scope in (root, self.driver):
                try:
                    candidates = scope.find_elements(By.CSS_SELECTOR, selector)
                except Exception:
                    continue
                for candidate in candidates:
                    try:
                        if candidate.is_displayed() and label_for(
                            candidate
                        ).casefold() == label.casefold():
                            return candidate
                    except StaleElementReferenceException:
                        continue
            return None

        option = visible_exact("Lowest rating")
        if option is None:
            trigger = None
            for scope in (root, self.driver):
                try:
                    candidates = scope.find_elements(By.CSS_SELECTOR, selector)
                except Exception:
                    continue
                for candidate in candidates:
                    try:
                        label = label_for(candidate).casefold()
                        if candidate.is_displayed() and (
                            label == "recommended"
                            or label == "recommended more selections"
                            or label == "sort by"
                        ):
                            trigger = candidate
                            break
                    except StaleElementReferenceException:
                        continue
                if trigger is not None:
                    break
            if trigger is None:
                return False
            try:
                self.driver.execute_script(
                    "arguments[0].scrollIntoView({block:'center'})", trigger
                )
                try:
                    trigger.click()
                except Exception:
                    self.driver.execute_script("arguments[0].click()", trigger)
                option = WebDriverWait(
                    self.driver, min(self.pagination_timeout, 5)
                ).until(lambda _driver: visible_exact("Lowest rating") or False)
            except (StaleElementReferenceException, TimeoutException):
                return False
        if product:
            # Discard the initial Recommended response. The next captured review
            # response must be the one triggered by Lowest rating.
            self._reset_product_network_capture(product)
        try:
            self.driver.execute_script(
                "arguments[0].scrollIntoView({block:'center'})", option
            )
            try:
                option.click()
            except Exception:
                self.driver.execute_script("arguments[0].click()", option)
            time.sleep(1.5)
            logging.info("Selected Temu review sort: Lowest rating")
            return True
        except Exception as exc:
            logging.warning("Could not select Temu Lowest rating sort: %s", exc)
            return False

    def _click_rating_filter(self, root, rating):
        previous_signature = self._review_signature(root)
        candidates = []
        for scope in (root, self.driver):
            try:
                candidates.extend(
                    scope.find_elements(
                        By.CSS_SELECTOR,
                        'button, [role="button"], [role="option"], label',
                    )
                )
            except Exception:
                continue
        pattern = re.compile(rf"(^|\D){rating}\s*(?:star|stars|â˜…)(?:\D|$)", re.I)
        for candidate in candidates:
            try:
                label = self._clean_text(
                    " ".join(
                        filter(
                            None,
                            (
                                candidate.text,
                                candidate.get_attribute("aria-label"),
                                candidate.get_attribute("title"),
                            ),
                        )
                    )
                )
                if len(label) > 100 or not candidate.is_displayed() or not pattern.search(label):
                    continue
                self.driver.execute_script(
                    "arguments[0].scrollIntoView({block:'center'})", candidate
                )
                try:
                    candidate.click()
                except Exception:
                    self.driver.execute_script("arguments[0].click()", candidate)
                WebDriverWait(self.driver, min(self.pagination_timeout, 10)).until(
                    lambda _: self._review_signature(self._review_root() or root)
                    != previous_signature
                )
                return True
            except TimeoutException:
                continue
            except Exception:
                continue
        return False

    def _advance_reviews(self, root, previous_signature):
        pattern = re.compile(
            r"^(?:next(?:\s+(?:page|reviews?))?|(?:load|show|view)\s+more\s+reviews?|more\s+reviews?)$",
            re.I,
        )
        self._last_pagination_stop_reason = ""
        for attempt in range(1, self.pagination_retries + 1):
            if self._review_signature(root) != previous_signature:
                return True
            # Keep pagination inside the review widget. Temu also has
            # unrelated product-description "See more" controls.
            try:
                candidates = root.find_elements(
                    By.CSS_SELECTOR, 'button, [role="button"], a'
                )
            except Exception:
                candidates = []
            button_to_click = None
            disabled_button_seen = False
            for button in candidates:
                label = self._clean_text(
                    " ".join(
                        filter(
                            None,
                            (
                                button.text,
                                button.get_attribute("aria-label"),
                                button.get_attribute("title"),
                            ),
                        )
                    )
                )
                if not button.is_displayed() or not pattern.search(label):
                    continue
                disabled = (button.get_attribute("aria-disabled") or "").casefold() == "true"
                if disabled or button.get_attribute("disabled") is not None:
                    disabled_button_seen = True
                    continue
                button_to_click = button
                break
            try:
                if button_to_click is not None:
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block:'center'})", button_to_click
                    )
                    try:
                        button_to_click.click()
                    except Exception:
                        self.driver.execute_script("arguments[0].click()", button_to_click)
                    self._last_pagination_stop_reason = (
                        f"page_signature_timeout_attempt_{attempt}"
                    )
                elif disabled_button_seen:
                    self._last_pagination_stop_reason = "next_button_disabled"
                else:
                    self.driver.execute_script(
                        self._review_scroll_javascript(),
                        root,
                    )
                    self._last_pagination_stop_reason = "no_additional_reviews"
                WebDriverWait(self.driver, self.pagination_timeout).until(
                    lambda _: self._review_signature(self._review_root() or root)
                    != previous_signature
                )
                return True
            except TimeoutException:
                pass
            except Exception as exc:
                self._last_pagination_stop_reason = f"page_transition_error_{type(exc).__name__}"
            if attempt < self.pagination_retries:
                logging.warning(
                    "Temu review pagination attempt %s/%s failed: %s",
                    attempt,
                    self.pagination_retries,
                    self._last_pagination_stop_reason,
                )
                time.sleep(min(1.5 * attempt, 5))
        if self._last_pagination_stop_reason != "next_button_disabled":
            self._save_debug_artifacts("review_pagination_failure")
        return False

    def _on_requested_product_page(self, product):
        if not product or not self.driver or not hasattr(self.driver, "current_url"):
            return True
        current_url = self.driver.current_url
        if isinstance(current_url, bytes):
            current_url = current_url.decode("utf-8", errors="replace")
        return self._normalize_product_url(current_url) == product.get("link")

    @staticmethod
    def _review_collection_was_redirected(product):
        checkpoint = product.get("review_checkpoint") or {}
        return checkpoint.get("stop_reason") == "product_redirected_during_reviews"

    def _collect_reviews(self, root, product=None, max_reviews=None, source_filter=None):
        target = (
            None if self.all_reviews else self.review_limit
        ) if max_reviews is None else max(0, max_reviews)
        if target == 0:
            return []
        existing_reviews = list(product.get("comments", [])) if product else []
        if source_filter is None:
            collected = self._deduplicate_reviews(existing_reviews)
        else:
            collected = self._deduplicate_reviews(
                review
                for review in existing_reviews
                if self._review_matches_source_filter(review, source_filter)
            )
            for review in collected:
                review.setdefault("source_rating_filter", source_filter)
        seen = {
            review.get("comment_id")
            or (
                review.get("author", ""),
                review.get("time", ""),
                review.get("content", ""),
                review.get("rating", 0),
            )
            for review in collected
        }
        stop_reason = "target_reached" if target is not None and len(collected) >= target else ""
        batch_number = 0
        if stop_reason:
            if product:
                checkpoint_reviews = (
                    collected
                    if source_filter is None
                    else [
                        review
                        for review in existing_reviews
                        if not self._review_matches_source_filter(review, source_filter)
                    ] + collected
                )
                self._checkpoint_reviews(
                    product,
                    checkpoint_reviews,
                    batch_number,
                    stop_reason=stop_reason,
                    context=str(source_filter) if source_filter is not None else None,
                )
            return collected
        description = "Collecting all exposed Temu reviews" if target is None else "Collecting Temu reviews"
        with tqdm(total=target, initial=len(collected), desc=description) as progress:
            for batch_number in range(1, self.max_review_batches + 1):
                root = self._review_root() or root
                if not self._on_requested_product_page(product):
                    stop_reason = "product_redirected_during_reviews"
                    self._save_debug_artifacts("review_product_redirect")
                    break
                dom_reviews = self._reviews_from_current_view(root)
                network_reviews = []
                if product:
                    network_reviews = self._capture_review_network(product)
                    if source_filter is not None:
                        network_reviews = [
                            review
                            for review in network_reviews
                            if self._review_matches_source_filter(
                                review, source_filter
                            )
                        ]
                    view_reviews = self._merge_review_sources(
                        dom_reviews, network_reviews
                    )
                else:
                    view_reviews = dom_reviews
                if source_filter is not None:
                    view_reviews = [
                        review
                        for review in view_reviews
                        if self._review_matches_source_filter(review, source_filter)
                    ]
                if network_reviews:
                    previous_count = len(collected)
                    collected = self._merge_review_sources(
                        collected, network_reviews
                    )
                    progress.update(len(collected) - previous_count)
                    seen = {
                        review.get("comment_id")
                        or (
                            review.get("author", ""),
                            review.get("time", ""),
                            review.get("content", ""),
                            review.get("rating", 0),
                        )
                        for review in collected
                    }
                    if target is not None and len(collected) >= target:
                        stop_reason = "target_reached"
                if not view_reviews:
                    stop_reason = "empty_review_batch"
                    self._save_debug_artifacts("reviews")
                    break
                for review in view_reviews:
                    if source_filter is not None:
                        review["source_rating_filter"] = source_filter
                    identity = review.get("comment_id") or (
                        review["author"],
                        review["time"],
                        review["content"],
                        review["rating"],
                    )
                    if identity in seen:
                        continue
                    seen.add(identity)
                    collected.append(review)
                    progress.update(1)
                    if target is not None and len(collected) >= target:
                        stop_reason = "target_reached"
                        break
                displayed_total = int(product.get("total_rating", 0) or 0) if product else 0
                rendered_card_count = (
                    self._review_card_count(root) if displayed_total > 0 else 0
                )
                if (
                    not stop_reason
                    and displayed_total > 0
                    and rendered_card_count >= displayed_total
                ):
                    stop_reason = "all_rendered_reviews_scanned"
                if product and batch_number % self.checkpoint_every == 0:
                    checkpoint_reviews = (
                        collected
                        if source_filter is None
                        else [
                            review
                            for review in existing_reviews
                            if not self._review_matches_source_filter(review, source_filter)
                        ] + collected
                    )
                    self._checkpoint_reviews(
                        product,
                        checkpoint_reviews,
                        batch_number,
                        context=str(source_filter) if source_filter is not None else None,
                    )
                if stop_reason:
                    break
                signature = self._review_signature(root)
                if not signature:
                    stop_reason = "missing_review_signature"
                    self._save_debug_artifacts("review_pagination_failure")
                    break
                if not self._advance_reviews(root, signature):
                    stop_reason = (
                        "product_redirected_during_reviews"
                        if not self._on_requested_product_page(product)
                        else self._last_pagination_stop_reason
                        or "page_transition_failed"
                    )
                    break
                root = self._review_root() or root
            else:
                stop_reason = "max_review_batches_reached"
        if stop_reason and stop_reason != "target_reached":
            logging.info(
                "Temu review pagination stopped after batch %s with %s review(s): %s",
                batch_number,
                len(collected),
                stop_reason,
            )
        if product:
            checkpoint_reviews = (
                collected
                if source_filter is None
                else [
                    review
                    for review in existing_reviews
                    if not self._review_matches_source_filter(review, source_filter)
                ] + collected
            )
            self._checkpoint_reviews(
                product,
                checkpoint_reviews,
                batch_number,
                stop_reason=stop_reason,
                context=str(source_filter) if source_filter is not None else None,
            )
        return collected

    def _collect_rating_filtered_reviews(self, root, product):
        """Collect visible rating groups from the lowest star value upward."""
        reviews = []
        applied_filters = []
        for star in range(1, 6):
            remaining = None if self.all_reviews else self.review_limit - len(
                self._deduplicate_reviews(reviews)
            )
            if remaining is not None and remaining <= 0:
                break
            active_root = self._review_root() or root
            if not self._click_rating_filter(active_root, star):
                continue
            active_root = self._review_root() or active_root
            star_reviews = self._collect_reviews(
                active_root,
                product=product,
                max_reviews=remaining,
                source_filter=star,
            )
            if self._review_collection_was_redirected(product):
                return reviews, applied_filters, True
            applied_filters.append(star)
            reviews.extend(star_reviews)
        return reviews, applied_filters, False

    def _scrape_details(self, product):
        # A layout/login failure saved by an older version may carry a false
        # terminal no_reviews claim. This attempt must establish it again.
        try:
            stored_extractor_version = int(
                product.get("review_extractor_version", 0) or 0
            )
        except (TypeError, ValueError):
            stored_extractor_version = 0
        revalidating_legacy_result = (
            stored_extractor_version < self.REVIEW_EXTRACTOR_VERSION
        )
        if revalidating_legacy_result:
            # Revalidate old snapshots, but retain already verified comments
            # if Temu temporarily omits the widget during this retry.
            product.pop("review_checkpoint", None)
        product["review_extractor_version"] = self.REVIEW_EXTRACTOR_VERSION
        product.pop("no_reviews_confirmed", None)
        product.pop("no_reviews_evidence", None)
        product.pop("review_collection_warning", None)
        product.pop("review_filter_warning", None)
        product.pop("applied_rating_filters", None)
        product.pop("review_sort", None)
        product.pop("sentiment_summary", None)
        cleaned_existing_comments = [
            review
            for review in product.get("comments", [])
            if isinstance(review, dict)
            and self._individual_review_rating(review.get("rating"))
            and (
                self._written_review_content(
                    review.get("content"), review.get("time")
                )
                or self.include_empty
            )
        ]
        legacy_comments = (
            cleaned_existing_comments if revalidating_legacy_result else []
        )
        product["comments"] = (
            [] if revalidating_legacy_result else cleaned_existing_comments
        )
        navigation_url = self.product_navigation_urls.get(
            product["link"], product["link"]
        )
        self._reset_product_network_capture(product)
        if self._safe_get(navigation_url):
            self._save_debug_artifacts("blocked_product")
            raise RuntimeError("Temu required login or verification before showing the product")
        if self._normalize_product_url(self.driver.current_url) != product["link"]:
            self._save_debug_artifacts("unexpected_product_redirect")
            raise RuntimeError("Temu redirected away from the requested product page")
        self._dismiss_popups()
        self._apply_json_ld(product, self._json_ld_product())

        page_title = self._normalize_product_name(getattr(self.driver, "title", ""))
        product["name"] = self._normalize_product_name(
            self._first_text(
                self.driver,
                ('h1', '[itemprop="name"]', '[data-uniqid*="goods_name" i]'),
            )
            or self._first_attribute(
                self.driver,
                ('meta[property="og:title"]', 'meta[name="twitter:title"]'),
                "content",
            )
            or product.get("name", "")
            or page_title
        )
        price = self._first_text(
            self.driver,
            (
                '[itemprop="price"]',
                '[data-uniqid*="price" i]',
                '[aria-label*="price" i]',
            ),
        ) or self._first_attribute(
            self.driver,
            (
                'meta[property="product:price:amount"]',
                'meta[property="og:price:amount"]',
                'meta[itemprop="price"]',
            ),
            "content",
        ) or self._price_from_page_source(getattr(self.driver, "page_source", ""))
        if price and len(price) < 80:
            product["price"] = price
        rating_label = self._first_attribute(
            self.driver,
            ('[aria-label*="out of 5" i]', '[itemprop="ratingValue"]'),
            "aria-label",
        ) or self._first_attribute(self.driver, ('[itemprop="ratingValue"]',), "content")
        product["rating"] = self._parse_rating(rating_label) or product.get("rating", 0.0)
        product["description"] = self._clean_text(
            product.get("description")
            or self._first_attribute(
                self.driver,
                ('meta[name="description"]', 'meta[property="og:description"]'),
                "content",
            )
        )
        product["img"] = product.get("img") or self._first_attribute(
            self.driver,
            ('meta[property="og:image"]', '[itemprop="image"]', 'img[src*="kwcdn.com"]'),
            "content",
        ) or self._first_attribute(self.driver, ('img[src*="kwcdn.com"]',), "src")

        body_text = self._clean_text(self.driver.find_element(By.TAG_NAME, "body").text)
        if not product.get("total_rating"):
            count_match = re.search(r"([\d,.]+\s*[kmb]?\+?)\s*(?:customer\s+)?reviews?", body_text, re.I)
            if count_match:
                product["total_rating"] = self._parse_compact_number(count_match.group(1))
        if not product.get("sold_count"):
            sold_match = re.search(r"([\d,.]+\s*[kmb]?\+?)\s*(?:sold|bought)", body_text, re.I)
            if sold_match:
                product["sold_count"] = self._parse_compact_number(sold_match.group(1))

        zero_review_evidence = self._explicit_no_review_evidence(body_text)
        root = self._open_review_section()
        if root is None:
            if zero_review_evidence:
                self._record_no_reviews(product, zero_review_evidence)
            else:
                self._save_debug_artifacts("reviews")
                if legacy_comments:
                    product["comments"] = legacy_comments
                self._record_review_widget_missing(
                    product,
                    "No review section was identified and Temu did not explicitly display zero reviews",
                )
            return
        if not product.get("rating"):
            rating_labels = []
            try:
                rating_labels = [
                    element.get_attribute("aria-label")
                    for element in root.find_elements(
                        By.CSS_SELECTOR,
                        '[aria-label*="out of 5" i], [aria-label*="star" i]',
                    )
                ]
            except Exception:
                pass
            rating_values = [
                self._parse_rating(label) for label in rating_labels if label
            ]
            aggregate = next(
                (value for value in rating_values if 0 < value <= 5 and not value.is_integer()),
                0.0,
            )
            if aggregate:
                product["rating"] = aggregate
        if not product.get("total_rating"):
            root_text = self._clean_text(getattr(root, "text", ""))
            count_match = re.search(
                r"([\d,.]+\s*[kmb]?\+?)\s*(?:customer\s+)?reviews?",
                root_text,
                re.I,
            )
            if count_match:
                product["total_rating"] = self._parse_compact_number(count_match.group(1))
        product["review_status"] = "collecting"
        lowest_sort_applied = self._click_lowest_rating_sort(root, product)
        if lowest_sort_applied:
            product["review_sort"] = "lowest_rating"
            active_root = self._review_root() or root
            reviews = self._collect_reviews(active_root, product=product)
            applied_filters = []
            review_collection_redirected = self._review_collection_was_redirected(
                product
            )
        else:
            reviews, applied_filters, review_collection_redirected = (
                self._collect_rating_filtered_reviews(root, product)
            )
        if review_collection_redirected:
            if len(legacy_comments) > len(product.get("comments", [])):
                product["comments"] = legacy_comments
            self._record_review_widget_missing(
                product,
                "Temu redirected away from the product while reviews were loading",
            )
            return
        if lowest_sort_applied:
            product["comments"] = self._reviews_for_output(
                self._order_reviews_low_to_high(reviews)
            )
            if not self.all_reviews:
                product["comments"] = product["comments"][: self.review_limit]
        elif applied_filters:
            product["applied_rating_filters"] = applied_filters
            product["comments"] = self._reviews_for_output(
                self._order_reviews_low_to_high(reviews)
            )
            if not self.all_reviews:
                product["comments"] = product["comments"][: self.review_limit]
        else:
            product["review_filter_warning"] = (
                "Temu did not expose the Lowest rating sort or identifiable "
                "star filters; reviews were collected in rendered order and "
                "sorted by extracted rating"
            )
            product["comments"] = self._reviews_for_output(
                self._order_reviews_low_to_high(
                    self._collect_reviews(root, product=product)
                )
            )
            review_collection_redirected = self._review_collection_was_redirected(
                product
            )
        if review_collection_redirected:
            if len(legacy_comments) > len(product.get("comments", [])):
                product["comments"] = legacy_comments
            self._record_review_widget_missing(
                product,
                "Temu redirected away from the product while reviews were loading",
            )
            return
        if product["comments"]:
            product["review_status"] = "complete"
            product["total_rating"] = max(
                int(product.get("total_rating", 0)), len(product["comments"])
            )
        elif zero_review_evidence:
            self._record_no_reviews(product, zero_review_evidence)
            return
        else:
            if legacy_comments:
                product["comments"] = legacy_comments
            self._record_review_widget_missing(
                product,
                "Temu exposed no written review cards and did not explicitly display zero reviews",
            )
            return
        total = product.get("total_rating", 0)
        requested_reviews = None if self.all_reviews else self.review_limit
        collection_shortfall = (
            total
            and len(product["comments"]) < total
            and (
                requested_reviews is None
                or len(product["comments"]) < requested_reviews
            )
        )
        if collection_shortfall:
            product["review_collection_warning"] = (
                f"Temu displayed {total} ratings but exposed only "
                f"{len(product['comments'])} unique written review(s) in this browser session"
            )

    def _qualifies(self, product):
        return product.get("rating", 0.0) >= self.min_rating and product.get(
            "total_rating", 0
        ) >= self.min_reviews

    def execute(self):
        major = self._detect_chrome_major()
        kwargs = {"options": self.options, "enable_cdp_events": False, "headless": False}
        if major:
            kwargs["version_main"] = major
            cached = self._cached_driver_for_major(major)
            if cached:
                kwargs["driver_executable_path"] = cached
        logging.info("Starting Chrome for Temu Philippines")
        self.driver = SafeChrome(**kwargs)
        try:
            self.driver.execute_cdp_cmd("Network.enable", {})
        except Exception as exc:
            self._performance_logging_available = False
            logging.warning(
                "Chrome network capture is unavailable; using rendered reviews only: %s",
                exc,
            )
        self.driver.maximize_window()
        successful_details = 0
        try:
            self._prepare_session()
            if self.product_urls:
                products = [self._product_from_url(url) for url in self.product_urls]
            else:
                products = self._retrieve_search_products()
            if not products:
                raise RuntimeError("No real Temu products were extracted; inspect debug artifacts")

            for product in tqdm(products, desc="Processing Temu products"):
                link = product["link"]
                existing = self.output_data.get(link)
                if existing:
                    expected = existing.get("total_rating", 0) if self.all_reviews else self.review_limit
                    if self._existing_result_complete(existing):
                        successful_details += 1
                        continue
                    logging.info(
                        "Retrying incomplete Temu result for %s (%s/%s reviews)",
                        link,
                        len(existing.get("comments", [])),
                        expected or "unknown",
                    )
                    product = existing
                try:
                    self._scrape_details(product)
                    successful_details += 1
                    if self._qualifies(product):
                        self.output_data[link] = product
                    else:
                        logging.info(
                            "Skipped %s: rating %.1f, %s reviews",
                            link,
                            product.get("rating", 0.0),
                            product.get("total_rating", 0),
                        )
                        self.output_data.pop(link, None)
                except Exception as exc:
                    logging.warning("Detail scrape failed for %s: %s", link, exc, exc_info=True)
                    if product.get("comments"):
                        self.output_data[link] = product
                    else:
                        self.output_data.pop(link, None)
                self._periodic_save()
        finally:
            self._save_cookies()
            if self.driver:
                self.driver.quit()
        self._periodic_save()
        if successful_details == 0:
            raise RuntimeError(
                "No Temu product pages were extracted; complete Temu login/verification "
                "manually and inspect the saved diagnostics"
            )
        return list(self.output_data.values())


def build_parser():
    parser = argparse.ArgumentParser(description="Scrape Temu Philippines products and reviews")
    parser.add_argument("keyword", nargs="?", default="", help="Temu search keyword")
    parser.add_argument("-n", "--num", type=int, default=10, help="Maximum products")
    parser.add_argument(
        "-r",
        "--review-limit",
        type=int,
        default=DEFAULT_REVIEW_LIMIT,
        help=f"Written reviews per product (default: {DEFAULT_REVIEW_LIMIT})",
    )
    parser.add_argument("--all-reviews", action="store_true", help="Collect until Temu exposes no more reviews")
    parser.add_argument("--max-review-batches", type=int, default=100, help="Safety cap for all-review loading")
    parser.add_argument("--pages", type=int, default=1, help="Search-result scroll batches")
    parser.add_argument("--product-url", action="append", default=[], help="Direct Temu product URL; repeatable")
    parser.add_argument("--min-rating", type=float, default=0.0, help="Minimum product rating")
    parser.add_argument("--min-reviews", type=int, default=0, help="Minimum displayed review count")
    parser.add_argument("--include-empty", action="store_true", help="Include ratings with no written comment")
    parser.add_argument("--chrome-user-data-dir", help="Optional Chrome user-data directory")
    parser.add_argument("--cookies-file", help="Optional Temu session-cookie file")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("--no-verification-pause", action="store_true", help="Do not prompt on verification pages")
    parser.add_argument("--verification-timeout", type=int, default=0, help="Seconds to wait for unattended verification")
    parser.add_argument("--pagination-retries", type=int, default=3, help="Retries for a failed review transition")
    parser.add_argument("--pagination-timeout", type=float, default=15, help="Seconds to wait for each review transition")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save review progress every N batches")
    return parser


def main():
    args = build_parser().parse_args()
    scraper = TemuScraper(
        keyword=args.keyword,
        max_products=args.num,
        review_limit=args.review_limit,
        pages=args.pages,
        product_urls=args.product_url,
        all_reviews=args.all_reviews,
        max_review_batches=args.max_review_batches,
        min_rating=args.min_rating,
        min_reviews=args.min_reviews,
        include_empty=args.include_empty,
        chrome_user_data_dir=args.chrome_user_data_dir,
        cookies_file=args.cookies_file,
        output=args.output,
        pause_for_verification=not args.no_verification_pause,
        verification_timeout=args.verification_timeout,
        pagination_retries=args.pagination_retries,
        pagination_timeout=args.pagination_timeout,
        checkpoint_every=args.checkpoint_every,
    )
    scraper.execute()


if __name__ == "__main__":
    main()
