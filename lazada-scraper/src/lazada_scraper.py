"""Visible-browser scraper for public Lazada Philippines products and reviews.

The control flow is adapted from the MIT-licensed Shopee scraper used by this
project. Lazada-specific URLs, selectors, parsers, and output fields are
implemented here. The scraper never substitutes generated/sample data when a
page cannot be parsed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import pickle
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

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
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm


def _position_scraper_window(driver):
    """Leave the left side of the screen visible for the app's progress panel."""
    screen_width, screen_height = 1366, 768
    try:
        screen = driver.execute_script(
            "return {width: window.screen.availWidth, height: window.screen.availHeight};"
        )
        screen_width = int(screen.get("width") or screen_width)
        screen_height = int(screen.get("height") or screen_height)
    except Exception:
        logging.debug("Could not read the available screen size", exc_info=True)
    width = min(1050, max(720, int(screen_width * 0.60)), max(640, screen_width - 32))
    height = min(820, max(560, int(screen_height * 0.84)), max(520, screen_height - 64))
    left = max(0, screen_width - width - 16)
    top = max(0, min(40, screen_height - height))
    try:
        driver.set_window_size(width, height)
        driver.set_window_position(left, top)
        logging.info("Chrome window positioned at %sx%s so app progress remains visible.", width, height)
    except Exception:
        logging.warning("Chrome window could not be resized; continuing normally.", exc_info=True)


class SafeChrome(uc.Chrome):
    """Suppress undetected-chromedriver's harmless Windows double-quit error."""

    def __del__(self):
        try:
            super().__del__()
        except OSError:
            pass


class LazadaScraper:
    """Scrape public Lazada Philippines product metadata and written reviews."""

    BASE_URL = "https://www.lazada.com.ph"
    # Three-star reviews are the highest-priority annotation stratum. Continue
    # with the other underrepresented middle ratings before the abundant
    # one- and five-star extremes.
    RATING_COLLECTION_ORDER = (3, 2, 4, 1, 5)
    PRODUCT_CARD_SELECTORS = (
        '[data-qa-locator="product-item"]',
        '[data-item-id]',
        'div.Bm3ON',
        'div[data-spm="list"] > div',
    )
    PRODUCT_LINK_SELECTORS = (
        'a[href*="/products/"]',
        'a[href*="-i"][href*=".html"]',
        '.RfADt a',
    )
    REVIEW_CONTAINER_SELECTORS = (
        '#module_product_review',
        '.mod-reviews',
        '[data-qa-locator="product-review"]',
        '[data-spm="product_review"]',
    )
    REVIEW_NEXT_SELECTORS = (
        '.iweb-pagination-next button',
        '.iweb-pagination-next',
        '.next-pagination-item.next button',
        '.next-pagination-item.next',
        'li.ant-pagination-next button',
        'button[aria-label="Next"]',
        'button[title="Next Page"]',
    )

    def __init__(
        self,
        keyword: str = "",
        max_products: int = 10,
        review_limit: int = 100,
        pages: int = 1,
        product_urls: list[str] | None = None,
        category_urls: list[str] | None = None,
        rating: int | None = None,
        all_star_types: bool = False,
        star_limit_per_type: int = 10,
        include_empty: bool = False,
        chrome_user_data_dir: str | None = None,
        output: str | None = None,
        pause_for_verification: bool = True,
        verification_timeout: int = 0,
        pagination_retries: int = 3,
        pagination_timeout: float = 15.0,
        checkpoint_every: int = 5,
        max_review_pages: int = 5000,
        min_rating: float = 0.0,
        min_reviews: int = 0,
        discovery_checkpoint: str | None = None,
        research_stratum: str = "",
        min_collected_reviews: int = 0,
        collection_report: str | None = None,
    ):
        if not keyword.strip() and not product_urls and not category_urls:
            raise ValueError(
                "Provide a keyword, Lazada category URL, or Lazada product URL"
            )
        if rating is not None and rating not in range(1, 6):
            raise ValueError("rating must be between 1 and 5")
        if not 0 <= min_rating <= 5:
            raise ValueError("min_rating must be between 0 and 5")
        if min_reviews < 0:
            raise ValueError("min_reviews cannot be negative")
        if min_collected_reviews < 0:
            raise ValueError("min_collected_reviews cannot be negative")

        self.driver = None
        self.keyword = keyword.strip()
        self.max_products = max(0, max_products)
        self.review_limit = min(100, max(0, review_limit))
        self.pages = max(1, pages)
        self.product_urls = [self._normalize_product_url(url) for url in (product_urls or [])]
        self.product_urls = [url for url in self.product_urls if url]
        self.category_urls = [
            self._normalize_category_url(url) for url in (category_urls or [])
        ]
        self.category_urls = [url for url in self.category_urls if url]
        if category_urls and not self.category_urls:
            raise ValueError("No valid Lazada Philippines category URLs were provided")
        self.rating = rating
        self.all_star_types = all_star_types
        self.star_limit_per_type = max(0, star_limit_per_type)
        self.include_empty = include_empty
        self.chrome_user_data_dir = chrome_user_data_dir
        self.pause_for_verification = pause_for_verification
        self.verification_timeout = max(0, int(verification_timeout))
        self.pagination_retries = max(1, pagination_retries)
        self.pagination_timeout = max(1.0, pagination_timeout)
        self.checkpoint_every = max(1, checkpoint_every)
        self.max_review_pages = max(1, max_review_pages)
        self.min_rating = float(min_rating)
        self.min_reviews = int(min_reviews)
        self.research_stratum = research_stratum.strip()
        self.min_collected_reviews = int(min_collected_reviews)
        self._last_pagination_stop_reason = ""
        self.cookies_file = "cookies_lazada_ph.dat"
        self._cookies_loaded = False

        category_slug = ""
        if self.category_urls:
            category_slug = urlsplit(self.category_urls[0]).path.strip("/").split("/")[-1]
        slug_source = self.keyword or category_slug or "products"
        slug = re.sub(r"[^a-z0-9_]+", "_", slug_source.lower()).strip("_")
        self.out_file = output or f"lazada_ph_{slug or 'products'}.json"
        output_path = Path(self.out_file)
        self.discovery_checkpoint_file = discovery_checkpoint or str(
            output_path.with_name(f"{output_path.stem}_discovery.json")
        )
        self.collection_report_file = collection_report or (
            str(output_path.with_name(f"{output_path.stem}_quality.json"))
            if self.research_stratum or self.min_collected_reviews
            else ""
        )
        self.output_data: dict[str, dict] = {}
        self.discovery_products: dict[str, dict] = {}
        self.discovery_completed_pages: dict[str, int] = {}
        self._setup_logging()
        self.options = uc.ChromeOptions()
        self._configure_options()
        self._load_existing_data()
        self._load_discovery_checkpoint()

    def _setup_logging(self):
        os.makedirs("logs", exist_ok=True)
        if logging.getLogger().handlers:
            return
        stamp = datetime.datetime.now().strftime("lazada_ph_%d_%m_%H_%M_%S.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(Path("logs") / stamp), logging.StreamHandler(sys.stdout)],
        )

    def _configure_options(self):
        if self.chrome_user_data_dir:
            profile = os.path.abspath(os.path.expanduser(self.chrome_user_data_dir))
            self.options.add_argument(f"--user-data-dir={profile}")
        if sys.platform.startswith("linux"):
            self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        # Lazada occasionally fails in Chrome with ERR_QUIC_PROTOCOL_ERROR.
        # Falling back to TCP/TLS avoids that browser-level failure.
        self.options.add_argument("--disable-quic")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_argument("--window-size=900,700")
        self.options.add_argument("--lang=en-PH")
        self.options.add_argument("--no-first-run")
        self.options.add_argument("--no-default-browser-check")
        self.options.add_argument("--disable-default-apps")
        self.options.add_argument("--disable-search-engine-choice-screen")

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
        host = parts.netloc.lower().split(":")[0]
        if host not in {"lazada.com.ph", "www.lazada.com.ph"}:
            return ""
        path = re.sub(r"/{2,}", "/", parts.path)
        if "/products/" not in path or not path.lower().endswith(".html"):
            return ""
        return urlunsplit(("https", "www.lazada.com.ph", path, "", ""))

    @classmethod
    def _normalize_category_url(cls, href):
        if not href:
            return ""
        absolute = urljoin(f"{cls.BASE_URL}/", str(href).strip())
        parts = urlsplit(absolute)
        host = parts.netloc.lower().split(":")[0]
        if host not in {"lazada.com.ph", "www.lazada.com.ph"}:
            return ""
        path = re.sub(r"/{2,}", "/", parts.path or "/")
        lowered_path = path.casefold()
        if path == "/" or "/products/" in lowered_path:
            return ""
        if any(
            blocked in lowered_path
            for blocked in ("/user/", "/customer/", "/helpcenter/", "/cart")
        ):
            return ""
        return urlunsplit(("https", "www.lazada.com.ph", path, parts.query, ""))

    def _build_search_url(self, page=1):
        query = urlencode({"q": self.keyword, "page": max(1, page)})
        return f"{self.BASE_URL}/catalog/?{query}"

    @classmethod
    def _build_category_page_url(cls, category_url, page=1):
        normalized = cls._normalize_category_url(category_url)
        if not normalized:
            return ""
        parts = urlsplit(normalized)
        query = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key.casefold() != "page"
        ]
        query.append(("page", str(max(1, page))))
        return urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), "")
        )

    @staticmethod
    def _parse_compact_number(text):
        if text is None:
            return 0
        value = re.sub(r"[^0-9.,kmbKMB]", "", str(text)).strip()
        if not value:
            return 0
        suffix = value[-1].lower() if value[-1].isalpha() else ""
        multiplier = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(suffix, 1)
        if suffix:
            value = value[:-1]
            if "," in value and "." not in value:
                value = value.replace(",", ".")
        else:
            value = value.replace(",", "")
        try:
            return int(float(value) * multiplier)
        except ValueError:
            return 0

    @staticmethod
    def _parse_rating(text):
        match = re.search(r"(?<!\d)([0-5](?:\.\d+)?)\s*(?:/\s*5)?(?!\d)", str(text or ""))
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def _clean_text(text):
        value = str(text or "")
        if any(marker in value for marker in ("â", "Â")):
            for source_encoding in ("cp1252", "latin-1"):
                try:
                    value = value.encode(source_encoding).decode("utf-8")
                    break
                except (UnicodeEncodeError, UnicodeDecodeError):
                    continue
        return re.sub(r"\s+", " ", value.replace("\u00a0", " ")).strip()

    @staticmethod
    def _first_text(root, selectors):
        for selector in selectors:
            try:
                for element in root.find_elements(By.CSS_SELECTOR, selector):
                    text = LazadaScraper._clean_text(element.text)
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

    @classmethod
    def _product_image_url(cls, root):
        gallery_selectors = (
            '.gallery-preview-panel-v2__content img',
            '.gallery-preview-panel__content img',
            '[data-qa-locator="product-image"] img',
            '[data-qa-locator="product-image"]',
        )
        return (
            cls._first_attribute(root, gallery_selectors, "src")
            or cls._first_attribute(root, gallery_selectors, "data-src")
            or cls._first_attribute(
                root,
                (
                    'meta[property="og:image"]',
                    'meta[name="twitter:image"]',
                    'meta[property="twitter:image"]',
                ),
                "content",
            )
        )

    def _save_debug_artifacts(self, kind):
        if not self.driver:
            return
        try:
            os.makedirs("debug", exist_ok=True)
            Path("debug", f"lazada_ph_{kind}.html").write_text(
                self.driver.page_source, encoding="utf-8"
            )
            self.driver.save_screenshot(str(Path("debug", f"lazada_ph_{kind}.png")))
            logging.warning("Saved %s diagnostics for %s", kind, self.driver.current_url)
        except Exception as exc:
            logging.warning("Could not save diagnostics: %s", exc)

    def _save_cookies(self):
        if not self.driver:
            return
        try:
            cookies = self.driver.get_cookies()
            if not cookies:
                logging.warning("Lazada returned no cookies; keeping the previous session file")
                return
            temporary_path = f"{self.cookies_file}.tmp"
            with open(temporary_path, "wb") as file:
                pickle.dump(cookies, file)
            os.replace(temporary_path, self.cookies_file)
            logging.info("Saved %s Lazada session cookie(s)", len(cookies))
        except Exception as exc:
            logging.warning("Could not save Lazada cookies: %s", exc)

    def _load_cookies(self):
        if self._cookies_loaded or not os.path.exists(self.cookies_file):
            return self._cookies_loaded
        try:
            if os.path.getsize(self.cookies_file) == 0:
                logging.warning("The Lazada session cookie file is empty; manual login is required")
                return False
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
                logging.info("Restored %s Lazada session cookie(s)", loaded)
            return self._cookies_loaded
        except Exception as exc:
            logging.warning("Could not load Lazada cookies: %s", exc)
            return False

    def _login_required(self):
        current_url = self.driver.current_url
        if isinstance(current_url, bytes):
            current_url = current_url.decode("utf-8", errors="replace")
        path = urlsplit(str(current_url or "")).path.casefold()
        if "/login" in path:
            return True
        selectors = (
            '#anonLogin',
            'a[href*="/user/login"]',
            'a[href*="login"]',
            '[data-spm-anchor-id*="login"]',
        )
        for selector in selectors:
            try:
                if any(element.is_displayed() for element in self.driver.find_elements(By.CSS_SELECTOR, selector)):
                    return True
            except Exception:
                continue
        return False

    def _wait_for_verification_clear(self, detector, label):
        if self.verification_timeout <= 0:
            return False
        logging.info("Waiting up to %s seconds for Lazada %s", self.verification_timeout, label)
        deadline = time.time() + self.verification_timeout
        while time.time() < deadline:
            if not detector():
                self._save_cookies()
                return True
            time.sleep(2)
        logging.warning("Lazada %s was not completed within the timeout", label)
        return False

    def _restore_or_request_login(self):
        restored = self._load_cookies()
        if restored:
            self.driver.refresh()
            try:
                WebDriverWait(self.driver, 25).until(
                    lambda driver: driver.execute_script("return document.readyState") == "complete"
                )
            except TimeoutException:
                logging.warning("Timed out refreshing the restored Lazada session")
            time.sleep(1)
        login_needed = self._login_required() or (
            not restored and not self.chrome_user_data_dir
        )
        if not login_needed:
            return True
        if not self.pause_for_verification:
            if self._wait_for_verification_clear(self._login_required, "login"):
                self._cookies_loaded = True
                return True
            logging.warning("Lazada is signed out and unattended login timed out")
            return False
        logging.info("Lazada is signed out. Log in manually in the opened Chrome window.")
        input("Press Enter after your Lazada account is logged in...")
        time.sleep(2)
        if self._login_required():
            logging.warning("Lazada still appears to be signed out")
            return False
        self._save_cookies()
        self._cookies_loaded = True
        return True

    def _load_existing_data(self):
        if not os.path.exists(self.out_file):
            return
        try:
            with open(self.out_file, "r", encoding="utf-8") as file:
                records = json.load(file)
            self.output_data = {
                item["link"]: item
                for item in records
                if isinstance(item, dict) and item.get("link")
            }
        except (OSError, ValueError, TypeError) as exc:
            logging.warning("Could not load %s: %s", self.out_file, exc)

    def _discovery_source_config(self):
        return {
            "keyword": self.keyword,
            "category_urls": list(self.category_urls),
        }

    def _has_discovery_source(self):
        return bool(self.keyword or self.category_urls)

    def _load_discovery_checkpoint(self):
        if not self._has_discovery_source() or not os.path.exists(
            self.discovery_checkpoint_file
        ):
            return
        try:
            with open(self.discovery_checkpoint_file, "r", encoding="utf-8") as file:
                checkpoint = json.load(file)
            if checkpoint.get("source") != self._discovery_source_config():
                logging.warning(
                    "Ignoring %s because its discovery source does not match this run",
                    self.discovery_checkpoint_file,
                )
                return
            products = checkpoint.get("products", [])
            self.discovery_products = {
                item["link"]: item
                for item in products
                if isinstance(item, dict) and item.get("link")
            }
            self.discovery_completed_pages = {
                str(source): max(0, int(page))
                for source, page in checkpoint.get("completed_pages", {}).items()
            }
            logging.info(
                "Restored %s discovered product(s) from %s",
                len(self.discovery_products),
                self.discovery_checkpoint_file,
            )
        except (OSError, ValueError, TypeError) as exc:
            logging.warning(
                "Could not load discovery checkpoint %s: %s",
                self.discovery_checkpoint_file,
                exc,
            )

    def _upsert_discovery_product(self, product, status=None, error=""):
        if not self._has_discovery_source() or not product.get("link"):
            return
        if (
            not product.get("discovery_source")
            and product["link"] not in self.discovery_products
        ):
            return
        existing = self.discovery_products.get(product["link"], {})
        entry = dict(existing)
        for key in (
            "link",
            "name",
            "price",
            "rating",
            "sold_count",
            "img",
            "location",
            "platform",
            "market",
            "currency",
            "discovery_source",
            "discovery_page",
            "reported_review_count",
            "written_reviews_collected",
            "review_status",
            "qualification",
        ):
            if key in product:
                entry[key] = product[key]
        entry["discovery_status"] = status or entry.get(
            "discovery_status", "discovered"
        )
        entry["last_error"] = str(error or "")
        entry["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        self.discovery_products[product["link"]] = entry
        product["discovery_status"] = entry["discovery_status"]

    def _save_discovery_checkpoint(self):
        if not self._has_discovery_source():
            return
        checkpoint_path = Path(self.discovery_checkpoint_file)
        temporary_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
        payload = {
            "version": 1,
            "source": self._discovery_source_config(),
            "completed_pages": self.discovery_completed_pages,
            "product_count": len(self.discovery_products),
            "products": list(self.discovery_products.values()),
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, checkpoint_path)
        logging.info(
            "Saved discovery checkpoint with %s product(s) to %s",
            len(self.discovery_products),
            self.discovery_checkpoint_file,
        )

    @staticmethod
    def _sync_review_counts(product):
        """Keep platform totals distinct from written comments actually saved."""
        reported = product.get("total_rating")
        if reported is None:
            reported = product.get("reported_review_count", 0)
        try:
            reported = int(reported or 0)
        except (TypeError, ValueError):
            reported = 0
        comments = product.get("comments", [])
        if not isinstance(comments, list):
            comments = []
        product["reported_review_count"] = reported
        product["written_reviews_collected"] = len(
            LazadaScraper._deduplicate_reviews(comments)
        )
        return product

    def _periodic_save(self):
        for product in self.output_data.values():
            self._sync_review_counts(product)
        output_path = Path(self.out_file)
        temporary_path = output_path.with_name(f"{output_path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(list(self.output_data.values()), file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, output_path)
        logging.info("Saved %s product(s) to %s", len(self.output_data), self.out_file)

    def _apply_research_quality(self, product):
        """Attach post-collection eligibility facts without hiding partial data."""
        self._sync_review_counts(product)
        comments = self._deduplicate_reviews(product.get("comments", []))
        usable = [review for review in comments if str(review.get("content") or "").strip()]
        per_star = Counter(
            str(int(review.get("source_rating_filter") or review.get("rating") or 0))
            for review in usable
            if int(review.get("source_rating_filter") or review.get("rating") or 0) in range(1, 6)
        )
        collected = len(usable)
        minimum = self.min_collected_reviews
        if product.get("qualification", {}).get("status") == "rejected":
            status = "not_collected"
        elif collected >= minimum:
            status = "eligible"
        else:
            status = "insufficient_written_reviews"
        product["research"] = {
            "stratum": self.research_stratum,
            "status": status,
            "minimum_collected_reviews": minimum,
            "usable_written_reviews": collected,
            "collected_by_star": {str(star): per_star.get(str(star), 0) for star in range(1, 6)},
            "review_status": product.get("review_status", "unknown"),
            "stop_reason": (product.get("review_checkpoint") or {}).get("stop_reason", ""),
            "filter_warning": product.get("review_filter_warning", ""),
        }
        return product["research"]

    def _write_collection_report(self):
        if not self.collection_report_file:
            return
        rows = []
        for product in self.output_data.values():
            quality = self._apply_research_quality(product)
            rows.append({
                "name": product.get("name", ""),
                "link": product.get("link", ""),
                "reported_review_count": product.get("reported_review_count", 0),
                **quality,
            })
        statuses = Counter(row["status"] for row in rows)
        payload = {
            "scope": "Lazada Philippines electronics and gadgets",
            "stratum": self.research_stratum,
            "minimum_collected_reviews": self.min_collected_reviews,
            "products_checked": len(rows),
            "status_counts": dict(sorted(statuses.items())),
            "products": rows,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        report_path = Path(self.collection_report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = report_path.with_name(f"{report_path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, report_path)
        logging.info("Saved research quality report to %s", report_path)

    def _checkpoint_reviews(self, product, reviews, page_number, stop_reason="", context="all"):
        product["comments"] = self._deduplicate_reviews(reviews)
        self._sync_review_counts(product)
        previous = product.get("review_checkpoint", {})
        product["review_checkpoint"] = {
            "context": context,
            "completed_pages": max(int(previous.get("completed_pages", 0)), page_number),
            "collected_reviews": len(product["comments"]),
            "stop_reason": stop_reason,
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        if product.get("link"):
            self.output_data[product["link"]] = product
        self._periodic_save()
        if (
            product.get("discovery_source")
            or product.get("link") in self.discovery_products
        ):
            self._upsert_discovery_product(product, "scraping")
            self._save_discovery_checkpoint()

    def _record_no_reviews(self, product):
        product["comments"] = []
        product["review_status"] = "no_reviews"
        product["review_collection_warning"] = "This product has no written reviews"
        logging.info("No written Lazada reviews found for %s", product.get("link"))
        self._checkpoint_reviews(
            product,
            [],
            page_number=0,
            stop_reason="no_reviews",
            context="all",
        )

    def _record_review_widget_missing(self, product):
        product["comments"] = []
        product["review_status"] = "review_widget_missing"
        product["review_collection_warning"] = (
            f"Lazada reports {product.get('total_rating', 0)} reviews, but the "
            "rendered review widget exposed no written-review records"
        )
        logging.warning("Lazada review widget did not finish loading for %s", product.get("link"))
        self._checkpoint_reviews(
            product,
            [],
            page_number=0,
            stop_reason="review_widget_missing",
            context="all",
        )

    def _qualify_product(self, product):
        """Apply detail-page rating/review thresholds and record the decision."""
        rating = float(product.get("rating", 0.0) or 0.0)
        reported_reviews = int(product.get("total_rating", 0) or 0)
        reasons = []
        if rating < self.min_rating:
            reasons.append("rating_below_minimum")
        if reported_reviews < self.min_reviews:
            reasons.append("review_count_below_minimum")

        product["reported_review_count"] = reported_reviews
        self._sync_review_counts(product)
        product["qualification"] = {
            "status": "rejected" if reasons else "qualified",
            "rating": rating,
            "reported_review_count": reported_reviews,
            "min_rating": self.min_rating,
            "min_reviews": self.min_reviews,
            "reasons": reasons,
            "checked_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        if not reasons:
            return True

        product["comments"] = []
        self._sync_review_counts(product)
        product["review_status"] = "not_qualified"
        product["review_checkpoint"] = {
            "context": "qualification",
            "completed_pages": 0,
            "collected_reviews": 0,
            "stop_reason": "not_qualified",
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        logging.info(
            "Rejected Lazada product %s: %s (rating=%s, reviews=%s)",
            product.get("link"),
            ", ".join(reasons),
            rating,
            reported_reviews,
        )
        return False

    def _saved_qualification_rejection_matches(self, product):
        qualification = product.get("qualification", {})
        return (
            qualification.get("status") == "rejected"
            and float(qualification.get("min_rating", -1)) == self.min_rating
            and int(qualification.get("min_reviews", -1)) == self.min_reviews
        )

    def _review_checkpoint_is_terminal(self, product):
        """Return whether a saved checkpoint exhausted the requested scope."""
        checkpoint = product.get("review_checkpoint", {})
        stop_reason = checkpoint.get("stop_reason")
        if product.get("review_status") == "no_reviews" or stop_reason == "no_reviews":
            return True
        if stop_reason not in {"next_button_disabled", "empty_review_page"}:
            return False
        context = str(checkpoint.get("context") or "")
        if context == "all":
            return True
        if self.rating is not None:
            return context == str(self.rating)
        return (
            context == str(self.RATING_COLLECTION_ORDER[-1])
            and product.get("applied_rating_filters") == list(self.RATING_COLLECTION_ORDER)
        )

    def _verification_detected(self):
        current_url = self.driver.current_url
        if isinstance(current_url, bytes):
            current_url = current_url.decode("utf-8", errors="replace")
        path = urlsplit(str(current_url or "")).path.casefold()
        title = (self.driver.title or "").casefold()
        body = ""
        try:
            body = self.driver.find_element(By.TAG_NAME, "body").text[:1500].casefold()
        except NoSuchElementException:
            pass
        blocked = any(token in path for token in ("/login", "/captcha", "/verify")) or any(
            token in f"{title} {body}"
            for token in ("security verification", "verify your identity", "unusual traffic")
        )
        return blocked

    def _check_verification(self):
        blocked = self._verification_detected()
        if blocked and self.pause_for_verification:
            logging.info("Lazada login or verification detected. Complete it in Chrome.")
            input("Press Enter after the Lazada page is ready...")
            time.sleep(2)
            blocked = self._verification_detected()
        elif blocked and self._wait_for_verification_clear(self._verification_detected, "verification"):
            blocked = False
        return blocked

    @staticmethod
    def _chrome_network_error(page_source):
        """Return Chrome's error code when navigation rendered an error page."""
        source = str(page_source or "")
        lowered = source.casefold()
        error_page_markers = (
            "main-frame-error",
            "sub-frame-error-details",
            '"errorcode"',
            "this site can’t be reached",
            "this site can't be reached",
        )
        if not any(marker in lowered for marker in error_page_markers):
            return ""
        match = re.search(r"\bERR_[A-Z0-9_]+\b", source, re.I)
        return match.group(0).upper() if match else "CHROME_NETWORK_ERROR"

    def _safe_get(self, url, check_verification=True):
        attempts = 3
        network_error = ""
        for attempt in range(1, attempts + 1):
            try:
                self.driver.get(url)
            except WebDriverException as exc:
                if attempt >= attempts:
                    raise RuntimeError(
                        f"Lazada navigation failed after {attempts} attempts: {exc}"
                    ) from exc
                logging.warning(
                    "Lazada navigation attempt %s/%s failed; retrying: %s",
                    attempt,
                    attempts,
                    exc,
                )
                time.sleep(attempt)
                continue

            try:
                WebDriverWait(self.driver, 25).until(
                    lambda driver: driver.execute_script("return document.readyState") == "complete"
                )
            except TimeoutException:
                logging.warning("Timed out waiting for %s", url)

            network_error = self._chrome_network_error(self.driver.page_source)
            if not network_error:
                break
            logging.warning(
                "Chrome returned %s for %s (attempt %s/%s)",
                network_error,
                url,
                attempt,
                attempts,
            )
            if attempt < attempts:
                time.sleep(attempt)
        else:  # pragma: no cover - the loop always exits through break or exhaustion
            network_error = network_error or "CHROME_NETWORK_ERROR"

        if network_error:
            self._save_debug_artifacts("network_error")
            raise RuntimeError(
                f"Lazada could not load after {attempts} attempts ({network_error}). "
                "Check the connection and retry the scan."
            )
        if check_verification:
            self._check_verification()

    def _prepare_session(self):
        """Restore saved authentication before waiting on Lazada login gates."""
        self._safe_get(self.BASE_URL, check_verification=False)
        return self._restore_or_request_login()

    def _product_link_from_card(self, card):
        try:
            if card.tag_name.casefold() == "a":
                link = self._normalize_product_url(card.get_attribute("href"))
                if link:
                    return link, card
        except Exception:
            pass
        for selector in self.PRODUCT_LINK_SELECTORS:
            try:
                for anchor in card.find_elements(By.CSS_SELECTOR, selector):
                    link = self._normalize_product_url(anchor.get_attribute("href"))
                    if link:
                        return link, anchor
            except Exception:
                continue
        return "", None

    def _find_product_cards(self):
        cards, seen = [], set()
        for selector in self.PRODUCT_CARD_SELECTORS:
            for card in self.driver.find_elements(By.CSS_SELECTOR, selector):
                link, _ = self._product_link_from_card(card)
                if link and link not in seen:
                    seen.add(link)
                    cards.append(card)
            if cards:
                return cards
        for anchor in self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/products/"]'):
            link = self._normalize_product_url(anchor.get_attribute("href"))
            if not link or link in seen:
                continue
            try:
                card = anchor.find_element(By.XPATH, "ancestor::div[@data-qa-locator='product-item'][1]")
            except NoSuchElementException:
                card = anchor
            seen.add(link)
            cards.append(card)
        return cards

    def _scroll_search_results(self):
        previous = -1
        unchanged = 0
        for _ in range(12):
            cards = self._find_product_cards()
            if len(cards) >= self.max_products:
                return cards
            unchanged = unchanged + 1 if len(cards) == previous else 0
            if unchanged >= 2:
                return cards
            previous = len(cards)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1)
        return self._find_product_cards()

    def _product_from_card(self, card):
        link, anchor = self._product_link_from_card(card)
        if not link:
            return None
        name = self._first_text(card, ('.RfADt', '[class*="title"]', '[data-qa-locator="product-item"] a'))
        if not name:
            name = (anchor.get_attribute("title") if anchor else "") or self._first_attribute(card, ("img",), "alt")
        price = self._first_text(card, ('.ooOxS', '[class*="price"]', '[data-qa-locator="product-price"]'))
        rating_text = self._first_text(card, ('[class*="rating"]', '[aria-label*="rating" i]'))
        sold = self._first_text(card, ('[class*="sold"]', '[class*="sale"]'))
        location = self._first_text(card, ('.oa6ri', '[class*="location"]'))
        image = self._first_attribute(card, ("img",), "src")
        if not image or image.startswith("data:image"):
            image = self._first_attribute(card, ("img",), "data-src")
        return {
            "link": link,
            "name": self._clean_text(name),
            "price": self._clean_text(price),
            "rating": self._parse_rating(rating_text),
            "sold_count": self._parse_compact_number(sold),
            "img": image,
            "location": self._clean_text(location),
            "platform": "lazada",
            "market": "PH",
            "currency": "PHP",
        }

    def _retrieve_search_products(self):
        products = [dict(product) for product in self.discovery_products.values()]
        seen = {product["link"] for product in products if product.get("link")}
        if len(products) >= self.max_products:
            logging.info(
                "Using %s product(s) already present in the discovery checkpoint",
                self.max_products,
            )
            return products[: self.max_products]
        discovery_sources = self.category_urls or [None]
        for category_url in discovery_sources:
            source_label = category_url or f"keyword:{self.keyword}"
            start_page = self.discovery_completed_pages.get(source_label, 0) + 1
            for page in range(start_page, self.pages + 1):
                if len(products) >= self.max_products:
                    break
                discovery_url = (
                    self._build_category_page_url(category_url, page)
                    if category_url
                    else self._build_search_url(page)
                )
                logging.info("Discovering Lazada products from %s page %s", source_label, page)
                self._safe_get(discovery_url)
                try:
                    WebDriverWait(self.driver, 20).until(
                        lambda _: bool(self._find_product_cards())
                    )
                except TimeoutException:
                    logging.warning(
                        "No Lazada product cards found for %s page %s",
                        source_label,
                        page,
                    )
                    self._save_debug_artifacts("search")
                    break
                for card in self._scroll_search_results():
                    if len(products) >= self.max_products:
                        break
                    product = self._product_from_card(card)
                    if product and product["link"] not in seen:
                        seen.add(product["link"])
                        product["discovery_source"] = source_label
                        product["discovery_page"] = page
                        self._upsert_discovery_product(product, "discovered")
                        products.append(product)
                # When the product cap is hit, the page may only have been
                # partially consumed. Leave it incomplete so a later run with
                # a higher cap safely rescans and deduplicates that page.
                if len(products) < self.max_products:
                    self.discovery_completed_pages[source_label] = page
                self._save_discovery_checkpoint()
            if len(products) >= self.max_products:
                break
        return products

    def _product_from_url(self, url):
        return {
            "link": url,
            "name": "",
            "price": "",
            "rating": 0.0,
            "sold_count": 0,
            "img": "",
            "location": "",
            "platform": "lazada",
            "market": "PH",
            "currency": "PHP",
        }

    def _scroll_to_reviews(self):
        for _ in range(45):
            container = self._review_container(timeout=1)
            if container:
                # Lazada's review module is intersection-observer driven. Move
                # beyond it and then return so the widget crosses the viewport
                # boundary instead of remaining a visible skeleton forever.
                for attempt in range(3):
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block:'start'}); "
                        "window.scrollBy(0, Math.max(500, Math.floor(window.innerHeight * 0.75)));",
                        container,
                    )
                    time.sleep(0.5)
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block:'center'}); window.scrollBy(0, -80);",
                        container,
                    )
                    try:
                        WebDriverWait(self.driver, 15, poll_frequency=0.5).until(
                            lambda driver: driver.execute_script(
                                """
                                const container = document.querySelector(
                                  '#module_product_review, .mod-reviews, '
                                  + '[data-qa-locator="product-review"], '
                                  + '[data-spm="product_review"]'
                                );
                                if (!container) return false;
                                const placeholder = container.querySelector(
                                  '.lazy-load-placeholder, .lazy-load-skeleton'
                                );
                                if (placeholder) return false;
                                const reviewItems = container.querySelectorAll(
                                  '.item, [data-qa-locator="review-item"], .review-item'
                                );
                                return reviewItems.length > 0;
                                """
                            )
                        )
                        return True
                    except TimeoutException:
                        logging.warning(
                            "Lazada review items did not load after activation attempt %s/3",
                            attempt + 1,
                        )
                logging.warning("Lazada review widget exposed no actual review items")
                return False
            self.driver.execute_script(
                "window.scrollBy(0, Math.max(700, Math.floor(window.innerHeight * 0.85)))"
            )
            time.sleep(0.45)
        return False

    def _review_container(self, timeout=3):
        for selector in self.REVIEW_CONTAINER_SELECTORS:
            try:
                return WebDriverWait(self.driver, timeout).until(
                    lambda driver: next(
                        (item for item in driver.find_elements(By.CSS_SELECTOR, selector) if item.is_displayed()),
                        False,
                    )
                )
            except TimeoutException:
                continue
        return None

    @staticmethod
    def _review_javascript():
        return r"""
        const itemSelectors = [
          '#module_product_review .item', '.mod-reviews .item',
          '[data-qa-locator="review-item"]', '.review-item'
        ];
        let items = [];
        for (const selector of itemSelectors) {
          items = Array.from(document.querySelectorAll(selector));
          if (items.length) break;
        }
        const text = (root, selectors) => {
          for (const selector of selectors) {
            const node = root.querySelector(selector);
            const value = ((node && (node.innerText || node.textContent)) || '').trim();
            if (value) return value;
          }
          return '';
        };
        return items.map((item, index) => {
          const ratingNode = item.querySelector(
            '[aria-label*="star" i], [class*="container-star"], [class*="rating"]'
          );
          const ratingLabel = ratingNode ? (
            ratingNode.getAttribute('aria-label') || ratingNode.getAttribute('title') || ''
          ) : '';
          const starNodes = item.querySelectorAll(
            '.item-middle .i-rate-star-item, .container-star img, '
            + '[class*="starCtns"] img, [class*="star-filled"], svg[class*="active"]'
          );
          const activeStars = Math.round(Array.from(starNodes).reduce((total, star) => {
            // Lazada draws two paths per star: a gray background followed by a
            // yellow foreground whose mask contains the visible percentage.
            // Looking only at the first path therefore reports every rating as 0.
            const paths = Array.from(star.querySelectorAll('path'));
            for (const path of paths) {
              const fill = path.style.fill || window.getComputedStyle(path).fill || '';
              const values = (fill.match(/[\d.]+/g) || []).map(Number);
              if (values.length < 3) continue;
              const spread = Math.max(values[0], values[1], values[2])
                - Math.min(values[0], values[1], values[2]);
              if (spread <= 25) continue;
              const mask = path.getAttribute('mask') || '';
              const percentage = mask.match(/half_([\d.]+)%/);
              return total + (percentage ? Number(percentage[1]) / 100 : 1);
            }
            // An SVG star with paths but no yellow foreground is an inactive
            // star. Only use the class-name fallback for non-SVG variants.
            if (paths.length) return total;
            const className = typeof star.className === 'string'
              ? star.className : ((star.className && star.className.baseVal) || '');
            return total + (/empty|inactive|off/i.test(className) ? 0 : 1);
          }, 0));
          return {
            comment_id: item.getAttribute('data-review-id') || item.getAttribute('data-id') || '',
            author: text(item, ['.reviewer', '.user-info .infos p:first-child', '.middle', '.user', '.review-user', '[data-qa-locator="review-author"]']),
            rating_label: ratingLabel,
            active_stars: activeStars,
            time: text(item, ['.user-info .time', '.title.right', '.review-date', '.date', '[data-qa-locator="review-date"]']),
            content: text(item, ['.item-content-main-content-reviews-item', '.item-content .content', '.content', '.review-content', '[data-qa-locator="review-content"]']),
            variation: text(item, ['.item-content-main-content-skuInfo', '.skuInfo', '.variation', '[class*="sku"]']),
            seller_respond: text(item, ['.seller-reply', '.seller-response', '[class*="reply"]']),
            helpful: text(item, ['.item-content-like', '.like', '.helpful', '[class*="helpful"]']),
            fallback_id: index
          };
        });
        """

    def _reviews_from_current_page(self):
        records = self.driver.execute_script(self._review_javascript()) or []
        reviews = []
        for record in records:
            rating = int(self._parse_rating(record.get("rating_label")) or record.get("active_stars") or 0)
            content = self._clean_text(record.get("content", ""))
            if not content and not self.include_empty:
                continue
            reviews.append(
                {
                    "comment_id": record.get("comment_id", ""),
                    "author": self._clean_text(record.get("author", "")),
                    "rating": rating,
                    "time": self._clean_text(record.get("time", "")),
                    "content": content,
                    "variation": self._clean_text(record.get("variation", "")),
                    "seller_respond": self._clean_text(record.get("seller_respond", "")),
                    "like_count": self._parse_compact_number(record.get("helpful", "")),
                }
            )
        return reviews

    def _current_review_signature(self):
        records = self.driver.execute_script(self._review_javascript()) or []
        return "|".join(
            str(item.get("comment_id") or f"{item.get('author')}:{item.get('time')}:{item.get('content')}")
            for item in records[:2]
        )

    def _next_review_page(self, previous_signature):
        self._last_pagination_stop_reason = ""
        for attempt in range(1, self.pagination_retries + 1):
            if self._current_review_signature() != previous_signature:
                return True
            next_button = None
            for selector in self.REVIEW_NEXT_SELECTORS:
                visible = [
                    item
                    for item in self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if item.is_displayed()
                ]
                if visible:
                    next_button = visible[-1]
                    break
            if next_button is None:
                self._last_pagination_stop_reason = "next_button_not_found"
            else:
                classes = (next_button.get_attribute("class") or "").casefold()
                aria_disabled = (next_button.get_attribute("aria-disabled") or "").casefold()
                disabled = (
                    next_button.get_attribute("disabled") is not None
                    or "disabled" in classes
                    or aria_disabled == "true"
                )
                if disabled:
                    self._last_pagination_stop_reason = "next_button_disabled"
                    return False
                else:
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block:'center'})", next_button
                    )
                    try:
                        next_button.click()
                    except Exception:
                        self.driver.execute_script("arguments[0].click()", next_button)
                    try:
                        WebDriverWait(self.driver, self.pagination_timeout).until(
                            lambda _: self._current_review_signature() != previous_signature
                        )
                        return True
                    except TimeoutException:
                        self._last_pagination_stop_reason = (
                            f"page_signature_timeout_attempt_{attempt}"
                        )
            if attempt < self.pagination_retries:
                logging.warning(
                    "Lazada review pagination attempt %s/%s failed: %s",
                    attempt,
                    self.pagination_retries,
                    self._last_pagination_stop_reason,
                )
                time.sleep(min(1.5 * attempt, 5))
        if self._last_pagination_stop_reason != "next_button_disabled":
            self._save_debug_artifacts("review_pagination_failure")
        return False

    def _click_rating_filter(self, rating):
        """Click a star option without retaining elements across DOM rebuilds."""
        click_option_script = r"""
        const rating = String(arguments[0]);
        const selectors = [
          '#module_product_review button',
          '#module_product_review [role="button"]',
          '#module_product_review [role="option"]',
          '#module_product_review .condition',
          '.next-menu-item', '.iweb-select-option', '[class*="dropdown"] li'
        ];
        const candidates = Array.from(new Set(
          selectors.flatMap((selector) => Array.from(document.querySelectorAll(selector)))
        ));
        const pattern = new RegExp('(^|\\D)' + rating + '\\s*(star|stars|★)(\\D|$)', 'i');
        for (const candidate of candidates) {
          const style = window.getComputedStyle(candidate);
          const visible = style.display !== 'none' && style.visibility !== 'hidden'
            && candidate.getClientRects().length > 0;
          const label = [
            candidate.innerText || candidate.textContent || '',
            candidate.getAttribute('aria-label') || '',
            candidate.getAttribute('title') || ''
          ].join(' ').replace(/\s+/g, ' ').trim();
          if (visible && pattern.test(label)) {
            candidate.click();
            return label;
          }
        }
        return '';
        """
        open_menu_script = r"""
        const selectors = [
          '#module_product_review .pdp-mod-filterSort-v2 .oper',
          '#module_product_review [class*="filter"] [role="button"]',
          '.mod-reviews .pdp-mod-filterSort-v2 .oper'
        ];
        for (const selector of selectors) {
          const control = Array.from(document.querySelectorAll(selector)).find((item) => {
            const style = window.getComputedStyle(item);
            return style.display !== 'none' && style.visibility !== 'hidden'
              && item.getClientRects().length > 0;
          });
          if (control) {
            control.click();
            return true;
          }
        }
        return false;
        """
        for attempt in range(1, self.pagination_retries + 1):
            try:
                label = self.driver.execute_script(click_option_script, rating)
                if not label:
                    opened = self.driver.execute_script(open_menu_script)
                    if opened:
                        time.sleep(0.5)
                    label = self.driver.execute_script(click_option_script, rating)
                if label:
                    time.sleep(1)
                    return True
            except StaleElementReferenceException:
                logging.warning(
                    "Lazada rebuilt the rating menu during attempt %s/%s; reacquiring it",
                    attempt,
                    self.pagination_retries,
                )
            if attempt < self.pagination_retries:
                time.sleep(min(0.5 * attempt, 2))
        logging.warning("Could not locate the %s-star Lazada review filter", rating)
        return False

    @staticmethod
    def _reviews_match_rating_filter(reviews, expected_rating):
        ratings = [review.get("rating", 0) for review in reviews if review.get("rating", 0)]
        return bool(ratings) and all(rating == expected_rating for rating in ratings)

    @staticmethod
    def _review_matches_source_filter(review, source_filter):
        recorded_filter = review.get("source_rating_filter")
        if recorded_filter is not None:
            return recorded_filter == source_filter
        return int(review.get("rating", 0) or 0) == int(source_filter)

    def _order_reviews_low_to_high(self, reviews):
        return sorted(
            self._deduplicate_reviews(reviews),
            key=lambda review: int(
                review.get("source_rating_filter") or review.get("rating", 0) or 6
            ),
        )

    def _collect_reviews_lowest_first(self, product, max_reviews):
        """Fill a review sample from verified 3-star feedback first."""
        if max_reviews <= 0:
            product["applied_rating_filters"] = []
            return []

        reviews = []
        applied_filters = []
        for star in self.RATING_COLLECTION_ORDER:
            remaining = max_reviews - len(reviews)
            if remaining <= 0:
                break
            if not self._click_rating_filter(star):
                product["review_filter_warning"] = (
                    f"Could not identify or apply the {star}-star Lazada filter; "
                    "three-star-first collection stopped"
                )
                break

            applied_filters.append(star)
            star_reviews = self._collect_reviews(
                remaining,
                product=product,
                source_filter=star,
            )
            if not star_reviews:
                continue
            if not self._reviews_match_rating_filter(star_reviews, star):
                product["review_filter_warning"] = (
                    f"Lazada did not return verified {star}-star reviews; "
                    "mismatched results were discarded"
                )
                break
            for review in star_reviews:
                review["source_rating_filter"] = star
            reviews.extend(star_reviews)
            reviews = self._order_reviews_low_to_high(reviews)

        product["applied_rating_filters"] = applied_filters
        return self._order_reviews_low_to_high(reviews)[:max_reviews]

    def _collect_reviews(self, max_reviews, product=None, source_filter=None):
        if max_reviews <= 0 or self._review_container() is None:
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
        stop_reason = "target_reached" if len(collected) >= max_reviews else ""
        page_number = 0
        initial = min(len(collected), max_reviews)
        progress_disabled = not getattr(sys.stderr, "isatty", lambda: False)()
        with tqdm(
            total=max_reviews,
            initial=initial,
            desc="Collecting Lazada reviews",
            disable=progress_disabled,
        ) as progress:
            while len(collected) < max_reviews and page_number < self.max_review_pages:
                page_number += 1
                page_reviews = self._reviews_from_current_page()
                if not page_reviews:
                    try:
                        WebDriverWait(
                            self.driver,
                            self.pagination_timeout,
                            poll_frequency=0.5,
                        ).until(lambda _: bool(self._reviews_from_current_page()))
                        page_reviews = self._reviews_from_current_page()
                    except TimeoutException:
                        stop_reason = "empty_review_page"
                        self._save_debug_artifacts("reviews")
                        break
                for review in page_reviews:
                    if len(collected) >= max_reviews:
                        break
                    if source_filter is not None:
                        review["source_rating_filter"] = source_filter
                    identity = review.get("comment_id") or (
                        review["author"], review["time"], review["content"], review["rating"]
                    )
                    if identity in seen:
                        continue
                    seen.add(identity)
                    collected.append(review)
                    progress.update(1)
                if product and page_number % self.checkpoint_every == 0:
                    if source_filter is None:
                        checkpoint_reviews = collected
                    else:
                        checkpoint_reviews = [
                            review
                            for review in existing_reviews
                            if not self._review_matches_source_filter(review, source_filter)
                        ] + collected
                    self._checkpoint_reviews(
                        product, checkpoint_reviews, page_number, context=str(source_filter or "all")
                    )
                if len(collected) >= max_reviews:
                    stop_reason = "target_reached"
                    break
                signature = self._current_review_signature()
                if not signature:
                    stop_reason = "missing_page_signature"
                    self._save_debug_artifacts("review_pagination_failure")
                    break
                if not self._next_review_page(signature):
                    stop_reason = self._last_pagination_stop_reason or "page_transition_failed"
                    break
            else:
                if page_number >= self.max_review_pages and len(collected) < max_reviews:
                    stop_reason = "max_review_pages_reached"
        if stop_reason and stop_reason != "target_reached":
            logging.info(
                "Lazada review pagination stopped after page %s with %s review(s): %s",
                page_number,
                len(collected),
                stop_reason,
            )
        if product:
            if source_filter is None:
                checkpoint_reviews = collected
            else:
                checkpoint_reviews = [
                    review
                    for review in existing_reviews
                    if not self._review_matches_source_filter(review, source_filter)
                ] + collected
            self._checkpoint_reviews(
                product,
                checkpoint_reviews,
                page_number,
                stop_reason=stop_reason,
                context=str(source_filter or "all"),
            )
        return collected

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

    def _scrape_details(self, product):
        product.pop("review_filter_warning", None)
        product.pop("review_collection_warning", None)
        self._safe_get(product["link"])
        product["name"] = self._first_text(
            self.driver, ('h1.pdp-mod-product-badge-title', 'h1[data-qa-locator="product-title"]', 'h1')
        ) or product.get("name", "")
        product["price"] = self._first_text(
            self.driver,
            (
                '.pdp-v2-product-price-content-salePrice',
                '.pdp-price_type_normal',
                '.pdp-product-price .pdp-price',
                '[data-qa-locator="product-price"]',
            ),
        ) or product.get("price", "")
        rating_text = self._first_text(
            self.driver,
            (
                '.container-star-v2-score',
                '.score-average',
                '.pdp-review-summary__stars',
                '[data-qa-locator="product-rating"]',
            ),
        )
        product["rating"] = self._parse_rating(rating_text) or product.get("rating", 0.0)
        product["brand"] = self._first_text(
            self.driver, ('.pdp-product-brand__brand-link', '[data-qa-locator="product-brand"]')
        )
        product["seller"] = self._first_text(
            self.driver, ('.seller-name__detail', '.pdp-link_size_l', '[data-qa-locator="seller-name"]')
        )
        product["description"] = self._first_text(
            self.driver, ('#module_product_detail', '.pdp-product-desc', '[data-qa-locator="product-description"]')
        )
        product["img"] = product.get("img") or self._product_image_url(self.driver)

        if not self._scroll_to_reviews():
            logging.warning("Review section not found for %s", product["link"])
            self._save_debug_artifacts("reviews")
            total_text = self._first_text(
                self.driver,
                (
                    '.container-star-v2-count',
                    '.pdp-review-summary__link',
                    '[data-qa-locator="review-count"]',
                ),
            )
            product["total_rating"] = self._parse_compact_number(total_text)
            if not self._qualify_product(product):
                return
            if product["total_rating"] > 0:
                self._record_review_widget_missing(product)
            else:
                self._record_no_reviews(product)
            return

        total_text = self._first_text(
            self.driver,
            (
                '#module_product_review .title-text',
                '.container-star-v2-count',
                '.mod-reviews .count',
                '.pdp-review-summary__link',
                '[data-qa-locator="review-count"]',
            ),
        )
        # The score in the lazily rendered review module is the authoritative
        # detail-page value and may not have existed during the first read.
        verified_rating_text = self._first_text(
            self.driver,
            (
                '#module_product_review .score-average',
                '.container-star-v2-score',
                '.score-average',
                '[data-qa-locator="product-rating"]',
            ),
        )
        product["rating"] = (
            self._parse_rating(verified_rating_text) or product.get("rating", 0.0)
        )
        product["total_rating"] = self._parse_compact_number(total_text)
        if not self._qualify_product(product):
            return
        if product["total_rating"] == 0 and not self._reviews_from_current_page():
            self._record_no_reviews(product)
            return
        product["review_status"] = "collecting"
        if self.all_star_types:
            reviews = []
            applied_filters = []
            for star in self.RATING_COLLECTION_ORDER:
                if not self._click_rating_filter(star):
                    product["review_filter_warning"] = (
                        f"Could not identify or apply the {star}-star Lazada filter"
                    )
                    break
                star_reviews = self._collect_reviews(
                    self.star_limit_per_type, product=product, source_filter=star
                )
                if not self._reviews_match_rating_filter(star_reviews, star):
                    product["review_filter_warning"] = (
                        f"Lazada did not return verified {star}-star reviews; "
                        "balanced collection stopped to prevent mislabeled data"
                    )
                    break
                applied_filters.append(star)
                for review in star_reviews:
                    review["source_rating_filter"] = star
                reviews.extend(star_reviews)
            product["applied_rating_filters"] = applied_filters
            if applied_filters:
                product["comments"] = self._order_reviews_low_to_high(reviews)
                product["review_status"] = "complete"
            else:
                product.setdefault(
                    "review_filter_warning",
                    "No star filters were identified in the rendered Lazada review widget",
                )
                product["comments"] = []
            return
        if self.rating is not None:
            if not self._click_rating_filter(self.rating):
                product["review_filter_warning"] = (
                    f"Could not identify or apply the {self.rating}-star Lazada filter"
                )
                product["comments"] = []
                return
            filtered_reviews = self._collect_reviews(
                self.review_limit, product=product, source_filter=self.rating
            )
            if not self._reviews_match_rating_filter(filtered_reviews, self.rating):
                product["review_filter_warning"] = (
                    f"Lazada did not return verified {self.rating}-star reviews; "
                    "the unverified results were discarded"
                )
                product["comments"] = []
                return
            product["comments"] = self._order_reviews_low_to_high(filtered_reviews)
            product["review_status"] = "complete"
            return

        product["review_order"] = "marketplace_default"
        product["sampling_method"] = "marketplace_default_unfiltered"
        product["sampling_representative"] = None
        product["comments"] = self._collect_reviews(self.review_limit, product=product)
        if product["comments"]:
            product["review_status"] = "complete"
        elif product.get("total_rating", 0) > 0:
            self._record_review_widget_missing(product)
        else:
            self._record_no_reviews(product)

    def execute(self):
        major = self._detect_chrome_major()
        kwargs = {"options": self.options, "enable_cdp_events": False, "headless": False}
        if major:
            kwargs["version_main"] = major
            cached = self._cached_driver_for_major(major)
            if cached:
                kwargs["driver_executable_path"] = cached
        logging.info("Starting Chrome for Lazada Philippines")
        self.driver = SafeChrome(**kwargs)
        _position_scraper_window(self.driver)
        try:
            self._prepare_session()
            if self.product_urls:
                products = [self._product_from_url(url) for url in self.product_urls]
            else:
                products = self._retrieve_search_products()
            if not products:
                raise RuntimeError("No real Lazada products were extracted; inspect debug artifacts")
            for product in tqdm(products, desc="Processing Lazada products"):
                link = product["link"]
                existing = self.output_data.get(link)
                if existing:
                    if self._saved_qualification_rejection_matches(existing):
                        logging.info(
                            "Skipping previously rejected Lazada product %s for the same qualification thresholds",
                            link,
                        )
                        self._upsert_discovery_product(existing, "rejected")
                        self._save_discovery_checkpoint()
                        continue
                    expected = (
                        self.star_limit_per_type * 5
                        if self.all_star_types
                        else self.review_limit
                    )
                    reached_end = self._review_checkpoint_is_terminal(existing)
                    if (
                        expected > 0
                        and (len(existing.get("comments", [])) >= expected or reached_end)
                        and not existing.get("review_filter_warning")
                    ):
                        self._upsert_discovery_product(existing, "complete")
                        self._save_discovery_checkpoint()
                        continue
                    logging.info(
                        "Retrying incomplete Lazada result for %s (%s/%s reviews)",
                        link,
                        len(existing.get("comments", [])),
                        expected,
                    )
                    product = existing
                self._upsert_discovery_product(product, "checking")
                self._save_discovery_checkpoint()
                detail_error = None
                try:
                    self._scrape_details(product)
                except Exception as exc:
                    detail_error = exc
                    logging.warning("Detail scrape failed for %s: %s", link, exc, exc_info=True)
                    product.setdefault("comments", [])
                self.output_data[link] = product
                if self.research_stratum or self.min_collected_reviews:
                    self._apply_research_quality(product)
                self._periodic_save()
                self._write_collection_report()
                qualification_status = product.get("qualification", {}).get("status")
                if qualification_status == "rejected":
                    discovery_status = "rejected"
                elif (
                    product.get("review_status") in {"complete", "no_reviews"}
                    and not product.get("review_filter_warning")
                ):
                    discovery_status = "complete"
                elif qualification_status == "qualified":
                    discovery_status = "retryable_error"
                else:
                    discovery_status = "retryable_error"
                self._upsert_discovery_product(
                    product,
                    discovery_status,
                    error=detail_error or product.get("review_collection_warning", ""),
                )
                self._save_discovery_checkpoint()
        finally:
            self._save_cookies()
            if self.driver:
                self.driver.quit()
        self._write_collection_report()
        self._periodic_save()
        return list(self.output_data.values())


def build_parser():
    parser = argparse.ArgumentParser(description="Scrape products and reviews from Lazada Philippines")
    parser.add_argument("keyword", nargs="?", default="", help="Lazada search keyword")
    parser.add_argument("-n", "--num", type=int, default=10, help="Maximum products")
    parser.add_argument("-r", "--review-limit", type=int, default=30, help="Written reviews per product")
    parser.add_argument("--pages", type=int, default=1, help="Maximum search-result pages")
    parser.add_argument("--product-url", action="append", default=[], help="Direct Lazada PH product URL; repeatable")
    parser.add_argument("--category-url", action="append", default=[], help="Lazada PH category URL; repeatable")
    parser.add_argument("--rating", type=int, choices=range(1, 6), help="Filter reviews by star rating")
    parser.add_argument("--all-star-types", action="store_true", help="Collect each 1-5 star filter")
    parser.add_argument("--star-limit-per-type", type=int, default=10, help="Written reviews per star filter")
    parser.add_argument("--include-empty", action="store_true", help="Include ratings with no written comment")
    parser.add_argument("--chrome-user-data-dir", help="Optional Chrome user-data directory")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("--no-verification-pause", action="store_true", help="Do not prompt on a verification page")
    parser.add_argument("--verification-timeout", type=int, default=0, help="Seconds to wait for unattended login/verification")
    parser.add_argument("--pagination-retries", type=int, default=3, help="Retries for a failed review-page transition")
    parser.add_argument("--pagination-timeout", type=float, default=15, help="Seconds to wait for each review-page transition")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save review progress every N pages")
    parser.add_argument("--max-review-pages", type=int, default=5000, help="Safety cap for review pages per filter")
    parser.add_argument("--min-rating", type=float, default=0.0, help="Minimum detail-page product rating (0-5)")
    parser.add_argument("--min-reviews", type=int, default=0, help="Minimum detail-page reported review count")
    parser.add_argument("--discovery-checkpoint", help="Optional discovery checkpoint JSON path")
    parser.add_argument("--research-stratum", default="", help="Research cohort stratum stored with every product")
    parser.add_argument("--min-collected-reviews", type=int, default=0, help="Minimum usable written reviews required after collection")
    parser.add_argument("--collection-report", help="Research collection-quality JSON report")
    return parser


def main():
    args = build_parser().parse_args()
    scraper = LazadaScraper(
        keyword=args.keyword,
        max_products=args.num,
        review_limit=args.review_limit,
        pages=args.pages,
        product_urls=args.product_url,
        category_urls=args.category_url,
        rating=args.rating,
        all_star_types=args.all_star_types,
        star_limit_per_type=args.star_limit_per_type,
        include_empty=args.include_empty,
        chrome_user_data_dir=args.chrome_user_data_dir,
        output=args.output,
        pause_for_verification=not args.no_verification_pause,
        verification_timeout=args.verification_timeout,
        pagination_retries=args.pagination_retries,
        pagination_timeout=args.pagination_timeout,
        checkpoint_every=args.checkpoint_every,
        max_review_pages=args.max_review_pages,
        min_rating=args.min_rating,
        min_reviews=args.min_reviews,
        discovery_checkpoint=args.discovery_checkpoint,
        research_stratum=args.research_stratum,
        min_collected_reviews=args.min_collected_reviews,
        collection_report=args.collection_report,
    )
    scraper.execute()


if __name__ == "__main__":
    main()
