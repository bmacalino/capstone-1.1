import os
import sys
import time
import json
import logging
import argparse
import re
import datetime
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from tqdm import tqdm


def _progress(iterable=None, **kwargs):
    """Keep dashboard subprocess output quiet while retaining terminal progress bars."""
    kwargs.setdefault("disable", not sys.stderr.isatty())
    return tqdm(iterable, **kwargs)


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

class ShopeeScraper:
    """Selenium scraper configured exclusively for Shopee Philippines."""

    SITE = "shopee.ph"
    BASE_URL = f"https://{SITE}"
    MARKET = "PH"
    CURRENCY = "PHP"

    PRODUCT_CARD_SELECTORS = (
        '[data-sqe="item"]',
        '.shopee-search-item-result__item',
        'section ul > li',
    )
    PRODUCT_LINK_SELECTORS = (
        'a[data-sqe="link"]',
        'a.contents',
        'a[href*="-i."]',
        'a[href*="/product/"]',
    )

    def __init__(
        self,
        search_term,
        max_products,
        index_only,
        review_limit,
        all_star_types=False,
        star_limit_per_type=10,
        product_urls=None,
        output=None,
        no_prompt=False,
        verification_timeout=900,
    ):
        self.driver = None
        self.search_term = str(search_term or "").strip()
        self.max_products = max(0, max_products)
        self.index_only = index_only
        self.review_limit = max(0, review_limit)
        self.all_star_types = all_star_types
        self.star_limit_per_type = max(0, star_limit_per_type)
        self.no_prompt = bool(no_prompt)
        self.verification_timeout = max(1, int(verification_timeout))
        self.product_urls = [self._normalize_product_url(url) for url in (product_urls or [])]
        self.product_urls = [url for url in self.product_urls if url]
        if not self.search_term and not self.product_urls:
            raise ValueError("Provide a search keyword or a Shopee PH product URL")

        self._setup_logging()
        self.options = uc.ChromeOptions()
        self._configure_options()
        self.output_data = {}
        slug_source = "products" if self.product_urls else self.search_term
        slug = re.sub(r"[^a-z0-9_]+", "_", slug_source.lower()).strip("_")
        self.out_file = output or f"shopee_ph_{slug or 'search'}.json"
        self._load_existing_data()

    def _setup_logging(self):
        os.makedirs("logs", exist_ok=True)
        if logging.getLogger().handlers:
            return
        log_filename = datetime.datetime.now().strftime("shopee_ph_%d_%m_%H_%M_%S.log")
        log_filepath = os.path.join("logs", log_filename)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_filepath),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def _configure_options(self):
        if sys.platform.startswith('linux'):
            self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_argument("--window-size=900,700")
        self.options.add_argument("--lang=en-PH")

    @staticmethod
    def _detect_chrome_major():
        """Return the installed Chrome major version on Windows when detectable."""
        if not sys.platform.startswith("win"):
            return None
        application_dirs = [
            os.path.join(os.environ.get("PROGRAMFILES", ""), "Google", "Chrome", "Application"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Google", "Chrome", "Application"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application"),
        ]
        versions = []
        for application_dir in application_dirs:
            if not os.path.isdir(application_dir):
                continue
            try:
                versions.extend(
                    tuple(int(part) for part in entry.split("."))
                    for entry in os.listdir(application_dir)
                    if re.fullmatch(r"\d+(?:\.\d+){1,3}", entry)
                )
            except OSError:
                continue
        return max(versions)[0] if versions else None

    @classmethod
    def _normalize_product_url(cls, value):
        """Return a canonical Shopee PH product URL, or reject another market."""
        if not value:
            return ""
        absolute = urljoin(f"{cls.BASE_URL}/", str(value).strip())
        parts = urlsplit(absolute)
        host = parts.netloc.casefold().split(":", 1)[0].removeprefix("www.")
        if host != cls.SITE:
            raise ValueError("Only shopee.ph product links are supported")
        if not re.search(r"(?:-i\.\d+\.\d+|/product/\d+/\d+)", parts.path, re.I):
            raise ValueError("The URL is not a recognized Shopee PH product link")
        return urlunsplit(("https", cls.SITE, parts.path, "", ""))

    def _build_search_url(self, page=0):
        query = urlencode(
            {
                "keyword": self.search_term,
                "page": max(0, page),
                "sortBy": "sales",
            }
        )
        return f"{self.BASE_URL}/search?{query}"

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    def _save_debug_artifacts(self, kind):
        """Save the rendered Shopee page when a required section is missing."""
        try:
            os.makedirs("debug", exist_ok=True)
            safe_kind = re.sub(r"[^a-z0-9_]+", "_", kind.casefold()).strip("_")
            html_path = os.path.join("debug", f"shopee_ph_{safe_kind}.html")
            image_path = os.path.join("debug", f"shopee_ph_{safe_kind}.png")
            with open(html_path, "w", encoding="utf-8") as file:
                file.write(self.driver.page_source)
            self.driver.save_screenshot(image_path)
            logging.warning(
                "Saved Shopee PH diagnostics for %s (URL=%s, title=%s).",
                kind,
                self.driver.current_url,
                self.driver.title,
            )
        except Exception as exc:
            logging.warning("Could not save Shopee PH diagnostics: %s", exc)

    def _load_existing_data(self):
        if os.path.exists(self.out_file):
            try:
                with open(self.out_file, 'r', encoding='utf-8') as f:
                    self.output_data = {
                        item['link']: item for item in json.load(f)
                    }
            except:
                self.output_data = {}

    def _periodic_save(self):
        """Save the current output data to JSON."""
        try:
            with open(self.out_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.output_data.values()), f, ensure_ascii=False, indent=2)
            logging.info("Periodic save successful.")
        except Exception as e:
            logging.warning(f"Periodic save failed: {e}")

    def _scrape_missing_comments(self):
        """Scrape comments for already known products that have no comments."""
        logging.info("Scraping missing comments from existing data...")
        for link, product in list(self.output_data.items()):
            if product.get("comments") == []:
                try:
                    self._scrape_details(product)
                    self.output_data[link] = product
                    self._periodic_save()
                except Exception as e:
                    logging.warning(f"Error scraping comments for {link}: {e}")

    @staticmethod
    def _first_text(root, selectors):
        for selector in selectors:
            try:
                element = root.find_element(By.CSS_SELECTOR, selector)
                text = (
                    element.text
                    or element.get_attribute("innerText")
                    or element.get_attribute("textContent")
                    or ""
                ).strip()
                if text:
                    return text
            except (NoSuchElementException, AttributeError):
                continue
        return ""

    @staticmethod
    def _element_text(element):
        return (
            element.text
            or element.get_attribute("innerText")
            or element.get_attribute("textContent")
            or ""
        ).strip()

    @staticmethod
    def _first_attribute(root, selectors, attribute):
        for selector in selectors:
            try:
                value = root.find_element(By.CSS_SELECTOR, selector).get_attribute(attribute)
                if value:
                    return value
            except (NoSuchElementException, AttributeError):
                continue
        return ""

    def _product_link_from_card(self, card):
        try:
            if card.tag_name.casefold() == "a":
                link = self._normalize_product_url(card.get_attribute("href"))
                if link:
                    return link, card
        except (AttributeError, ValueError):
            pass
        for selector in self.PRODUCT_LINK_SELECTORS:
            try:
                anchors = card.find_elements(By.CSS_SELECTOR, selector)
            except Exception:
                continue
            for anchor in anchors:
                try:
                    link = self._normalize_product_url(anchor.get_attribute("href"))
                except ValueError:
                    continue
                if link:
                    return link, anchor
        return "", None

    def _find_product_cards(self):
        cards = []
        seen = set()
        for selector in self.PRODUCT_CARD_SELECTORS:
            for card in self.driver.find_elements(By.CSS_SELECTOR, selector):
                link, _ = self._product_link_from_card(card)
                if link and link not in seen:
                    cards.append(card)
                    seen.add(link)
            if cards:
                break
        if not cards:
            for anchor in self.driver.find_elements(
                By.CSS_SELECTOR, 'a[href*="-i."], a[href*="/product/"]'
            ):
                try:
                    link = self._normalize_product_url(anchor.get_attribute("href"))
                except ValueError:
                    continue
                if link and link not in seen:
                    cards.append(anchor)
                    seen.add(link)
        return cards

    @staticmethod
    def _price_from_text(text):
        match = re.search(
            r"₱\s*[\d,.]+(?:\s*-\s*₱?\s*[\d,.]+)?", str(text).replace("\n", " ")
        )
        return re.sub(r"\s+", "", match.group(0)) if match else ""

    def _product_from_card(self, card):
        link, anchor = self._product_link_from_card(card)
        if not link:
            return None
        name = self._first_text(
            card,
            ('[data-sqe="name"]', '[class*="line-clamp-2"]', '[class*="break-words"]'),
        )
        if not name:
            name = (anchor.get_attribute("title") if anchor else "") or self._first_attribute(
                card, ("img",), "alt"
            )
        card_text = getattr(card, "text", "") or ""
        price = self._first_text(
            card,
            ('[class*="items-baseline"]', '[class*="text-shopee-primary"]', '[class*="price"]'),
        )
        if "₱" not in price:
            price = self._price_from_text(card_text)
        rating = self._first_text(
            card,
            ('[aria-label*="rating" i]', '[class*="rating"]'),
        )
        rating_match = re.search(r"(?<!\d)[0-5](?:\.\d)?(?!\d)", rating)
        image = self._first_attribute(card, ("img",), "src")
        if not image or image.startswith("data:image"):
            image = self._first_attribute(card, ("img",), "data-src")
        shipping = next(
            (line.strip() for line in card_text.splitlines() if "shipping" in line.casefold()),
            "",
        )
        location = self._first_text(
            card,
            ('[class*="text-shopee-black54"][class*="truncate"]', '[class*="location"]'),
        )
        return {
            "link": link,
            "name": name,
            "price": re.sub(r"\s+", "", price) if price else "",
            "rating": rating_match.group(0) if rating_match else "",
            "img": image,
            "shipping": shipping,
            "location": location,
            "platform": "shopee",
            "market": self.MARKET,
            "currency": self.CURRENCY,
        }

    def _retrieve_products(self):
        logging.info("Retrieving Shopee PH product data...")
        product_elements = self._find_product_cards()
        if not product_elements:
            logging.warning("Could not locate Shopee PH product cards.")
        result = []
        seen = set()
        for idx, card in enumerate(product_elements):
            if idx >= self.max_products:
                break
            try:
                product = self._product_from_card(card)
                if product and product["link"] not in seen:
                    result.append(product)
                    seen.add(product["link"])
            except Exception as e:
                logging.warning(f"Skipping item due to error: {e}")
        return result

    def _check_captcha(self):
        """Check for captcha and wait if detected"""
        blacklist = ["login", "captcha","verify","security","check","auth","error"]
        current_url = self.driver.current_url
        if isinstance(current_url, bytes):
            current_url = current_url.decode("utf-8", errors="replace")
        if any(x in current_url.lower() for x in blacklist):
            logging.info("Captcha/Login detected! Please solve it...")
            try:
                page_text = self.driver.find_element(By.TAG_NAME, "body").text.casefold()
            except Exception:
                page_text = ""
            if "try again later" in page_text or "verification can't be completed" in page_text:
                raise TimeoutException(
                    "SHOPEE_VERIFICATION_BLOCKED: Shopee could not complete verification"
                )
            if self.no_prompt:
                deadline = time.monotonic() + self.verification_timeout
                while time.monotonic() < deadline:
                    current_url = self.driver.current_url
                    if isinstance(current_url, bytes):
                        current_url = current_url.decode("utf-8", errors="replace")
                    try:
                        page_text = self.driver.find_element(By.TAG_NAME, "body").text.casefold()
                    except Exception:
                        page_text = ""
                    if "try again later" in page_text or "verification can't be completed" in page_text:
                        raise TimeoutException(
                            "SHOPEE_VERIFICATION_BLOCKED: Shopee could not complete verification"
                        )
                    if not any(x in current_url.lower() for x in blacklist):
                        break
                    time.sleep(1)
                else:
                    raise TimeoutError(
                        f"Shopee verification was not completed within {self.verification_timeout} seconds"
                    )
            else:
                input("Press Enter after solving the captcha...")
            time.sleep(5)  # Wait for page to settle after captcha
            return True
        return False

    def _safe_get(self, url):
        """Wrapper for driver.get() with captcha checking"""
        self.driver.get(url)
        time.sleep(5)  # Wait for page to load
        while self._check_captcha():
            logging.info("Retrying after captcha...")
            self.driver.get(url)
            time.sleep(5)  # Wait for page to load
            self._check_captcha()
            self.driver.implicitly_wait(3)

    def _scrape_page(self):
        logging.info("Loading Shopee Philippines search page...")
        url = self._build_search_url()
        self._safe_get(url)
        self.driver.implicitly_wait(5)
        return self._retrieve_products()

    @staticmethod
    def _parse_star_text(text):
        """Parse PH/English counts such as 1,234, 1.2K, 2M, or 1,2k."""
        value = re.sub(r"[^0-9.,kmbKMB]", "", str(text or "")).strip()
        if not value:
            return 0
        suffix = value[-1].casefold() if value[-1].isalpha() else ""
        if suffix:
            value = value[:-1]
            multiplier = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(suffix, 1)
            if "," in value and "." not in value:
                value = value.replace(",", ".")
        else:
            multiplier = 1
            value = value.replace(",", "")
        try:
            return int(float(value) * multiplier)
        except ValueError:
            return 0

    def _parse_rating_filter(self, text):
        """Parse PH rating labels with parenthesized or multiline counts."""
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        count_match = re.search(r"\(([^)]*)\)", normalized)
        count = self._parse_star_text(count_match.group(1)) if count_match else 0
        star_match = re.match(r"^([1-5])\s*stars?\b", normalized, re.I)
        if star_match:
            if not count_match:
                count = self._parse_star_text(normalized[star_match.end():])
            return f"{star_match.group(1)}_star", count
        lowered = normalized.casefold()
        if lowered.startswith("all"):
            if not count_match:
                count = self._parse_star_text(normalized[3:])
            return "all", count
        if "comment" in lowered:
            if not count_match:
                count = self._parse_star_text(normalized)
            return "commented", count
        if any(term in lowered for term in ("media", "photo", "video")):
            if not count_match:
                count = self._parse_star_text(normalized)
            return "media", count
        return "", count

    @staticmethod
    def _total_rating_from_breakdown(detailed_rating):
        all_count = int(detailed_rating.get("all") or 0)
        star_total = sum(
            int(count or 0)
            for key, count in detailed_rating.items()
            if key.endswith("_star")
        )
        return all_count or star_total

    def _product_from_url(self, url):
        return {
            "link": self._normalize_product_url(url),
            "name": "",
            "price": "",
            "rating": "",
            "img": "",
            "shipping": "",
            "location": "",
            "platform": "shopee",
            "market": self.MARKET,
            "currency": self.CURRENCY,
        }

    def _meta_content(self, property_name):
        for selector in (
            f'meta[property="{property_name}"]',
            f'meta[name="{property_name}"]',
        ):
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                value = (element.get_attribute("content") or "").strip()
                if value:
                    return value
            except Exception:
                continue
        return ""

    def _fill_product_header(self, product):
        """Fill direct-link identity fields from PH product-page metadata."""
        current_name = str(product.get("name") or "").strip()
        generic_name = "shop online with promos and vouchers" in current_name.casefold()
        if not current_name or generic_name:
            candidate = self._meta_content("og:title") or self._first_text(
                self.driver, ("h1",)
            )
            if "shop online with promos and vouchers" not in candidate.casefold():
                product["name"] = candidate
        product["name"] = re.sub(
            r"\s*\|\s*Shopee Philippines\s*$", "", product.get("name", ""), flags=re.I
        ).strip()
        if not product.get("img"):
            product["img"] = self._meta_content("og:image")
        if not product.get("price"):
            amount = (
                self._meta_content("product:price:amount")
                or self._meta_content("og:price:amount")
            )
            if amount:
                product["price"] = f"₱{amount}"
            else:
                price_text = self._first_text(
                    self.driver,
                    (
                        '[data-testid="pdp-product-price"]',
                        '[class*="product-price"]',
                        'section[aria-live="polite"]',
                    ),
                )
                product["price"] = self._price_from_text(price_text)
        if not product.get("rating"):
            rating_text = (
                self._meta_content("product:rating:value")
                or self._first_text(
                    self.driver,
                    (
                        '.product-rating-overview__rating-score',
                        '[class*="product-rating-overview__briefing"]',
                        '[class*="rating-score"]',
                    ),
                )
            )
            rating_match = re.search(r"(?<!\d)[0-5](?:\.\d+)?(?!\d)", rating_text)
            product["rating"] = rating_match.group(0) if rating_match else ""

    def _load_rating_filters(self):
        """Scroll progressively until Shopee PH mounts its lazy ratings panel."""
        self.driver.implicitly_wait(0)
        filters = []
        try:
            for _ in range(48):
                filters = self.driver.find_elements(
                    By.CSS_SELECTOR, ".product-rating-overview__filter"
                )
                if filters:
                    break
                self.driver.execute_script(
                    "window.scrollBy(0, Math.max(700, Math.floor(window.innerHeight * 0.8)));"
                )
                time.sleep(0.5)
            if not filters:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)
                filters = self.driver.find_elements(
                    By.CSS_SELECTOR, ".product-rating-overview__filter"
                )
        finally:
            self.driver.implicitly_wait(3)
        if not filters:
            self._save_debug_artifacts("ratings_missing")
        return filters

    def _find_rating_filter(self, target_key):
        for rating_filter in self.driver.find_elements(
            By.CSS_SELECTOR, ".product-rating-overview__filter"
        ):
            key, count = self._parse_rating_filter(self._element_text(rating_filter))
            if key == target_key:
                return rating_filter, count
        return None, 0

    def _collect_lowest_rating_reviews(self, review_limit, per_star_limit=None):
        """Collect 1-star reviews first, moving upward only as needed."""
        reviews = []
        for star in range(1, 6):
            remaining = review_limit - len(reviews)
            if per_star_limit is None and remaining <= 0:
                break
            rating_filter, count = self._find_rating_filter(f"{star}_star")
            if rating_filter is None or count <= 0:
                continue
            target = min(count, per_star_limit if per_star_limit is not None else remaining)
            if target <= 0:
                continue
            try:
                rating_filter.click()
                time.sleep(1)
            except Exception as exc:
                try:
                    self.driver.execute_script("arguments[0].click();", rating_filter)
                    time.sleep(1)
                except Exception:
                    logging.warning(
                        "Could not select the %s-star Shopee PH filter: %s", star, exc
                    )
                    continue
            reviews.extend(self._collect_reviews(target))
        return reviews

    def _scrape_details(self, product):
        """Collect product metadata and English-language ratings from Shopee PH."""
        try:
            self._safe_get(product["link"])
            self.driver.implicitly_wait(3)
            self._fill_product_header(product)

            product["category"] = self._first_text(
                self.driver,
                ('[data-testid="pdp-product-category"]', '[class*="product-category"]'),
            )
            product["description"] = self._first_text(
                self.driver,
                ('[data-testid="pdp-product-description"]', '[class*="product-description"]'),
            )
            if not product["description"]:
                product["description"] = self._meta_content("og:description")

            rating_filters = self._load_rating_filters()
            self._fill_product_header(product)
            previous_breakdown = dict(product.get("detailed_rating") or {})
            current_breakdown = {}
            for rating_filter in rating_filters:
                key, count = self._parse_rating_filter(
                    self._element_text(rating_filter)
                )
                if key:
                    current_breakdown[key] = count
            if rating_filters and not current_breakdown:
                logging.warning(
                    "Unrecognized Shopee PH rating filters: %r",
                    [self._element_text(rating_filter) for rating_filter in rating_filters],
                )
                self._save_debug_artifacts("rating_filters_unparsed")
            product["detailed_rating"] = current_breakdown or previous_breakdown
            product["total_rating"] = self._total_rating_from_breakdown(
                product["detailed_rating"]
            )
            product["review_order"] = "lowest_rating_first"

            if self.index_only:
                return
            if self.all_star_types:
                product["comments"] = self._collect_lowest_rating_reviews(
                    sum(
                        min(count, self.star_limit_per_type)
                        for key, count in product["detailed_rating"].items()
                        if key.endswith("_star")
                    ),
                    per_star_limit=self.star_limit_per_type,
                )
            else:
                product["comments"] = self._collect_lowest_rating_reviews(
                    self.review_limit
                )
                if not product["comments"] and not rating_filters and self.review_limit:
                    product["comments"] = self._collect_reviews(self.review_limit)
        except Exception as e:
            logging.warning(f"Shopee PH detail scrape failed for {product.get('link')}: {e}")
            product.setdefault("comments", [])

    def _reviews_from_current_page(self):
        """Read Shopee's hashed review markup using stable structure and attributes."""
        records = self.driver.execute_script(
            r"""
            const nodeText = (node) => (
                (node && (node.innerText || node.textContent)) || ''
            ).trim();
            return Array.from(
                document.querySelectorAll('.product-ratings__list [data-cmtid]')
            ).map((item) => {
                const main = item.children.length > 1 ? item.children[1] : item;
                const header = main.children.length ? main.children[0] : main;
                const author = header.children.length ? nodeText(header.children[0]) : '';
                const dated = Array.from(main.querySelectorAll('*'))
                    .map((node) => nodeText(node))
                    .filter((text) => /\d{4}-\d{2}-\d{2}/.test(text))
                    .sort((left, right) => left.length - right.length);
                const contentBlock = main.children.length > 1 ? main.children[1] : null;
                const content = contentBlock && contentBlock.children.length
                    ? nodeText(contentBlock.children[contentBlock.children.length - 1])
                    : nodeText(contentBlock);
                const response = Array.from(main.querySelectorAll('*'))
                    .map((node) => nodeText(node))
                    .filter((text) => /seller.{0,3}(response|reply)/i.test(text))
                    .sort((left, right) => left.length - right.length);
                return {
                    comment_id: item.getAttribute('data-cmtid') || '',
                    author,
                    rating: main.querySelectorAll('svg.icon-rating-solid').length,
                    time: dated.length ? dated[0] : '',
                    content,
                    seller_respond: response.length ? response[0] : '',
                    like_count_text: nodeText(
                        main.querySelector('.shopee-product-rating__like-count')
                    )
                };
            });
            """
        )
        return [
            {
                "comment_id": str(record.get("comment_id") or ""),
                "author": str(record.get("author") or ""),
                "rating": int(record.get("rating") or 0),
                "time": str(record.get("time") or ""),
                "content": str(record.get("content") or ""),
                "seller_respond": str(record.get("seller_respond") or ""),
                "like_count": self._parse_star_text(record.get("like_count_text")),
            }
            for record in (records or [])
        ]

    def _collect_reviews(self, max_reviews):
        """Helper to collect up to max_reviews from the current filtered view."""
        collected_reviews = []
        seen_reviews = set()
        try:
            # Wait up to 2 seconds for rating list to appear
            start_time = time.time()
            rating_container = None
            while time.time() - start_time < 5:
                try:
                    rating_container = self.driver.find_element(By.CLASS_NAME, 'product-ratings__list')
                    break
                except:
                    time.sleep(0.1)
            if not rating_container:
                return collected_reviews
        except:
            return collected_reviews

        with _progress(total=max_reviews, desc="Collecting reviews") as pbar:
            while len(collected_reviews) < max_reviews:
                previous_count = len(collected_reviews)
                rating_items = []
                item_deadline = time.time() + 5
                while time.time() < item_deadline and not rating_items:
                    rating_items = rating_container.find_elements(
                        By.CSS_SELECTOR, ".shopee-product-comment-list > [data-cmtid]"
                    )
                    if not rating_items:
                        rating_items = rating_container.find_elements(
                            By.CSS_SELECTOR, "[data-cmtid]"
                        )
                    if not rating_items:
                        rating_items = rating_container.find_elements(
                            By.CSS_SELECTOR, ".shopee-product-rating__main"
                        )
                    if not rating_items:
                        time.sleep(0.2)
                page_reviews = self._reviews_from_current_page() if rating_items else []
                for review_data in page_reviews:
                    if len(collected_reviews) >= max_reviews:
                        break
                    fingerprint = review_data["comment_id"] or (
                        review_data["author"], review_data["rating"],
                        review_data["time"], review_data["content"],
                    )
                    if fingerprint in seen_reviews:
                        continue
                    seen_reviews.add(fingerprint)
                    collected_reviews.append(review_data)
                    pbar.update(1)
                if page_reviews:
                    rating_items = []
                for item in rating_items:
                    if len(collected_reviews) >= max_reviews:
                        break
                    review_data = {
                        "comment_id": (item.get_attribute("data-cmtid") or "").strip()
                    }
                    # Author
                    review_data["author"] = self._first_text(
                        item,
                        ('.shopee-product-rating__author-name', 'a[href*="/shop/buyer/"]'),
                    )

                    # Rating
                    try:
                        solid_stars = item.find_elements(
                            By.CSS_SELECTOR,
                            '[class*="rating-solid--active"], svg.icon-rating-solid',
                        )
                        review_data["rating"] = len(solid_stars)
                    except Exception:
                        review_data["rating"] = 0

                    # Time
                    review_data["time"] = self._first_text(
                        item, ('.shopee-product-rating__time', '[class*="rating__time"]')
                    )

                    # Content
                    review_data["content"] = self._first_text(
                        item,
                        ('.shopee-product-rating__content', '[style*="white-space: pre-wrap"]'),
                    )

                    # Seller respond
                    review_data["seller_respond"] = self._first_text(
                        item,
                        ('.shopee-product-rating__response', '[class*="seller-response"]', '.TQTPT9 .qiTixQ'),
                    )

                    # Like count
                    try:
                        like_elem = item.find_element(By.XPATH, './/div[@class="shopee-product-rating__like-count"]')
                        like_text = like_elem.text.strip()
                        review_data["like_count"] = self._parse_star_text(like_text)
                    except Exception:
                        review_data["like_count"] = 0

                    fingerprint = review_data["comment_id"] or (
                        review_data["author"], review_data["rating"],
                        review_data["time"], review_data["content"],
                    )
                    if fingerprint in seen_reviews:
                        continue
                    seen_reviews.add(fingerprint)

                    collected_reviews.append(review_data)
                    pbar.update(1)
                if len(collected_reviews) >= max_reviews or len(collected_reviews) == previous_count:
                    break
                # Try to click next page if available
                elements_to_try = [
                    (By.CSS_SELECTOR, 'button.shopee-icon-button--right:not([disabled])'),
                    (By.CSS_SELECTOR, 'button[aria-label*="next" i]:not([disabled])'),
                    (By.XPATH, '//nav//button[not(@disabled)][last()]'),
                ]

                next_clicked = False
                for by, value in elements_to_try:
                    try:
                        next_button = self.driver.find_element(by, value)
                        next_button.click()
                        next_clicked = True
                        time.sleep(1)
                        rating_container = self.driver.find_element(By.CLASS_NAME, 'product-ratings__list')
                        break
                    except:
                        continue
                if not next_clicked:
                    break
        return collected_reviews

    def execute(self):
        sys.excepthook = self._handle_exception
        chrome_kwargs = {
            "options": self.options,
            "enable_cdp_events": False,
            "headless": False,
        }
        chrome_major = self._detect_chrome_major()
        if chrome_major:
            chrome_kwargs["version_main"] = chrome_major
            logging.info("Starting Chrome with matching major version %s.", chrome_major)
        self.driver = uc.Chrome(**chrome_kwargs)
        _position_scraper_window(self.driver)

        # Run the missing comments scraper if we have existing data
        if self.output_data:
            self._scrape_missing_comments()

        products = []
        try:
            if self.product_urls:
                logging.info("Loading Shopee Philippines for direct product collection...")
                products = [self._product_from_url(url) for url in self.product_urls]
            else:
                products = self._scrape_page()
        except Exception as e:
            logging.error(f"Error scraping: {e}")
        final_list = []
        try:
            if not products:
                return
            for prod in _progress(products, desc="Processing products"):
                link = prod["link"]
                if link in self.output_data:
                    final_list.append(self.output_data[link])
                    continue
                if not self.index_only:
                    self._scrape_details(prod)
                final_list.append(prod)
                self.output_data[link] = prod
                # Periodic save after each product
                self._periodic_save()
        finally:
            logging.info("Quitting driver...")
            self.driver.quit()

        # Final save (redundant but safe)
        self._periodic_save()
        logging.info(f"Data saved to {self.out_file}")

def build_parser():
    parser = argparse.ArgumentParser(
        description="Scrape public products and reviews from Shopee Philippines"
    )
    parser.add_argument("-k", "--keyword", default="", help="Shopee PH search term")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of products")
    parser.add_argument("-r", "--review-limit", type=int, default=30, help="Max reviews per product")
    parser.add_argument("--index-only", action="store_true", default=False, help="If set, only retrieve index data")
    parser.add_argument("--all-star-types", action="store_true", default=False, help="Retrieve comments by filtering each star rating.")
    parser.add_argument("--star-limit-per-type", type=int, default=10, help="Number of reviews to retrieve per star type.")
    parser.add_argument(
        "--product-url",
        action="append",
        default=[],
        help="Direct shopee.ph product URL; may be supplied more than once",
    )
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Resume automatically after browser login/captcha verification",
    )
    parser.add_argument(
        "--verification-timeout",
        type=int,
        default=900,
        help="Seconds to wait for browser login/captcha verification",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if not args.keyword.strip() and not args.product_url:
        build_parser().error("provide --keyword or --product-url")
    scraper = ShopeeScraper(
        args.keyword,
        args.num,
        args.index_only,
        args.review_limit,
        all_star_types=args.all_star_types,
        star_limit_per_type=args.star_limit_per_type,
        product_urls=args.product_url,
        output=args.output,
        no_prompt=args.no_prompt,
        verification_timeout=args.verification_timeout,
    )
    scraper.execute()
