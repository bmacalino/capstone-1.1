import json
import unittest

from dashboard_utils import (
    analysis_is_current,
    badge_view,
    cached_product_matches,
    csv_bytes,
    detect_marketplace,
    load_products,
    has_usable_products,
    merge_product_catalogs,
    normalized_risk_score,
    price_value,
    platform_name,
    product_rows,
    product_reliability,
    rank_recommendations,
    reliability_score,
    review_rows,
    search_slug,
    sentiment_model_version,
)


def quote(text, span=None, subtype=None, confident=True, score=5.0, negated=False, rating=None):
    """One evidence sentence as the badge aggregator records it."""
    terms = [{"span": list(span), "negated": negated}] if span else []
    return {
        "text": text,
        "score": score,
        "high_confidence": confident,
        "hazard_subtype": subtype,
        "terms": terms,
        "rating": rating,
    }


def badge_summary(
    state="no_warning_signs",
    *,
    usable=40,
    safety=None,
    counterfeit=0,
    defect=0,
    detail="",
    positive_lb=0.8,
):
    """A minimal schema-2.0 summary carrying only the fields the view reads."""
    return {
        "schema_version": "defaketive-summary-2.0",
        "verdict": {"state": state, "detail": detail},
        "usable_review_count": usable,
        "positive_ratio_lower_bound": positive_lb,
        "safety": safety or {"flagged_reviews": 0, "hazard_subtypes": {}, "evidence": []},
        "flags": {
            "counterfeit": {"flagged_reviews": counterfeit, "evidence": []},
            "defect": {"flagged_reviews": defect, "evidence": []},
        },
    }


def fire_reports(count):
    return {
        "flagged_reviews": count,
        "hazard_subtypes": {"fire_explosion_swelling": {"reviews": count, "severity": 4}},
        "evidence": [
            quote(f"nasunog yung charger {index}", span=(0, 7), subtype="fire_explosion_swelling")
            for index in range(count)
        ],
    }


class TestBadgeView(unittest.TestCase):
    def test_safety_warning_says_what_reviewers_report(self):
        view = badge_view(
            {"sentiment_summary": badge_summary("safety_warning", safety=fire_reports(3))}
        )
        self.assertEqual(view["tone"], "high")
        self.assertEqual(view["headline"], "Reviewers report safety problems")
        self.assertEqual(
            view["detail"],
            "Of 40 reviews, 3 mention fire, explosion or a swollen battery. "
            "Read them below and judge for yourself.",
        )
        self.assertEqual(view["counts"], {"safety": 3, "counterfeit": 0, "defect": 0})

    def test_counts_agree_with_their_verbs(self):
        view = badge_view(
            {"sentiment_summary": badge_summary(safety=fire_reports(1), counterfeit=1, defect=2)}
        )
        summaries = {badge["category"]: badge["summary"] for badge in view["badges"]}
        self.assertEqual(summaries["safety"], "1 review mentions fire, explosion or a swollen battery")
        self.assertEqual(summaries["counterfeit"], "1 review says it may not be genuine")
        self.assertEqual(summaries["defect"], "2 reviews say it is faulty or stopped working")

    def test_labels_are_worded_for_shoppers(self):
        view = badge_view({"sentiment_summary": badge_summary()})
        self.assertEqual(
            [(item["category"], item["label"]) for item in view["labels"]],
            [
                ("safety", "Safety problem"),
                ("defect", "Faulty or broken"),
                ("quality", "Poor quality"),
                ("counterfeit", "Possible fake"),
                ("not_as_described", "Not as advertised"),
            ],
        )
        self.assertTrue(all(item["helper"] for item in view["labels"]))
        # A checker that never looked for the two newer labels gets no badge for
        # them: "not checked" must not read as "0 reviews".
        self.assertEqual([badge["category"] for badge in view["badges"]], ["safety", "defect", "counterfeit"])
        self.assertEqual(badge_view({})["labels"], view["labels"])

    def test_the_two_informing_labels_are_shown_when_the_checker_ran_them(self):
        summary = badge_summary(defect=1)
        summary["reports"] = {
            "quality": {
                "flagged_reviews": 3,
                "evidence": [{"text": "mahina yung tunog", "terms": [{"span": [0, 6], "category": "quality"}]}],
            },
            "not_as_described": {"flagged_reviews": 0, "evidence": []},
        }
        view = badge_view({"sentiment_summary": summary})
        self.assertEqual(
            [badge["category"] for badge in view["badges"]],
            ["safety", "defect", "quality", "counterfeit", "not_as_described"],
        )
        quality, unlike = view["badges"][2], view["badges"][4]
        self.assertEqual((quality["label"], quality["informs_only"]), ("Poor quality", True))
        self.assertEqual(quality["summary"], "3 reviews say it works but is weak, flimsy or marked")
        self.assertEqual(quality["evidence"][0]["highlights"], [[0, 6]])
        self.assertEqual(unlike["summary"], "No reviews say it differs from the listing")
        self.assertEqual(view["counts"]["quality"], 3)
        self.assertFalse(view["badges"][1]["informs_only"])

    def test_poor_quality_alone_is_reported_without_a_warning(self):
        summary = badge_summary(
            detail="No warning signs found in 40 reviews. Absence of complaints "
            "is not proof that a product is safe."
        )
        summary["reports"] = {
            "quality": {"flagged_reviews": 2, "evidence": []},
            "not_as_described": {"flagged_reviews": 0, "evidence": []},
        }
        view = badge_view({"sentiment_summary": summary})
        self.assertEqual((view["headline"], view["tone"]), ("No warning signs found", "low"))
        self.assertEqual(
            view["detail"],
            "No problem reaches a warning level in 40 reviews, but 2 reviews say it works but "
            "is weak, flimsy or marked. Read them below. This is not proof that a product is safe.",
        )

    def test_too_few_reviews_still_surfaces_a_report(self):
        # "Not enough reviews to judge" must not hide a fire report among them.
        view = badge_view(
            {
                "sentiment_summary": badge_summary(
                    "insufficient_evidence",
                    usable=4,
                    safety=fire_reports(1),
                    detail="Only 4 usable reviews; at least 30 are needed",
                )
            }
        )
        self.assertEqual(view["headline"], "Not enough reviews to judge")
        self.assertEqual(
            view["detail"],
            "Only 4 usable reviews; at least 30 are needed. Even so, 1 review mentions "
            "fire, explosion or a swollen battery. Read it below.",
        )

    def test_a_clean_result_is_never_called_safe(self):
        view = badge_view(
            {
                "sentiment_summary": badge_summary(
                    detail="No warning signs found in 40 reviews. Absence of complaints "
                    "is not proof that a product is safe."
                )
            }
        )
        self.assertEqual(view["headline"], "No warning signs found")
        self.assertIn("not proof", view["detail"])
        for text in [view["headline"]] + [badge["summary"] for badge in view["badges"]]:
            self.assertNotIn("safe", text.casefold().split())

    def test_an_isolated_report_is_not_described_as_no_complaints(self):
        view = badge_view(
            {
                "sentiment_summary": badge_summary(
                    counterfeit=1,
                    detail="No warning signs found in 40 reviews. Absence of complaints "
                    "is not proof that a product is safe.",
                )
            }
        )
        self.assertEqual(view["headline"], "No warning signs found")
        self.assertNotIn("Absence of complaints", view["detail"])
        self.assertEqual(
            view["detail"],
            "No problem reaches a warning level in 40 reviews, but 1 review says it may not "
            "be genuine. Read it below. This is not proof that a product is safe.",
        )

    def test_the_most_serious_hazard_is_named_and_others_noted(self):
        safety = {
            "flagged_reviews": 3,
            "hazard_subtypes": {
                "smoke_smell": {"reviews": 2, "severity": 3},
                "shock_spark_short": {"reviews": 1, "severity": 4},
            },
            "evidence": [],
        }
        badge = badge_view({"sentiment_summary": badge_summary(safety=safety)})["badges"][0]
        self.assertEqual(
            badge["summary"],
            "3 reviews mention electric shocks, sparks or short circuits, and other hazards",
        )

    def test_evidence_is_distinct_capped_and_ranked_by_confidence(self):
        evidence = (
            [quote("weak one", confident=False, score=9.0)]
            + [quote(f"strong {index}", score=float(index)) for index in range(6)]
            + [quote("Strong 5")]
        )
        safety = {"flagged_reviews": 8, "hazard_subtypes": {}, "evidence": evidence}
        badge = badge_view({"sentiment_summary": badge_summary(safety=safety)})["badges"][0]
        texts = [item["text"] for item in badge["evidence"]]
        self.assertEqual(len(texts), 5)
        self.assertEqual(texts[0], "strong 5")
        self.assertEqual(texts.count("Strong 5") + texts.count("strong 5"), 1)
        self.assertNotIn("weak one", texts)  # confidence outranks a higher score
        # Seven distinct quotes ("Strong 5" repeats "strong 5"), five shown.
        self.assertEqual(badge["more_evidence"], 2)

    def test_only_active_terms_are_highlighted(self):
        safety = {
            "flagged_reviews": 1,
            "hazard_subtypes": {},
            "evidence": [quote("hindi nasunog", span=(6, 13), negated=True)],
        }
        badge = badge_view({"sentiment_summary": badge_summary(safety=safety)})["badges"][0]
        self.assertEqual(badge["evidence"][0]["highlights"], [])

    def test_a_legacy_summary_is_reported_as_not_analyzed(self):
        view = badge_view({"sentiment_summary": {"risk_score": 12, "risk_score_scale": "percent"}})
        self.assertFalse(view["available"])
        self.assertEqual(view["headline"], "Not analyzed yet")

    def test_export_rows_carry_badges_not_scores(self):
        row = product_rows(
            [
                {
                    "name": "X",
                    "comments": [],
                    "sentiment_summary": badge_summary(
                        "safety_warning", safety=fire_reports(2), defect=1
                    ),
                }
            ]
        )[0]
        self.assertEqual(row["Verdict"], "Reviewers report safety problems")
        self.assertEqual((row["Safety reports"], row["Faulty or broken reports"]), (2, 1))
        # Not checked for this product, so blank rather than 0.
        self.assertIsNone(row["Poor quality reports"])
        self.assertNotIn("Defect reports", row)
        self.assertNotIn("Counterfeit reports", row)
        self.assertNotIn("Risk score", row)
        self.assertNotIn("Reliability score", row)

    def test_a_current_badge_analysis_is_not_reanalysed(self):
        product = {
            "comments": [{"sentiment_analysis": {"model_version": sentiment_model_version()}}],
            "sentiment_summary": badge_summary(),
        }
        self.assertTrue(analysis_is_current([product]))


class TestDashboardUtils(unittest.TestCase):
    def test_analysis_version_check_rejects_stale_saved_results(self):
        product = {
            "comments": [
                {
                    "content": "Sira agad",
                    "sentiment_analysis": {"model_version": "old-model"},
                }
            ],
            "sentiment_summary": {"risk_score_scale": "percent"},
        }
        self.assertFalse(analysis_is_current([product]))
        product["comments"][0]["sentiment_analysis"]["model_version"] = (
            sentiment_model_version()
        )
        self.assertTrue(analysis_is_current([product]))

    def test_cached_product_matching_uses_names_and_normalized_links(self):
        products = [
            {"name": "Raspberry Pi Pico 2 Board", "link": "https://shopee.ph/sample-i.1.2?tracking=1", "sentiment_summary": {"risk_score": 20}},
            {"name": "Wireless Earbuds", "link": "https://www.lazada.com.ph/products/earbuds-i1-s2.html", "sentiment_summary": {"risk_score": 10}},
        ]
        self.assertEqual(cached_product_matches(products, "raspberry pico")[0]["name"], "Raspberry Pi Pico 2 Board")
        self.assertEqual(
            cached_product_matches(products, "https://shopee.ph/sample-i.1.2?from=share")[0]["name"],
            "Raspberry Pi Pico 2 Board",
        )
        self.assertEqual(cached_product_matches(products, "not in database"), [])

    def test_recommendations_are_similar_and_show_no_warning_signs(self):
        def listing(link, name, summary, category="Power Banks", rating=4.8, brand=""):
            return {
                "link": link,
                "name": name,
                "brand": brand,
                "category": category,
                "rating": rating,
                "sentiment_summary": summary,
            }

        target = listing(
            "target",
            "Anker 25000mAh Laptop Power Bank",
            badge_summary("safety_warning", safety=fire_reports(3)),
            brand="Anker",
        )
        good = listing("good", "UGREEN 20000mAh USB-C Power Bank", badge_summary())
        risky = listing(
            "risky",
            "Anker Power Bank",
            badge_summary("safety_warning", safety=fire_reports(2)),
            brand="Anker",
        )
        # "No warning signs" overall, yet one review reports a fire: never
        # offer that as the safer choice.
        one_fire = listing(
            "one-fire", "Baseus 20000mAh Power Bank", badge_summary(safety=fire_reports(1))
        )
        thin_praise = listing("thin", "Romoss 20000mAh Power Bank", badge_summary(positive_lb=0.3))
        # A legacy score cannot vouch for a product under the current checks.
        legacy = listing(
            "legacy",
            "Xiaomi 20000mAh Power Bank",
            {"risk_score": 1, "risk_score_scale": "percent", "review_count": 50},
        )
        unrelated = listing(
            "unrelated",
            "Raspberry Pi Development Board",
            badge_summary(),
            category="Microcontrollers",
            rating=5,
        )

        recommendations = rank_recommendations(
            target, [target, risky, one_fire, thin_praise, legacy, unrelated, good], limit=3
        )

        self.assertEqual([item["link"] for item in recommendations], ["good"])

    def test_a_product_is_read_as_the_kind_its_name_leads_with(self):
        # Until 13 Oct 2026 the first kind in a fixed list won. On the saved
        # products that read a laptop stand with a "monitor lifter" as a
        # monitor and a helmet intercom headset as headphones.
        from dashboard_utils import _recommendation_type

        kinds = {
            # Of two ordinary kinds, the one named first.
            "Wired Dynamic Microphone hand box Speaker": "microphone",
            "USB Type C Cable for Fast Charger": "cable",
            # What a product comes with is not what it is.
            "Bluetooth Speaker with Microphone": "speaker",
            "Headset with Microphone with mic": "headphones",
            "K12 Mini Home With 1 Microphones Professional Karaoke Speaker": "speaker",
            # Named after another product, and never that product.
            "Anker 25000mAh Laptop Power Bank": "powerbank",
            "MOFOX S7 Pro Motorcycle Helmet Intercom Headset 2in1 Mic earphone": "intercom",
            "Jiditech Laptop Stand 360° With Fan, Non-Slip Monitor Lifter for 9.7-17 Inch Laptops": "stand",
            "10-level adjustable portable laptop universal stand suitable for Pad": "stand",
            "Doorbell Wireless With Camera Waterproof Video Intercom": "doorbell",
            "3D Movies VR Box Immersive Virtual Reality glasses Headset": "vr",
            "Multi Port USB Hub Splitter for Laptop": "hub",
            "M10 TWS Bluetooth Headphone Waterproof With Charging Case": "earbuds",
            "Vention Cat 6 Ethernet Cable Lan Cable 20 Meter": "network cable",
            "Essager 2.5W Apple Watch Wireless Chargers Fast Charging": "wireless charger",
            "KASIDA Bluetooth WIFI Tuya smart lock 6 in 1 fingerprint door knob": "lock",
            "Smart 5G Max Home WiFi": "router",
            "Headphone Stand Aluminum Desk Holder": "stand",
            "4in1 For samsung Galaxy A07 phone case + Tempered Glass + Camera lens film": "case",
            # ...unless an ordinary kind is named before it: these are sold as
            # a speaker and as earphones.
            "Lazy People 3 in 1 BlueTooth Speaker Power bank With Phone Stand": "speaker",
            "M25 Wireless Bluetooth Earphone with Emergency Power Bank and Digital LED Display": "earbuds",
            # Memory and storage are listed only for a phone or a tablet.
            "SAMSUNG A26 5G 8+256 - Black (ANKER Zolo 30w USB-C Wall Charger + BELKIN earphones)": "phone",
            # Words that mostly describe something else.
            "COROS PACE 4 GPS Sport Watch": "smartwatch",
            "Original GT Bluetooth Call Smart Digital Watch for Men Waterproof Color Touch Screen": "smartwatch",
            "Lenovo M20 Laptop Office Wired Mouse Desktop Computer": "mouse",
            "ASUS Vivobook 15 Laptop Intel i5 with 8GB RAM": "laptop",
            "USB C Adapter for Laptop and Monitor": "",
            "Power Bank 20000mAh with LCD Display": "powerbank",
            "Ultra Thin 8.5 inch LCD Writing Tablet": "",
            "Original LCD For Oppo Realme Note 50 RMX3834 Touch Screen Replacement": "part",
        }
        for name, kind in kinds.items():
            with self.subTest(name=name):
                self.assertEqual(_recommendation_type({"name": name}), kind)

    def clean_listing(self, link, name):
        return {"link": link, "name": name, "rating": 4.8, "sentiment_summary": badge_summary()}

    def test_a_part_for_another_phone_is_not_an_alternative(self):
        # Seen on the saved products, 13 Oct 2026: the two names share "LCD
        # Display", "Frame", "Touch Screen" and "Replacement", and the Infinix
        # screen was offered to a shopper looking at the Oppo one.
        oppo = self.clean_listing(
            "oppo", "Original LCD For Oppo Realme Note 50/Note 60 RMX3834 LCD Display with Frame Touch Screen Replacement"
        )
        infinix = self.clean_listing(
            "infinix", "Original LCD For Infinix Smart 8 / Smart 8 HD X6525 X6525B LCD Display Touch Screen with Frame Replacement"
        )
        same_phone = self.clean_listing("same", "LCD For Oppo Realme Note 50 RMX3834 Touch Screen Digitizer Assembly")
        recommendations = rank_recommendations(oppo, [oppo, infinix, same_phone])
        self.assertEqual([item["link"] for item in recommendations], ["same"])

    def test_a_case_for_another_phone_is_not_an_alternative(self):
        galaxy = self.clean_listing("galaxy", "4in1 For samsung Galaxy A07 phone case +Tempered Glass + Camera lens film")
        realme = self.clean_listing("realme", "4in1 For Realme C21Y phone case anti-fall mobile phone case + tempered glass")
        same_phone = self.clean_listing("same", "Shockproof Case for Samsung Galaxy A07 Clear Back Cover")
        # "4in1" is on both names and is not a model: only "a07" pairs two.
        recommendations = rank_recommendations(galaxy, [galaxy, realme, same_phone])
        self.assertEqual([item["link"] for item in recommendations], ["same"])

    def test_a_helmet_intercom_is_not_offered_for_earphones(self):
        earphones = self.clean_listing("target", "high quality semi in ear wired headphones microphones")
        intercom = self.clean_listing(
            "intercom", "MOFOX S7 Pro Motorcycle Helmet Intercom Headset 2in1 Mic earphone Music Share"
        )
        headphones = self.clean_listing("headphones", "Wired Over Ear Headphones with Mic")
        recommendations = rank_recommendations(earphones, [earphones, intercom, headphones])
        self.assertEqual([item["link"] for item in recommendations], ["headphones"])

    def test_a_stand_is_offered_stands_not_the_things_it_holds(self):
        stand = self.clean_listing(
            "target",
            "Jiditech Laptop Stand 360° With Fan and Tablet Holder, Non-Slip Monitor Lifter, "
            "Compatible with 9.7-17 Inch Laptops",
        )
        other_stand = self.clean_listing("stand", "luckluck Adjustable Laptop Stand with Phone Holder 360° Rotating")
        monitor = self.clean_listing("monitor", "Dell 24 Inch Monitor Full HD")
        laptop = self.clean_listing("laptop", "ASUS Vivobook 15 Laptop Intel i5")
        recommendations = rank_recommendations(stand, [stand, monitor, laptop, other_stand])
        self.assertEqual([item["link"] for item in recommendations], ["stand"])
        # And a shopper looking at a laptop is not offered a stand for one.
        self.assertEqual(rank_recommendations(laptop, [laptop, stand, other_stand]), [])

    def test_the_same_word_does_not_make_two_products_alike(self):
        # Seen when the scanned listings were tried in the pool, 13 Oct 2026.
        charging = self.clean_listing("charging", "Type C Fast Charging Cable 120W Super Charge Braided Cable")
        ethernet = self.clean_listing("ethernet", "Vention Cat 6 Ethernet Cable Lan Cable 20 Meter RJ45")
        wall = self.clean_listing("wall", "Anker Zolo GaN 30W USB C Charger Foldable")
        watch = self.clean_listing("watch", "Essager 2.5W Apple Watch Wireless Chargers Fast Charging Portable")
        lock = self.clean_listing("lock", "KASIDA Bluetooth WIFI Tuya smart lock 6 in 1 fingerprint main door lock for home")
        wifi = self.clean_listing("wifi", "Smart 5G Max Home WiFi")
        everything = [charging, ethernet, wall, watch, lock, wifi]
        for product in everything:
            with self.subTest(product=product["link"]):
                self.assertEqual(rank_recommendations(product, everything), [])

    def test_a_product_of_no_known_kind_is_not_offered_one_of_a_known_kind(self):
        # Seen in the filled pool, 13 Oct 2026: both names list the devices
        # they work with, and the stylus was offered the mouse.
        stylus = self.clean_listing("stylus", "DANYCASE Stylus Pen 3 in 1 Universal Pencil for Mobile Android ios Phone iPad Tablet")
        mouse = self.clean_listing("mouse", "GOOJODOQ Bluetooth Wireless Mouse Slim Silent Rechargeable iPad Tablet Phone Android ios")
        self.assertEqual(rank_recommendations(stylus, [stylus, mouse]), [])
        self.assertEqual(rank_recommendations(mouse, [stylus, mouse]), [])
        # Nor do the devices two unnamed kinds work with make them alike.
        holder = self.clean_listing("holder", "Desk Holder Bracket for Mobile Android ios Phone iPad Tablet")
        self.assertEqual(rank_recommendations(stylus, [stylus, holder]), [])

    def test_words_every_gadget_uses_do_not_make_two_unnamed_kinds_alike(self):
        # Neither name holds a kind the list knows, so only shared words can
        # pair them, and "smart", "wireless", "portable", "mini" must not.
        fan = self.clean_listing("fan", "Smart Wireless Portable Mini Fan Rechargeable")
        scale = self.clean_listing("scale", "Smart Wireless Portable Mini Scale Rechargeable")
        self.assertEqual(rank_recommendations(fan, [fan, scale]), [])
        # Words that do name the thing still pair two listings of it.
        fan_two = self.clean_listing("fan2", "Handheld Desk Fan 3 Speed Quiet Turbo Cooling")
        fan_three = self.clean_listing("fan3", "Quiet Turbo Desk Fan Handheld Cooling 5 Speed")
        self.assertEqual([item["link"] for item in rank_recommendations(fan_two, [fan_two, fan_three, scale])], ["fan3"])

    def test_merge_product_catalogs_normalizes_tracking_links(self):
        current = {"name": "Current", "link": "https://www.lazada.com.ph/products/item-i1.html?tracking=1"}
        stale = {"name": "Stale", "link": "https://lazada.com.ph/products/item-i1.html?tracking=2"}
        other = {"name": "Other", "link": "https://shopee.ph/other-i.1.2"}
        self.assertEqual(merge_product_catalogs([current], [stale, other]), [current, other])

    def test_usable_product_detection_rejects_blank_failed_scrapes(self):
        self.assertFalse(has_usable_products([{"link": "https://shopee.ph/item-i.1.2", "comments": []}]))
        self.assertTrue(has_usable_products([{"name": "Real product", "comments": []}]))
        self.assertTrue(has_usable_products([{"review_status": "no_reviews", "comments": []}]))

    def test_search_slug(self):
        self.assertEqual(search_slug("Laptop Stand & Bag"), "laptop_stand_bag")
        self.assertEqual(search_slug("!!!"), "search")

    def test_marketplace_detection_and_display(self):
        self.assertEqual(detect_marketplace("https://shopee.ph/item-i.1.2"), "shopee")
        self.assertEqual(detect_marketplace("https://www.lazada.com.ph/products/item.html"), "lazada")
        self.assertEqual(detect_marketplace("https://www.temu.com/ph-en/goods.html"), "temu")
        self.assertEqual(detect_marketplace("wireless earbuds", "lazada"), "lazada")
        self.assertEqual(platform_name({"platform": "temu"}), "Temu PH")
        with self.assertRaises(ValueError):
            detect_marketplace("https://example.com/item")

    def test_price_value_uses_range_midpoint(self):
        self.assertEqual(price_value("₱1,000 - ₱1,500"), 1250.0)
        self.assertEqual(price_value("â‚±299"), 299.0)
        self.assertIsNone(price_value(""))

    def test_load_and_flatten_reviews(self):
        payload = [{"name": "Item", "comments": [{"rating": 5, "content": "Good"}]}]
        products = load_products(json.dumps(payload).encode())
        rows = review_rows(products)
        self.assertEqual(rows[0]["Product"], "Item")
        self.assertEqual(rows[0]["Sentiment"], "unanalyzed")
        self.assertIn(b"Review", csv_bytes(rows))

    def test_risk_display_helpers(self):
        self.assertEqual(normalized_risk_score(61), 0.61)
        self.assertEqual(normalized_risk_score(0.4), 0.4)
        self.assertEqual(normalized_risk_score(1.0, "percent"), 0.01)
        self.assertEqual(normalized_risk_score(1.0, "ratio"), 1.0)
        self.assertEqual(reliability_score(61), 39.0)
        self.assertEqual(reliability_score(20, 0.75), 55.0)
        self.assertEqual(reliability_score(1.0, 0.6667, "percent"), 65.7)

    def test_product_reliability_respects_explicit_percent_scale(self):
        product = {
            "sentiment_summary": {
                "risk_score": 1.0,
                "risk_score_scale": "percent",
                "sentiment_ratios": {"positive": 0.6667},
            }
        }
        self.assertEqual(product_reliability(product), 65.7)

    def test_product_reliability(self):
        same_category = {"link": "same", "category": "Audio", "sentiment_summary": {"risk_score": 10, "sentiment_ratios": {"positive": .7}, "review_count": 5}}
        self.assertEqual(product_reliability(same_category), 60.0)


if __name__ == "__main__":
    unittest.main()
