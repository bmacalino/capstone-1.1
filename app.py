"""DeFaketive shopper and administrator dashboard.

Run with: streamlit run app.py
"""

from __future__ import annotations

import html
import hashlib
import subprocess
import sys
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard_utils import (
    ROOT,
    analysis_is_current,
    analyze_products,
    cached_product_matches,
    csv_bytes,
    detect_marketplace,
    has_usable_products,
    json_bytes,
    load_products,
    merge_product_catalogs,
    output_path,
    platform_name,
    price_value,
    product_platform,
    product_image_url,
    legacy_product_rows,
    product_reliability,
    product_risk_level,
    product_risk_percent,
    legacy_rank_recommendations,
    recommendation_search_query,
    result_files,
    review_signal_counts,
    review_rows,
    risk_keyword_counts,
    risk_level,
    weighted_risk_breakdown,
)
from data_store import (
    database_counts,
    load_saved_products,
    save_evaluation_run,
    save_products,
    save_survey_response,
)
from research_utils import evaluate_labels, score_sus, score_umux


INDIGO = "#3E4784"
GREEN = "#12B76A"
AMBER = "#F79009"
RED = "#F04438"
INK = "#101828"
RISK_COLORS = {"Low": GREEN, "Moderate": AMBER, "High": RED}
MANUAL_VERIFICATION_TIMEOUT = 900
SCRAPER_PROCESS_TIMEOUT = MANUAL_VERIFICATION_TIMEOUT + 300
NAVIGATION_PAGES = (
    "Search",
    "Search results",
    "Product analysis",
    "Feedback survey",
    "Admin overview",
    "Scraper manager",
    "Lexicon manager",
    "Product database",
    "Model evaluation",
)

st.set_page_config(
    page_title="DeFaketive · Review Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        :root {{
            --indigo:{INDIGO}; --green:{GREEN}; --amber:{AMBER}; --red:{RED};
            --def-bg:var(--background-color,#F8F9FC);
            --def-surface:var(--secondary-background-color,#FFFFFF);
            --def-text:var(--text-color,#101828);
            --def-muted:color-mix(in srgb,var(--text-color,#101828) 64%,transparent);
            --def-border:color-mix(in srgb,var(--text-color,#101828) 18%,transparent);
            --def-accent:color-mix(in srgb,var(--primary-color,#3E4784) 78%,var(--text-color,#101828) 22%);
            --def-shadow:color-mix(in srgb,#000000 14%,transparent);
        }}
        html, body, [class*="css"] {{ font-family:Inter,sans-serif;color:var(--def-text); }}
        .stApp, [data-testid="stAppViewContainer"] > .main {{ background:var(--def-bg);color:var(--def-text); }}
        [data-testid="stSidebar"] {{ background:#242B52; }}
        [data-testid="stSidebar"] * {{ color:#F2F4F7; }}
        [data-testid="stSidebar"] hr {{ border-color:#475467; }}
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
            background:#353F74; border-color:#6670A8;
        }}
        .block-container {{ max-width:1440px; padding-top:1.7rem; padding-bottom:4rem; }}
        h1,h2,h3 {{ letter-spacing:-.03em; }}
        .brand {{ font-size:1.25rem; font-weight:800; color:white; padding:.45rem 0 1rem; }}
        .brand-dot {{ color:#6CE9A6; }}
        .eyebrow {{ color:var(--def-accent); font-size:.74rem; letter-spacing:.12em; text-transform:uppercase; font-weight:800; }}
        .hero {{ padding:2.6rem 2.8rem; border-radius:24px; background:linear-gradient(135deg,#30386B,#5B67A6); color:white; box-shadow:0 18px 45px rgba(62,71,132,.18); }}
        .hero h1 {{ color:white; font-size:2.65rem; max-width:760px; margin:.35rem 0 .75rem; }}
        .hero p {{ color:#E4E7EC; max-width:700px; font-size:1.05rem; }}
        .card {{ background:var(--def-surface); color:var(--def-text); border:1px solid var(--def-border); border-radius:16px; padding:1.25rem; box-shadow:0 4px 14px var(--def-shadow); height:100%; }}
        .metric-label {{ color:var(--def-muted); font-size:.78rem; font-weight:600; }}
        .metric-value {{ color:var(--def-text); font-size:1.75rem; font-weight:800; line-height:1.2; margin-top:.35rem; }}
        .muted {{ color:var(--def-muted); }}
        .risk-pill {{ display:inline-flex; align-items:center; gap:.35rem; padding:.3rem .65rem; border-radius:999px; color:white; font-size:.74rem; font-weight:700; }}
        .platform-pill {{ display:inline-block; background:#FFF0EB; color:#C4320A; border-radius:999px; padding:.27rem .62rem; font-size:.72rem; font-weight:700; }}
        .alert-high {{ background:#FEF3F2; border:1px solid #FECDCA; border-left:5px solid {RED}; border-radius:14px; padding:1rem 1.15rem; color:#912018; }}
        .review {{ background:var(--def-surface); color:var(--def-text); border:1px solid var(--def-border); border-radius:14px; padding:1rem 1.1rem; margin-bottom:.65rem; }}
        .review-top {{ display:flex; justify-content:space-between; color:var(--def-muted); font-size:.76rem; margin-bottom:.55rem; }}
        mark {{ background:#FEE4E2; color:#B42318; border-radius:4px; padding:1px 3px; font-weight:700; }}
        .step {{ text-align:center; border-top:3px solid var(--def-border); padding-top:.85rem; color:var(--def-muted); font-size:.85rem; }}
        .step strong {{ display:block; color:var(--def-text); margin-bottom:.2rem; }}
        div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button {{ border-radius:10px; font-weight:700; }}
        div[data-testid="stFormSubmitButton"] button[kind="primary"], div[data-testid="stButton"] button[kind="primary"] {{ background:{INDIGO}; border-color:{INDIGO}; }}
        div[data-testid="stMetric"] {{ background:var(--def-surface); color:var(--def-text); border:1px solid var(--def-border); border-radius:16px; padding:1rem; box-shadow:0 4px 14px var(--def-shadow); }}
        [data-testid="stDataFrame"] {{ border:1px solid var(--def-border); border-radius:14px; overflow:hidden; }}
        @media(max-width:700px) {{ .hero {{ padding:1.5rem; }} .hero h1 {{ font-size:2rem; }} .block-container {{ padding:.8rem 1rem 3rem; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_landing_theme() -> None:
    """Public-site styling based on the supplied DeFaketive landing mockup."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600&display=swap');
        body:has(.st-key-landing_hero) section[data-testid="stSidebar"],
        body:has(.st-key-landing_hero) [data-testid="collapsedControl"] {{ display:none !important; }}
        body:has(.st-key-landing_hero) [data-testid="stHeader"],
        body:has(.st-key-landing_hero) [data-testid="stToolbar"],
        body:has(.st-key-landing_hero) .stAppToolbar,
        body:has(.st-key-landing_hero) #MainMenu,
        body:has(.st-key-landing_hero) footer {{ display:none !important; }}
        body:has(.st-key-landing_hero) [data-testid="stAppViewContainer"] > .main {{ background:#F5F6F8; }}
        body:has(.st-key-landing_hero) .block-container {{ max-width:1440px; padding:0 .9rem 3rem !important; }}
        .st-key-landing_hero, .st-key-how_it_works, .st-key-safety_section,
        .st-key-member_section, .st-key-landing_footer, .st-key-inline_analysis,
        .st-key-inline_search_results {{
            font-family:'DM Sans',Inter,sans-serif;
        }}
        .st-key-landing_hero {{
            background:
              radial-gradient(circle at 72% 45%,rgba(52,65,190,.34),transparent 36%),
              linear-gradient(135deg,#0C1028 0%,#151936 100%);
            color:white; border-radius:0 0 30px 30px; padding:1.2rem 3.6rem 4.7rem;
            overflow:hidden; box-shadow:0 24px 60px rgba(17,21,47,.12);
        }}
        .st-key-landing_hero p, .st-key-landing_hero label {{ color:#D7DAE8 !important; }}
        .st-key-landing_hero [data-testid="stHorizontalBlock"] {{ align-items:center; }}
        .landing-brand {{ display:flex;align-items:center;gap:.7rem;color:white;font-size:1.45rem;font-weight:600;letter-spacing:-.03em;white-space:nowrap; }}
        .shield-logo {{ width:43px;height:49px;filter:drop-shadow(0 8px 20px rgba(24,200,255,.2)); }}
        .landing-nav-link {{ color:#AEB4CC;font-size:.78rem;font-weight:600;text-align:center;padding:.7rem .2rem; }}
        .st-key-landing_nav [data-testid="stButton"] button {{
            min-height:2.6rem;border:1px solid rgba(255,255,255,.12);border-radius:999px;
            background:rgba(40,49,115,.36);color:white;padding:.45rem .9rem;
        }}
        .st-key-landing_nav [data-testid="stButton"] button:hover {{ border-color:#5965FF;color:white;background:#242B66; }}
        .hero-kicker {{ color:#AEB4CC;font-size:.78rem;font-weight:600;margin:3.9rem 0 1.25rem; }}
        .hero-kicker span {{ display:inline-block;width:9px;height:9px;border-radius:50%;background:#29D99D;margin-right:.55rem;box-shadow:0 0 0 5px rgba(41,217,157,.08); }}
        .landing-title {{ font-family:'Fraunces',Georgia,serif !important;font-size:clamp(3.4rem,6vw,6.5rem) !important;line-height:.95 !important;letter-spacing:-.055em !important;color:#FFF !important;margin:0 0 1.6rem !important;max-width:700px; }}
        .landing-lead {{ max-width:650px;color:#CED2E2;font-size:1.04rem;line-height:1.85;margin-bottom:1.65rem; }}
        .st-key-hero_search {{ background:white;border-radius:22px;padding:.45rem .55rem;max-width:660px;box-shadow:0 12px 0 rgba(0,0,0,.18); }}
        .st-key-hero_search [data-testid="stForm"] {{ border:0;padding:0; }}
        .st-key-hero_search [data-testid="stTextInput"] div[data-baseweb="input"] {{ border:0 !important;background:white !important;box-shadow:none !important; }}
        .st-key-hero_search [data-testid="stTextInput"] input {{ border:0 !important;background:white !important;color:#12152E !important;font-size:1rem;box-shadow:none !important;-webkit-text-fill-color:#12152E !important; }}
        .st-key-hero_search [data-testid="stTextInput"] input::placeholder {{ color:#7B8297 !important;-webkit-text-fill-color:#7B8297 !important;opacity:1; }}
        .st-key-hero_search [data-testid="stFormSubmitButton"] button {{ width:100%;border:0;background:linear-gradient(135deg,#6157FF,#4338DB);color:white;border-radius:16px;min-height:3.3rem; }}
        .platform-row {{ display:flex;gap:.65rem;align-items:center;margin-top:1.45rem;flex-wrap:wrap; }}
        .market-pill {{ display:inline-flex;align-items:center;gap:.4rem;background:#F8FAFC;color:#344054;padding:.38rem .8rem;border-radius:999px;font-size:.72rem;font-weight:700; }}
        .market-pill.shopee b {{ color:#EE4D2D; }} .market-pill.off {{ opacity:.56; }}
        .beta {{ font-size:.55rem;background:#E4E7EC;padding:.08rem .3rem;border-radius:999px;margin-left:.15rem; }}
        .hero-stats {{ border:1px solid rgba(255,255,255,.18);background:rgba(16,21,56,.54);backdrop-filter:blur(10px);border-radius:22px;padding:1.55rem 1.65rem;margin-top:5rem;box-shadow:inset 0 1px rgba(255,255,255,.04); }}
        .stats-kicker {{ color:#C8CCDD;font-size:.72rem;font-weight:700;margin-bottom:1.4rem; }}
        .stats-grid {{ display:grid;grid-template-columns:repeat(3,1fr);gap:1rem; }}
        .stat-number {{ font-size:2.25rem;font-weight:700;letter-spacing:-.05em;color:white; }}
        .stat-number.red {{ color:#F04438; }} .stat-number.green {{ color:#32D583; }}
        .stat-label {{ color:#8F96B2;font-size:.68rem;margin-top:.2rem; }}
        .stats-bottom {{ display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;border-top:1px solid rgba(255,255,255,.55);margin-top:1.45rem;padding-top:1rem; }}
        .mini-stat {{ display:flex;align-items:center;gap:.7rem;color:white;font-size:1.1rem;font-weight:700; }}
        .mini-icon {{ display:grid;place-items:center;width:38px;height:38px;background:#E9ECF2;color:#667085;border-radius:7px; }}
        .mini-stat small {{ display:block;color:#8F96B2;font-size:.68rem;font-weight:500; }}
        .st-key-how_it_works {{ background:#FFF;border-radius:30px;padding:5.8rem 4.8rem;margin-top:1rem; }}
        .section-label {{ color:#4F46E5;font-size:.75rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;margin-bottom:.8rem; }}
        .section-title {{ color:#0E1021 !important;font-size:clamp(3.4rem,7vw,6.8rem) !important;letter-spacing:-.075em !important;line-height:.86 !important;margin:0 !important; }}
        .section-copy {{ color:#525866;font-size:1.05rem;line-height:1.75;max-width:570px;margin:.8rem 0 1.3rem; }}
        .learn-pill {{ display:inline-block;background:#5965FF;color:white;border-radius:999px;padding:.72rem 1.15rem;font-size:.8rem;font-weight:700; }}
        .process-row {{ padding:2.7rem 0;border-top:1px solid #EAECF0; }}
        .step-number {{ font-size:.77rem;color:#667085;margin-bottom:.65rem; }}
        .step-title {{ font-size:clamp(2.4rem,4vw,4.5rem) !important;line-height:1 !important;letter-spacing:-.065em !important;color:#101323 !important;margin:0 0 1rem !important; }}
        .step-title span {{ color:#4F46E5; }}
        .step-copy {{ color:#667085;line-height:1.6;max-width:300px;font-size:.92rem; }}
        .process-visual {{ min-height:230px;border-radius:999px;background:linear-gradient(135deg,#F0F2F7,#E4E7EE);padding:2rem 3rem;display:flex;align-items:center;justify-content:center;overflow:hidden; }}
        .visual-browser {{ width:min(520px,90%);background:white;border:1px solid #D0D5DD;border-radius:16px;box-shadow:0 18px 45px rgba(16,24,40,.12);padding:.7rem; }}
        .browser-bar {{ display:flex;gap:.3rem;margin-bottom:.8rem; }} .browser-bar i {{ width:7px;height:7px;border-radius:50%;background:#D0D5DD; }}
        .product-result {{ display:flex;align-items:center;gap:.8rem;border:1px solid #EAECF0;border-radius:12px;padding:.75rem; }}
        .product-image {{ width:55px;height:55px;border-radius:10px;background:linear-gradient(145deg,#1C255E,#6471FF);display:grid;place-items:center;color:white;font-size:1.3rem; }}
        .result-lines {{ flex:1; }} .result-lines b {{ color:#182230;font-size:.75rem; }} .result-lines span {{ display:block;height:6px;border-radius:10px;background:#EAECF0;margin-top:.4rem; }}
        .link-card {{ width:min(520px,90%);background:white;border-radius:18px;padding:1rem;box-shadow:0 18px 45px rgba(16,24,40,.1); }}
        .link-input {{ display:flex;align-items:center;gap:.65rem;background:#F2F4F7;border-radius:12px;padding:.8rem;color:#667085;font-size:.78rem; }}
        .link-button {{ margin-left:auto;background:#4F46E5;color:white;padding:.5rem .7rem;border-radius:9px;font-weight:700; }}
        .analysis-card {{ width:min(480px,90%);background:#101633;color:white;border-radius:18px;padding:1.2rem;box-shadow:0 20px 46px rgba(17,21,47,.24);display:grid;grid-template-columns:120px 1fr;gap:1rem;align-items:center; }}
        .mini-gauge {{ width:105px;height:105px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(#20C7C9 0 28%,#26305F 28% 100%);position:relative; }}
        .mini-gauge:after {{ content:'';position:absolute;width:76px;height:76px;border-radius:50%;background:#101633; }} .mini-gauge b {{ position:relative;z-index:1;font-size:1.6rem; }}
        .signal {{ display:flex;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,.1);padding:.4rem 0;font-size:.68rem;color:#C7CCDE; }} .signal i {{ color:#32D583;font-style:normal; }}
        .st-key-safety_section {{ background:#F0F1F3;border-radius:30px;padding:5.7rem 4.8rem;margin-top:1rem; }}
        .safety-grid {{ display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;max-width:480px; }}
        .safety-card {{ min-height:140px;background:white;border-radius:22px;padding:1.25rem;box-shadow:0 12px 30px rgba(16,24,40,.05); }}
        .safety-icon {{ font-size:1.55rem;margin-bottom:1.5rem; }} .safety-card b {{ color:#101323;display:block; }} .safety-card small {{ color:#667085; }}
        .safety-title {{ font-size:clamp(3.6rem,7vw,7.2rem) !important;line-height:.88 !important;letter-spacing:-.075em !important;color:#0E1021 !important;margin:0 0 1.5rem !important; }}
        .safety-title em {{ display:inline-block;font-style:normal;background:#5965FF;color:white;border-radius:999px;padding:.08em .28em;font-size:.36em;vertical-align:middle;letter-spacing:-.03em; }}
        .st-key-member_section {{ background:linear-gradient(135deg,#11152F,#161A3B);color:white;border-radius:30px;padding:5.7rem 4.8rem;margin-top:1rem; }}
        .member-title {{ font-size:clamp(3.5rem,7vw,7rem) !important;line-height:.88 !important;letter-spacing:-.07em !important;color:white !important;margin:0 0 1.5rem !important; }}
        .member-copy {{ color:#C6CADB;font-size:1.1rem;line-height:1.7;max-width:520px; }}
        .member-benefits {{ color:white;margin:1.5rem 0 2rem;line-height:2; }}
        .member-benefits span {{ color:#32D583;margin-right:.55rem; }}
        .st-key-member_cta [data-testid="stButton"] button {{ background:white;color:#11152F;border:0;border-radius:14px;min-height:3.2rem;padding:0 1.3rem;font-weight:800; }}
        .dashboard-preview {{ background:#07142E;border:1px solid rgba(61,195,255,.24);border-radius:22px;padding:1.2rem;box-shadow:0 28px 60px rgba(0,0,0,.28);transform:rotate(1deg); }}
        .preview-head {{ color:white;font-weight:700;margin-bottom:1rem; }} .preview-head span {{ color:#18C8FF; }}
        .preview-grid {{ display:grid;grid-template-columns:1fr 1.5fr;gap:.75rem; }}
        .preview-panel {{ background:#0B2147;border:1px solid #143466;border-radius:14px;padding:1rem;min-height:170px; }}
        .preview-risk {{ color:#8EDFFB;font-size:.65rem; }} .preview-score {{ font-size:2.3rem;font-weight:700;color:white;margin:.6rem 0; }}
        .preview-low {{ display:inline-block;padding:.3rem .6rem;border-radius:999px;background:rgba(18,183,106,.16);color:#6CE9A6;font-size:.65rem; }}
        .chart-lines {{ height:88px;margin-top:1rem;background:linear-gradient(165deg,transparent 47%,#2ED3B7 48%,#2ED3B7 50%,transparent 51%),linear-gradient(190deg,transparent 58%,#6172F3 59%,#6172F3 61%,transparent 62%); }}
        .st-key-landing_footer {{ background:#101010;color:#D0D5DD;border-radius:30px;padding:3.4rem 4.8rem;margin-top:1rem; }}
        .footer-brand {{ color:white;font-size:1.4rem;font-weight:700; }} .footer-note {{ color:#98A2B3;font-size:.8rem;line-height:1.65;max-width:550px; }}
        .footer-links {{ display:flex;justify-content:flex-end;gap:1.4rem;flex-wrap:wrap;color:#D0D5DD;font-size:.78rem; }}
        .st-key-inline_search_results {{ background:#F7F6F0;color:#11152F;border-radius:30px;padding:4.6rem 4rem;margin-top:1rem; }}
        .search-results-head {{ display:flex;justify-content:space-between;gap:2rem;align-items:flex-end;margin-bottom:1.5rem; }}
        .search-results-count {{ color:#344054;font-size:.76rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase; }}
        .search-results-title {{ color:#101323 !important;font-size:clamp(2rem,4vw,3.65rem) !important;letter-spacing:-.055em !important;line-height:1 !important;margin:.4rem 0 .55rem !important; }}
        .search-results-source {{ color:#667085;font-size:.88rem; }}
        .st-key-search_result_filters {{ border-top:1px solid #D8D9D5;border-bottom:1px solid #D8D9D5;padding:1rem 0 .45rem;margin-bottom:1.7rem; }}
        .st-key-search_result_filters label {{ color:#475467 !important;font-size:.72rem !important;font-weight:700 !important; }}
        .st-key-search_result_filters div[data-baseweb="select"] > div {{ background:white;border-color:#D0D5DD;border-radius:999px;min-height:2.55rem; }}
        [class*="st-key-search_result_card_"] {{ background:white;border:1px solid #EAECF0;border-radius:18px;padding:.8rem;box-shadow:0 10px 28px rgba(16,24,40,.055);height:100%;overflow:hidden; }}
        [class*="st-key-search_result_card_"] [data-testid="stButton"] button {{ border:0;border-radius:12px;background:#11152F;color:white;min-height:2.8rem;font-size:.78rem;font-weight:800; }}
        [class*="st-key-search_result_card_"] [data-testid="stButton"] button:hover {{ background:#4F46E5;color:white; }}
        .search-card-image {{ width:100%;height:210px;object-fit:cover;border-radius:13px;background:#E4E7EC;display:block; }}
        .search-card-placeholder {{ width:100%;height:210px;border-radius:13px;background:linear-gradient(135deg,#E4E7EC,#F2F4F7);display:grid;place-items:center;color:#98A2B3;font-size:2rem; }}
        .search-card-body {{ padding:.85rem .2rem .35rem; }}
        .search-card-flags {{ display:flex;align-items:center;justify-content:space-between;gap:.5rem;margin-bottom:.65rem; }}
        .search-card-platform {{ color:#4F46E5;background:#EEF0FF;border-radius:999px;padding:.22rem .5rem;font-size:.62rem;font-weight:800; }}
        .search-card-reliability {{ color:#067647;font-size:.67rem;font-weight:800; }}
        .search-card-name {{ color:#101323;font-size:1rem;font-weight:800;line-height:1.3;min-height:2.6rem;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden; }}
        .search-card-rating {{ color:#F79009;font-size:.75rem;font-weight:700;margin:.5rem 0; }}
        .search-card-rating span {{ color:#667085;font-weight:500;margin-left:.35rem; }}
        .search-card-bottom {{ display:flex;justify-content:space-between;align-items:flex-end;gap:.5rem;margin-top:.8rem; }}
        .search-card-price {{ color:#101323;font-size:1.3rem;font-weight:800; }}
        .search-card-risk {{ font-size:.65rem;font-weight:800;text-transform:uppercase; }}
        [class*="st-key-search_result_row_"] {{ margin-bottom:1rem; }}
        .search-pagination {{ color:#667085;text-align:center;font-size:.75rem;margin-top:.5rem; }}
        .st-key-search_back [data-testid="stButton"] button {{ border-radius:999px;border:1px solid #D0D5DD;background:white;color:#11152F;font-weight:800; }}
        .st-key-inline_analysis {{ background:#F8F9FC;color:#11152F;border-radius:30px;padding:4.6rem 4rem;margin-top:1rem; }}
        .analysis-overline {{ font-size:.78rem;font-weight:800;color:#475467;letter-spacing:.08em;text-transform:uppercase; }}
        .analysis-heading {{ color:#101323 !important;font-size:clamp(2rem,4vw,3.7rem) !important;letter-spacing:-.055em !important;line-height:1 !important;margin:.45rem 0 2.4rem !important; }}
        .analysis-product {{ background:white;border:1px solid #E4E7EC;border-radius:22px;padding:1.2rem;box-shadow:0 1px 2px rgba(16,24,40,.04); }}
        .analysis-product-grid {{ display:grid;grid-template-columns:220px 1fr auto;gap:1.25rem;align-items:center; }}
        .analysis-product-image {{ width:220px;height:180px;border-radius:16px;object-fit:cover;background:#D8D9D5; }}
        .analysis-product-placeholder {{ display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.55rem;background:linear-gradient(145deg,#EEF0FF,#F8F9FC);border:1px dashed #C7CCF5;color:#59627A;text-align:center;font-size:.72rem;font-weight:700; }}
        .analysis-product-placeholder svg {{ width:42px;height:42px;color:#3E4784; }}
        .analysis-product h3 {{ color:#11152F !important;font-size:1.55rem !important;margin:.2rem 0 .5rem !important; }}
        .analysis-meta {{ color:#667085;font-size:.76rem;margin:.25rem 0; }}
        .rating-star {{ color:#F79009;font-size:1rem;line-height:1;vertical-align:-.04em;margin-right:.2rem; }}
        .analysis-description {{ color:#525866;font-size:.8rem;line-height:1.55;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden; }}
        .analysis-price {{ color:#344054;font-size:1.6rem;font-weight:700;margin-top:.75rem; }}
        .analysis-alert {{ display:flex;align-items:center;gap:1rem;color:white;border-radius:18px;padding:1.15rem 1.4rem;margin:1rem 0;font-size:1.25rem;font-weight:700; }}
        .analysis-alert.high {{ background:#F04438; }} .analysis-alert.moderate {{ background:#F79009; }} .analysis-alert.low {{ background:#12B76A; }}
        .analysis-alert-icon {{ display:grid;place-items:center;width:34px;height:34px;border:1px solid rgba(255,255,255,.75);border-radius:9px;flex:0 0 auto; }}
        .analysis-alert small {{ margin-left:auto;font-size:.68rem;font-weight:600;color:white;opacity:.88; }}
        .analysis-panel {{ background:white;border:1px solid #E6E8EC;border-radius:20px;padding:1.35rem;height:100%; }}
        .st-key-inline_risk_panel,.st-key-inline_breakdown_panel,.st-key-inline_sentiment_panel,.st-key-inline_terms_panel,
        .st-key-product_breakdown_panel,.st-key-product_terms_panel {{ background:white;border:1px solid #E6E8EC;border-radius:20px;padding:1.25rem 1.35rem;box-shadow:0 1px 2px rgba(16,24,40,.04); }}
        .st-key-inline_risk_panel,.st-key-inline_breakdown_panel {{ min-height:390px; }}
        .st-key-inline_sentiment_panel,.st-key-inline_terms_panel {{ min-height:430px;margin-top:.4rem; }}
        .analysis-panel-title {{ color:#11152F;font-size:1.02rem;font-weight:800;margin-bottom:.22rem;letter-spacing:-.015em; }}
        .analysis-panel-subtitle {{ color:#7A8291;font-size:.74rem;line-height:1.45;margin-bottom:1rem; }}
        .formula-note {{ display:flex;justify-content:space-between;gap:1rem;color:#667085;font-size:.72rem;padding-top:.7rem;border-top:1px solid #EAECF0; }}
        .wsm-row {{ display:grid;grid-template-columns:minmax(170px,1fr) auto;align-items:center;gap:1rem;padding:1rem 0;border-top:1px solid #EEF0F3; }}
        .wsm-row:first-of-type {{ border-top:0;padding-top:.3rem; }}
        .wsm-name {{ display:flex;align-items:center;gap:.65rem;min-width:0; }}
        .wsm-dot {{ width:9px;height:9px;border-radius:50%;flex:0 0 auto; }}
        .wsm-name b {{ display:block;color:#11152F;font-size:.82rem;line-height:1.3; }}
        .wsm-name small {{ display:block;color:#7A8291;font-size:.7rem;margin-top:.16rem; }}
        .wsm-equation {{ display:grid;grid-template-columns:68px 10px 60px 10px 78px;align-items:center;gap:.28rem;text-align:right; }}
        .wsm-equation span,.wsm-equation strong {{ color:#11152F;font-size:.8rem;white-space:nowrap; }}
        .wsm-equation small {{ display:block;color:#98A2B3;font-size:.52rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.12rem; }}
        .wsm-equation i {{ color:#98A2B3;font-style:normal;text-align:center; }}
        .wsm-total {{ background:#F4F5FF;border:1px solid #DDE1FF;border-radius:13px;padding:.82rem .9rem;margin-top:.55rem; }}
        .wsm-total-head {{ display:flex;align-items:flex-end;justify-content:space-between;gap:1rem; }}
        .wsm-total-label {{ color:#242B52;font-size:.7rem;font-weight:800;letter-spacing:.04em; }}
        .wsm-total-note {{ color:#667085;font-size:.62rem;margin-top:.12rem; }}
        .wsm-total-score {{ font-size:1.22rem;font-weight:800;white-space:nowrap; }}
        .wsm-score-track {{ height:7px;background:#E1E4F5;border-radius:99px;overflow:hidden;margin-top:.7rem; }}
        .wsm-score-track span {{ display:block;height:100%;border-radius:99px; }}
        .term-panel-head {{ display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;margin-bottom:1rem; }}
        .term-status {{ display:inline-flex;align-items:center;gap:.3rem;border-radius:99px;padding:.26rem .52rem;font-size:.58rem;font-weight:800;white-space:nowrap; }}
        .term-status.clear {{ color:#3E4784;background:#EEF0FF; }} .term-status.flagged {{ color:#B42318;background:#FEF3F2; }}
        .term-state {{ display:flex;align-items:center;gap:.75rem;background:#F4F5FF;border:1px solid #DDE1FF;border-radius:13px;padding:.8rem .85rem; }}
        .term-state.flagged {{ background:#FEF3F2;border-color:#FECDCA; }}
        .term-state-icon {{ display:grid;place-items:center;width:38px;height:38px;border-radius:11px;background:white;color:#3E4784;font-size:1rem;font-weight:800;flex:0 0 auto; }}
        .term-state.flagged .term-state-icon {{ color:#D92D20; }}
        .term-state b {{ display:block;color:#242B52;font-size:.8rem; }} .term-state.flagged b {{ color:#912018; }}
        .term-state small {{ display:block;color:#667085;font-size:.68rem;margin-top:.12rem; }}
        .signal-heading {{ display:flex;align-items:center;justify-content:space-between;gap:1rem;margin:1rem 0 .65rem;color:#344054;font-size:.74rem;font-weight:750; }}
        .signal-heading span {{ color:#7A8291;font-size:.62rem;font-weight:500; }}
        .signal-chips {{ display:flex;flex-wrap:wrap;gap:.5rem; }}
        .signal-chip {{ display:inline-flex;align-items:center;gap:.38rem;color:#3E4784;background:#F8F9FC;border:1px solid #E4E7EC;border-radius:99px;padding:.42rem .58rem;font-size:.72rem;text-transform:capitalize; }}
        .signal-chip b {{ display:grid;place-items:center;min-width:19px;height:19px;border-radius:99px;background:#EEF0FF;color:#242B52;font-size:.6rem; }}
        .signal-chip.risk {{ color:#B42318;background:#FFF8F7;border-color:#FECDCA; }} .signal-chip.risk b {{ color:#B42318;background:#FEE4E2; }}
        .term-coverage {{ display:grid;grid-template-columns:repeat(3,1fr);gap:.55rem;border-top:1px solid #EEF0F3;margin-top:1rem;padding-top:.85rem; }}
        .term-coverage div {{ color:#7A8291;font-size:.58rem;line-height:1.3; }}
        .term-coverage b {{ display:block;color:#344054;font-size:.85rem;margin-bottom:.08rem; }}
        .keyword-empty {{ min-height:130px;display:grid;place-items:center;text-align:center;border:1px dashed #D8DCE3;border-radius:14px;background:#FCFCFD;padding:1.25rem;color:#7A8291;font-size:.72rem;line-height:1.5; }}
        .evidence-shell {{ background:white;border:1px solid #D8D9D5;border-radius:22px;padding:1.5rem;margin-top:1rem; }}
        .review-grid {{ display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:#D0D5DD;border:1px solid #D0D5DD;margin-top:1rem; }}
        .inline-review {{ background:white;padding:1rem;min-height:190px; }}
        .inline-review-head {{ display:flex;align-items:center;gap:.65rem;margin-bottom:.8rem; }}
        .review-avatar {{ display:grid;place-items:center;width:38px;height:38px;border-radius:50%;background:#006B58;color:white;font-weight:800; }}
        .review-author {{ color:#11152F;font-size:.78rem;font-weight:800; }} .review-stars {{ color:#12B76A;font-size:.72rem; }}
        .review-text {{ color:#344054;font-size:.78rem;line-height:1.45; }}
        .review-sentiment {{ display:inline-block;border-radius:999px;padding:.2rem .45rem;font-size:.6rem;font-weight:800;margin-top:.7rem;text-transform:uppercase; }}
        .review-sentiment.positive {{ background:#D1FADF;color:#067647; }} .review-sentiment.negative {{ background:#FEE4E2;color:#B42318; }} .review-sentiment.uncertain,.review-sentiment.neutral {{ background:#F2F4F7;color:#475467; }}
        .alternatives-shell {{ background:white;border-radius:22px;padding:1.5rem;margin-top:1rem; }}
        .alternatives-title {{ color:#11152F;font-size:1.25rem;font-weight:800; }} .alternatives-subtitle {{ color:#667085;font-size:.76rem;margin-bottom:1.1rem; }}
        .alternatives-grid {{ display:grid;grid-template-columns:repeat(3,1fr);gap:1rem; }}
        .alternative-card {{ border:1px solid #EAECF0;border-radius:18px;overflow:hidden;background:white;box-shadow:0 10px 25px rgba(16,24,40,.06);text-decoration:none; }}
        .alternative-cover {{ height:125px;background:linear-gradient(135deg,#0F9F68,#31B5A4);display:grid;place-items:center;color:white;font-size:2rem;overflow:hidden; }}
        .alternative-cover img {{ width:100%;height:100%;object-fit:cover; }}
        .alternative-body {{ padding:1rem; }} .alternative-name {{ color:#11152F;font-weight:800;font-size:.88rem;min-height:2.6rem; }}
        .alternative-meta {{ color:#667085;font-size:.67rem;line-height:1.45;margin-top:.55rem; }}
        .alternative-score {{ color:#12B76A;font-size:.72rem;font-weight:800; }} .alternative-price {{ color:#11152F;font-size:1.1rem;font-weight:800;margin-top:.5rem; }}
        [class*="recommendation_action"] {{ background:#EEF0FF;border:1px solid #D5D9FF;border-radius:18px;padding:1rem 1.1rem;margin-top:1rem;box-shadow:0 8px 24px rgba(62,71,132,.08); }}
        .recommendation-action-title {{ color:#242B52;font-size:.9rem;font-weight:800;margin-bottom:.2rem; }}
        .recommendation-action-copy {{ color:#59627A;font-size:.7rem;line-height:1.45; }}
        [class*="recommendation_action"] [data-testid="stButton"] button {{ width:100%;min-height:3rem;border:0 !important;border-radius:12px !important;background:#3E4784 !important;color:#FFF !important;font-weight:800 !important;box-shadow:0 7px 16px rgba(62,71,132,.2); }}
        [class*="recommendation_action"] [data-testid="stButton"] button * {{ color:#FFF !important; }}
        [class*="recommendation_action"] [data-testid="stButton"] button:hover {{ background:#2F376C !important;color:#FFF !important; }}
        @media(max-width:900px) {{
          .st-key-landing_hero,.st-key-how_it_works,.st-key-safety_section,.st-key-member_section,.st-key-landing_footer,.st-key-inline_analysis,.st-key-inline_search_results {{ padding-left:1.5rem;padding-right:1.5rem; }}
          .hero-stats {{ margin-top:2rem; }} .stats-grid {{ grid-template-columns:1fr 1fr; }} .stats-grid > div:last-child {{ grid-column:1/-1; }}
          .process-visual {{ border-radius:36px;padding:1.3rem; }} .analysis-card {{ grid-template-columns:1fr;justify-items:center; }}
          .landing-nav-link, .st-key-landing_nav [data-testid="stButton"] {{ display:none !important; }} .review-grid,.alternatives-grid {{ grid-template-columns:1fr 1fr; }} .analysis-product-grid {{ grid-template-columns:160px 1fr; }} .analysis-product-image {{ width:160px;height:150px; }}
          [class*="st-key-search_result_row_"] [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap; }} [class*="st-key-search_result_row_"] [data-testid="stColumn"] {{ min-width:calc(50% - 1rem);flex:1 1 calc(50% - 1rem); }}
        }}
        @media(max-width:600px) {{
          .block-container {{ padding:0 .35rem 2rem !important; }} .st-key-landing_hero {{ padding-top:.8rem;padding-bottom:3rem;border-radius:0 0 22px 22px; }}
          .landing-title {{ font-size:3.25rem; }} .hero-kicker {{ margin-top:2.5rem; }} .hero-stats {{ padding:1.1rem; }} .stat-number {{ font-size:1.65rem; }}
          .stats-bottom {{ grid-template-columns:1fr; }} .st-key-how_it_works,.st-key-safety_section,.st-key-member_section {{ padding-top:3.5rem;padding-bottom:3.5rem;border-radius:22px; }}
          .section-title,.safety-title,.member-title {{ font-size:3.35rem; }} .safety-grid {{ grid-template-columns:1fr; }} .preview-grid {{ grid-template-columns:1fr; }} .review-grid,.alternatives-grid {{ grid-template-columns:1fr; }} .analysis-product-grid {{ grid-template-columns:1fr; }} .analysis-product-image {{ width:100%;height:220px; }} .analysis-alert {{ font-size:.95rem; }} .analysis-alert small {{ display:none; }}
          .st-key-inline_search_results {{ padding-top:3rem;padding-bottom:3rem;border-radius:22px; }} .search-results-head {{ display:block; }} .search-card-image,.search-card-placeholder {{ height:200px; }} [class*="st-key-search_result_row_"] [data-testid="stColumn"] {{ min-width:100%;flex-basis:100%; }}
          .wsm-row {{ grid-template-columns:1fr;gap:.75rem; }} .wsm-equation {{ grid-template-columns:1fr auto 1fr auto 1fr;text-align:left; }} .wsm-total-head {{ align-items:flex-start; }}
          .st-key-inline_risk_panel,.st-key-inline_breakdown_panel,.st-key-inline_sentiment_panel,.st-key-inline_terms_panel {{ min-height:0; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(score: Any, scale: Any | None = None) -> str:
    level = risk_level(score, scale)
    dot = {"Low": "●", "Moderate": "▲", "High": "◆"}[level]
    return f'<span class="risk-pill" style="background:{RISK_COLORS[level]}">{dot} {level} risk</span>'


def money(value: Any) -> str:
    amount = price_value(value)
    return f"₱{amount:,.0f}" if amount is not None else str(value or "Price unavailable")


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="card"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="muted" style="font-size:.75rem;margin-top:.4rem">{html.escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def analyzed_file(path: str, modified: float) -> list[dict[str, Any]]:
    del modified
    products = load_products(path)
    if analysis_is_current(products):
        return products
    return analyze_products(products)


def get_products() -> list[dict[str, Any]]:
    uploaded = st.session_state.get("uploaded_products")
    if uploaded is not None:
        if not analysis_is_current(uploaded):
            uploaded = analyze_products(uploaded)
            st.session_state.uploaded_products = uploaded
        return uploaded
    files = result_files()
    if not files:
        return []
    failures = []
    for path in files:
        try:
            products = analyzed_file(str(path), path.stat().st_mtime)
            if not products or not has_usable_products(products):
                continue
            save_products(products, source=str(path))
            return products
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")
    if failures:
        st.warning("No usable saved result could be loaded. " + failures[0])
    return []


def request_page(page: str) -> None:
    """Queue a page change so the sidebar widget and routed page stay in sync."""
    if page not in NAVIGATION_PAGES:
        raise ValueError(f"Unknown page: {page}")
    st.session_state.page = page
    st.session_state["_navigation_target"] = page


def back_to_home_button(key: str) -> None:
    """Render a reliable public-page return action."""
    if st.button("← Back to home", key=key):
        request_page("Search")
        st.rerun()


def set_selected(product: dict[str, Any]) -> None:
    st.session_state.selected_link = product.get("link")
    request_page("Product analysis")


def selected_product(products: list[dict[str, Any]]) -> dict[str, Any] | None:
    link = st.session_state.get("selected_link")
    return next((item for item in products if item.get("link") == link), products[0] if products else None)


def run_live_scrape(keyword: str, product_count: int, review_count: int, platform: str | None = None) -> list[dict[str, Any]]:
    parsed = urlsplit(keyword)
    direct_url = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    try:
        marketplace = detect_marketplace(keyword, platform or "shopee")
    except ValueError as exc:
        st.error(str(exc))
        return []
    direct_id = hashlib.sha256(keyword.strip().encode("utf-8")).hexdigest()[:12]
    target = output_path(f"direct_{direct_id}" if direct_url else keyword, marketplace)
    if marketplace == "shopee":
        command = [sys.executable, str(ROOT / "src" / "retriv.py"), "-n", str(product_count), "-r", str(review_count), "--output", str(target), "--no-prompt", "--verification-timeout", str(MANUAL_VERIFICATION_TIMEOUT)]
        command.extend(["--product-url", keyword] if direct_url else ["-k", keyword])
    elif marketplace == "lazada":
        command = [sys.executable, str(ROOT / "lazada-scraper" / "src" / "lazada_scraper.py"), "-n", str(product_count), "-r", str(review_count), "--output", str(target), "--no-verification-pause", "--verification-timeout", "600"]
        command.extend(["--product-url", keyword] if direct_url else [keyword])
    else:
        browser_profile = ROOT / ".browser_profiles" / marketplace
        browser_profile.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT / "temu-scraper" / "src" / "temu_scraper.py"), "-n", str(product_count), "-r", str(review_count), "--output", str(target), "--no-verification-pause", "--verification-timeout", "600", "--chrome-user-data-dir", str(browser_profile)]
        command.extend(["--product-url", keyword] if direct_url else [keyword])
    with st.status("Starting live analysis…", expanded=True) as status:
        st.write(f"1 of 3 · Scraping public {marketplace.title()} product reviews…")
        st.caption(
            "A dedicated marketplace Chrome window will open. If a login or CAPTCHA appears, complete it "
            "yourself within 15 minutes and leave the window open. DeFaketive resumes automatically after "
            "verification and remembers the session for later runs."
        )
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=SCRAPER_PROCESS_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            status.update(label="Marketplace verification timed out", state="error")
            st.error("The CAPTCHA or login was not completed within 15 minutes. Start a new scan when you are ready to finish the verification.")
            return []
        if result.returncode:
            status.update(label="The scrape could not be completed", state="error")
            diagnostic = f"{result.stdout}\n{result.stderr}"
            if "SHOPEE_VERIFICATION_BLOCKED" in diagnostic:
                st.error("Shopee showed 'Try Again Later' instead of a solvable CAPTCHA. There is nothing to complete on that page; wait for Shopee's cooldown before another live attempt, or use a saved result.")
            else:
                st.code((result.stderr or result.stdout)[-3000:], language="text")
            return []
        st.write("2 of 3 · Analyzing English, Filipino, and Taglish sentiment…")
        products = load_products(target)
        if not has_usable_products(products):
            status.update(label="No usable marketplace data was collected", state="error")
            st.error("The marketplace did not return a usable product. Your previous saved results were left unchanged.")
            return []
        analyzed = analyze_products(products)
        target.write_bytes(json_bytes(analyzed))
        save_products(analyzed, source=str(target))
        st.write("3 of 3 · Computing weighted risk and reliability scores…")
        status.update(label=f"Analysis complete · {len(analyzed)} products found", state="complete")
        analyzed_file.clear()
        return analyzed


def render_recommendations(
    product: dict[str, Any],
    products: list[dict[str, Any]],
    limit: int,
    key_prefix: str,
) -> None:
    """Show analyzed Low Risk matches and allow an on-demand marketplace search."""
    try:
        saved_products = load_saved_products()
    except Exception:
        saved_products = []
    catalog = merge_product_catalogs(products, saved_products)
    recommendations = legacy_rank_recommendations(product, catalog, limit=limit)
    high_risk = product_risk_level(product) == "High"
    title = "Safer alternatives" if high_risk else "Recommended similar products"
    if recommendations:
        cards = []
        for alternative in recommendations:
            summary = alternative.get("sentiment_summary") or {}
            positive = float((summary.get("sentiment_ratios") or {}).get("positive") or 0)
            if positive > 1:
                positive /= 100
            review_count = int(summary.get("review_count") or len(alternative.get("comments") or []))
            alt_name = html.escape(str(alternative.get("name") or "Recommended product"))
            alt_url = html.escape(str(alternative.get("link") or "#"), quote=True)
            alt_image = html.escape(product_image_url(alternative), quote=True)
            alt_cover = f'<img src="{alt_image}" alt="">' if alt_image else "&#9671;"
            cards.append(
                f'<a class="alternative-card" href="{alt_url}" target="_blank" rel="noopener">'
                f'<div class="alternative-cover">{alt_cover}</div><div class="alternative-body">'
                f'<div class="alternative-score">{product_reliability(alternative):.0f}% reliability &middot; Low Risk</div>'
                f'<div class="alternative-name">{alt_name}</div>'
                f'<div class="alternative-price">{html.escape(money(alternative.get("price")))}</div>'
                f'<div class="alternative-meta">{html.escape(platform_name(alternative))} &middot; '
                f'{positive:.0%} positive &middot; {review_count:,} reviews</div></div></a>'
            )
        st.markdown(
            f'<section class="alternatives-shell"><div class="alternatives-title">{title}</div>'
            f'<div class="alternatives-subtitle">Similar products with Low Risk, good positive-review ratios, '
            f'and at least five analyzed reviews.</div><div class="alternatives-grid">{"".join(cards)}</div></section>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No sufficiently similar Low Risk product is available in the analyzed catalog yet.")

    query = recommendation_search_query(product)
    button_label = "Find more reviewed alternatives" if recommendations else "Find similar reviewed products"
    action_key = f"{key_prefix.replace('-', '_')}_recommendation_action"
    with st.container(key=action_key):
        action_copy, action_button = st.columns([1.6, 1], vertical_alignment="center")
        action_copy.markdown(
            '<div class="recommendation-action-title">Compare with reviewed alternatives</div>'
            '<div class="recommendation-action-copy">Search the same marketplace, analyze the reviews, '
            'and rank similar products by reliability.</div>',
            unsafe_allow_html=True,
        )
        find_recommendations = action_button.button(
            button_label,
            key=f"{key_prefix}-find-recommendations",
            type="primary",
            width="stretch",
        )
    if find_recommendations:
        discovered = run_live_scrape(query, 6, 30, platform=product_platform(product))
        if discovered:
            st.session_state.uploaded_products = merge_product_catalogs(products, discovered)
            st.rerun()
    st.caption(f'Recommendation search: “{query}” · Results are ranked only after their reviews are analyzed.')


def landing_page(products: list[dict[str, Any]]) -> None:
    inject_landing_theme()
    review_total = sum(len(item.get("comments") or []) for item in products)
    high_total = sum(product_risk_level(item) == "High" for item in products)
    analyzed_total = sum(bool(item.get("sentiment_summary")) for item in products)
    reliabilities = [product_reliability(item) for item in products]
    average_reliability = sum(reliabilities) / len(reliabilities) if reliabilities else 0

    with st.container(key="landing_hero"):
        with st.container(key="landing_nav"):
            brand, nav_a, nav_b, nav_c, spacer, report, dashboard = st.columns([2.7, 1.1, 1.1, 1.1, .7, 1.25, 1.4])
            brand.markdown(
                """<div class="landing-brand">
                <svg class="shield-logo" viewBox="0 0 64 72" aria-hidden="true"><defs><linearGradient id="shieldGradient" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#18D7FF"/><stop offset="1" stop-color="#1975FF"/></linearGradient></defs><path d="M32 3C23 9 15 11 7 12v20c0 18 10 29 25 37 15-8 25-19 25-37V12C49 11 41 9 32 3Z" fill="none" stroke="url(#shieldGradient)" stroke-width="5"/><path d="m21 35 7 7 15-17" fill="none" stroke="#18C8FF" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/></svg>
                <span>DeFaketive</span></div>""",
                unsafe_allow_html=True,
            )
            nav_a.markdown('<div class="landing-nav-link">How it works</div>', unsafe_allow_html=True)
            nav_b.markdown('<div class="landing-nav-link">Risk scoring</div>', unsafe_allow_html=True)
            nav_c.markdown('<div class="landing-nav-link">Support</div>', unsafe_allow_html=True)
            if report.button("Reports", use_container_width=True, key="landing-reports"):
                request_page("Search results")
                st.rerun()
            if dashboard.button("Dashboard", type="primary", use_container_width=True, key="landing-dashboard"):
                request_page("Admin overview")
                st.rerun()

        hero_copy, hero_numbers = st.columns([1.05, .95], gap="large")
        with hero_copy:
            st.markdown(
                f'<div class="hero-kicker"><span></span>{review_total:,} real reviews available in the local research dataset</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<h1 class="landing-title">Bago mo bilhin,<br>i-check muna.</h1>', unsafe_allow_html=True)
            st.markdown(
                '<p class="landing-lead">Paste a supported marketplace link or search for a gadget to collect and analyze its reviews. '
                'DeFaketive detects English, Filipino, and Taglish risk signals, '
                'and explains what influenced the product score before you buy.</p>',
                unsafe_allow_html=True,
            )
            with st.container(key="hero_search"):
                with st.form("search", border=False):
                    query_col, submit_col = st.columns([4.7, 1.35], vertical_alignment="center")
                    keyword = query_col.text_input(
                        "Product link or search",
                        placeholder="Paste a product link or search for one",
                        label_visibility="collapsed",
                    )
                    submitted = submit_col.form_submit_button("Check now", type="primary", use_container_width=True)
                    source_col, live_col = st.columns([1.4, 2.6], vertical_alignment="bottom")
                    shopper_platform = source_col.selectbox(
                        "Marketplace",
                        options=("shopee", "lazada", "temu"),
                        format_func=lambda value: {"shopee": "Shopee PH", "lazada": "Lazada", "temu": "Temu PH"}[value],
                        label_visibility="collapsed",
                    )
                    live_scan = live_col.checkbox(
                        "Live marketplace scan",
                        value=True,
                        help="Collect fresh public reviews in a dedicated browser. Marketplace verification may be required.",
                    )
            st.markdown(
                """<div class="platform-row" aria-label="Supported marketplaces">
                <span class="market-pill shopee"><b>S</b> Shopee PH</span>
                <span class="market-pill shopee">Lazada <span class="beta">BETA</span></span>
                <span class="market-pill off">TikTok Shop <span class="beta">SOON</span></span>
                <span class="market-pill shopee">Temu PH <span class="beta">BETA</span></span>
                </div>""",
                unsafe_allow_html=True,
            )

        with hero_numbers:
            st.markdown(
                f"""<div class="hero-stats">
                <div class="stats-kicker">Current DeFaketive dataset</div>
                <div class="stats-grid">
                  <div><div class="stat-number">{len(products):,}</div><div class="stat-label">Products collected</div></div>
                  <div><div class="stat-number red">{high_total:,}</div><div class="stat-label">High-risk flags</div></div>
                  <div><div class="stat-number green">{average_reliability:.0f}%</div><div class="stat-label">Average reliability</div></div>
                </div>
                <div class="stats-bottom">
                  <div class="mini-stat"><span class="mini-icon">&#9733;</span><span>{review_total:,}<small>Sampled reviews</small></span></div>
                  <div class="mini-stat"><span class="mini-icon">&#10003;</span><span>{analyzed_total:,}<small>Products analyzed</small></span></div>
                </div></div>""",
                unsafe_allow_html=True,
            )

        if submitted:
            if not keyword.strip():
                st.error("Enter a product name or supported marketplace link to begin.")
            else:
                submitted_query = keyword.strip()
                parsed_query = urlsplit(submitted_query)
                is_link_search = parsed_query.scheme in {"http", "https"} and bool(parsed_query.netloc)
                for stale_key in ("search_mode", "search_result_query", "search_result_page", "inline_result_link", "selected_link"):
                    st.session_state.pop(stale_key, None)
                cached = cached_product_matches(products, submitted_query, 1 if is_link_search else 6)
                found = (
                    run_live_scrape(
                        submitted_query,
                        1 if is_link_search else 6,
                        30,
                        shopper_platform,
                    )
                    if live_scan
                    else cached
                )
                if live_scan and not found and cached:
                    st.info("The live scan was unavailable, so DeFaketive is showing the most recent saved analysis instead.")
                    found = cached
                if found:
                    st.session_state.uploaded_products = found
                    st.session_state.search_result_query = submitted_query
                    st.session_state.search_result_page = 0
                    for filter_key in ("inline-platform-filter", "inline-price-filter", "inline-risk-filter", "inline-sort"):
                        st.session_state.pop(filter_key, None)
                    if is_link_search:
                        st.session_state.search_mode = "link"
                        st.session_state.selected_link = found[0].get("link")
                        st.session_state.inline_result_link = found[0].get("link")
                    else:
                        st.session_state.search_mode = "name"
                        st.session_state.pop("selected_link", None)
                        st.session_state.pop("inline_result_link", None)
                    request_page("Search")
                    st.rerun()
                else:
                    if live_scan:
                        st.warning("No usable product data was collected and no saved analysis matched this search.")
                    else:
                        st.warning("No saved analysis matched this search. Enable Live marketplace scan to collect fresh reviews.")

    inline_link = st.session_state.get("inline_result_link")
    inline_product = next((item for item in products if item.get("link") == inline_link), None)
    if inline_product:
        if st.session_state.get("search_mode") == "name":
            with st.container(key="search_back"):
                if st.button("← Back to product results", key="back-to-inline-results"):
                    st.session_state.pop("inline_result_link", None)
                    st.session_state.pop("selected_link", None)
                    st.rerun()
        inline_product_analysis(inline_product, products)
    elif st.session_state.get("search_mode") == "name" and st.session_state.get("search_result_query") and products:
        inline_name_results(st.session_state.search_result_query, products)

    with st.container(key="how_it_works"):
        heading, intro = st.columns([1, 1], gap="large")
        with heading:
            st.markdown('<div id="how-it-works" class="section-label">Transparent by design</div><h2 class="section-title">How it<br>works</h2>', unsafe_allow_html=True)
        with intro:
            st.markdown(
                '<p class="section-copy">DeFaketive turns an unfamiliar listing into an explainable review-risk report. Every signal remains traceable to the sampled review and matched term, so the result is easier to inspect.</p>'
                '<span class="learn-pill">Risk score = 30% negative-review ratio + 70% detected-risk ratio</span>',
                unsafe_allow_html=True,
            )

        step_text, step_visual = st.columns([.52, 1.48], gap="large", vertical_alignment="center")
        with step_text:
            st.markdown('<div class="process-row"><div class="step-number">Step 01</div><h3 class="step-title">Find <span>&#8599;</span></h3><p class="step-copy">Search Shopee or copy a Shopee, Lazada, or Temu listing you want to investigate.</p></div>', unsafe_allow_html=True)
        with step_visual:
            st.markdown(
                """<div class="process-visual"><div class="visual-browser"><div class="browser-bar"><i></i><i></i><i></i></div><div class="product-result"><div class="product-image">&#128241;</div><div class="result-lines"><b>Wireless gadget listing</b><span style="width:88%"></span><span style="width:58%"></span></div><div style="color:#F79009;font-weight:800">4.7 &#9733;</div></div></div></div>""",
                unsafe_allow_html=True,
            )

        step_text, step_visual = st.columns([.52, 1.48], gap="large", vertical_alignment="center")
        with step_text:
            st.markdown('<div class="process-row"><div class="step-number">Step 02</div><h3 class="step-title">Paste <span>&#8599;</span></h3><p class="step-copy">Paste the product link or enter its name. The live scraper collects the public listing and review sample.</p></div>', unsafe_allow_html=True)
        with step_visual:
            st.markdown(
                """<div class="process-visual"><div class="link-card"><div style="color:#101828;font-weight:700;margin:0 0 .8rem">Check a product</div><div class="link-input"><span>&#128279;</span><span>shopee.ph/product-name-i.123.456</span><span class="link-button">Analyze</span></div><div style="font-size:.68rem;color:#98A2B3;margin-top:.7rem">Shopee search plus Shopee, Lazada, and Temu product links are supported.</div></div></div>""",
                unsafe_allow_html=True,
            )

        step_text, step_visual = st.columns([.52, 1.48], gap="large", vertical_alignment="center")
        with step_text:
            st.markdown('<div class="process-row"><div class="step-number">Step 03</div><h3 class="step-title">Analyze <span>&#8599;</span></h3><p class="step-copy">Inspect the risk gauge, sentiment mix, failure keywords, duplicate markers, and highlighted evidence.</p></div>', unsafe_allow_html=True)
        with step_visual:
            st.markdown(
                """<div class="process-visual"><div class="analysis-card"><div class="mini-gauge"><b>28</b></div><div><div style="font-size:.72rem;color:#8EDFFB;font-weight:700">LOW RISK</div><div class="signal"><span>Packaging signals</span><i>&#10003;</i></div><div class="signal"><span>Review consistency</span><i>&#10003;</i></div><div class="signal"><span>Failure keywords</span><i>2 found</i></div><div class="signal"><span>Duplicate reviews</span><i>checked</i></div></div></div></div>""",
                unsafe_allow_html=True,
            )

    with st.container(key="safety_section"):
        features, copy = st.columns([.9, 1.1], gap="large", vertical_alignment="center")
        with features:
            st.markdown(
                """<div class="safety-grid">
                <div class="safety-card"><div class="safety-icon">&#128269;</div><b>Review evidence</b><small>See the text behind every warning.</small></div>
                <div class="safety-card"><div class="safety-icon">&#128200;</div><b>Sentiment trends</b><small>Compare positive, uncertain, and negative feedback.</small></div>
                <div class="safety-card"><div class="safety-icon">&#128737;</div><b>Risk signals</b><small>Spot defect and counterfeit language.</small></div>
                <div class="safety-card"><div class="safety-icon">&#8644;</div><b>Safer options</b><small>Rank stored products by reliability.</small></div>
                </div>""",
                unsafe_allow_html=True,
            )
        with copy:
            st.markdown(
                '<div class="section-label">Decision support for shoppers</div><h2 class="safety-title">Search for<br><em>evidence</em> safety.</h2>'
                '<p class="section-copy">A high score is a prompt to look closer, not a legal determination that a seller or product is fraudulent. DeFaketive keeps the calculation visible and the review evidence within reach.</p>',
                unsafe_allow_html=True,
            )

    with st.container(key="member_section"):
        member_copy, member_preview = st.columns([.9, 1.1], gap="large", vertical_alignment="center")
        with member_copy:
            st.markdown(
                '<div class="section-label" style="color:#8EDFFB">Continue exploring</div><h2 class="member-title">Your review<br>risk dashboard.</h2>'
                '<p class="member-copy">Open the working dashboard to browse saved products, inspect the lexicon, manage scraper jobs, and export the local research dataset.</p>'
                '<div class="member-benefits"><div><span>&#10003;</span>Search and report history</div><div><span>&#10003;</span>Product-level evidence</div><div><span>&#10003;</span>Administrator research tools</div></div>',
                unsafe_allow_html=True,
            )
            with st.container(key="member_cta"):
                if st.button("Open dashboard  ->", key="member-dashboard"):
                    request_page("Admin overview")
                    st.rerun()
        with member_preview:
            st.markdown(
                """<div class="dashboard-preview"><div class="preview-head"><span>&#9671;</span> DeFaketive &nbsp; / &nbsp; Product analysis</div><div class="preview-grid"><div class="preview-panel"><div class="preview-risk">RISK SCORE</div><div class="preview-score">28<small style="font-size:.75rem;color:#8AA0C8"> /100</small></div><span class="preview-low">Low risk</span><div style="font-size:.68rem;color:#8AA0C8;margin-top:1rem">This sample shows fewer high-risk indicators.</div></div><div class="preview-panel"><div style="font-size:.7rem;color:#D0D5DD">Sentiment over sampled reviews</div><div class="chart-lines"></div><div class="signal"><span>Positive</span><i>68%</i></div><div class="signal"><span>Uncertain</span><i>20%</i></div><div class="signal"><span>Negative</span><i>12%</i></div></div></div></div>""",
                unsafe_allow_html=True,
            )

    with st.container(key="landing_footer"):
        footer_left, footer_right = st.columns([1.2, 1])
        footer_left.markdown(
            """<div class="footer-brand">&#9671; DeFaketive</div><p class="footer-note">An explainable review-risk research prototype for Philippine e-commerce. Shopee is the primary integration; Lazada and Temu are beta connectors. Not sponsored, endorsed, or operated by any marketplace.</p><p class="footer-note">&copy; 2026 DeFaketive research project.</p>""",
            unsafe_allow_html=True,
        )
        footer_right.markdown(
            '<div class="footer-links"><span>How it works</span><span>Methodology</span><span>Feedback</span><span>Dashboard</span></div>',
            unsafe_allow_html=True,
        )


def results_page(products: list[dict[str, Any]]) -> None:
    back_to_home_button("reports-back-home")
    st.markdown('<div class="eyebrow">Shopper portal</div>', unsafe_allow_html=True)
    st.title("Search results")
    st.caption("Products are ranked by reliability score. Open a result to inspect the underlying review evidence.")
    if not products:
        st.info("No scrape results yet. Start a live analysis from Search or upload a scraper JSON file in the sidebar.")
        return
    f1, f2, f3 = st.columns([2, 2, 2])
    query = f1.text_input("Filter products", placeholder="Search within results")
    risk_filter = f2.multiselect("Risk level", ["Low", "Moderate", "High"], default=["Low", "Moderate", "High"])
    sort = f3.selectbox("Sort by", ["Reliability score", "Price: low to high", "Review count"])
    visible = [
        p
        for p in products
        if query.casefold() in p.get("name", "").casefold()
        and product_risk_level(p) in risk_filter
    ]
    if sort == "Reliability score":
        visible.sort(key=product_reliability, reverse=True)
    elif sort == "Price: low to high":
        visible.sort(key=lambda p: price_value(p.get("price")) or float("inf"))
    else:
        visible.sort(key=lambda p: len(p.get("comments") or []), reverse=True)
    for row_start in range(0, len(visible), 3):
        columns = st.columns(3)
        for column, product in zip(columns, visible[row_start:row_start + 3]):
            summary = product.get("sentiment_summary") or {}
            score = summary.get("risk_score", 0)
            with column:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                image = product_image_url(product)
                if image:
                    st.image(image, use_container_width=True)
                st.markdown(
                    f'<span class="platform-pill">{platform_name(product)}</span> '
                    f'{risk_badge(score, summary.get("risk_score_scale"))}',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{product.get('name') or 'Unnamed product'}**")
                st.markdown(f"### {money(product.get('price'))}")
                st.caption(f"★ {product.get('rating') or '—'} · {len(product.get('comments') or []):,} sampled reviews")
                reliability = product_reliability(product)
                st.progress(reliability / 100, text=f"{reliability:.0f}% reliability")
                if st.button("View analysis →", key=f"view-{row_start}-{product.get('link')}", use_container_width=True):
                    set_selected(product)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)


def gauge(score_100: float) -> go.Figure:
    level = risk_level(score_100, "percent")
    return go.Figure(go.Indicator(
        mode="gauge+number", value=score_100,
        number={"valueformat": ".1f", "suffix": "/100", "font": {"size": 36, "color": INK}},
        title={"text": f"{level.upper()} RISK", "font": {"size": 14, "color": RISK_COLORS[level]}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "white"}, "bar": {"color": RISK_COLORS[level], "thickness": .28},
               "bgcolor": "white", "borderwidth": 0,
               "steps": [{"range": [0, 30], "color": "#D1FADF"}, {"range": [30, 60], "color": "#FEF0C7"}, {"range": [60, 100], "color": "#FEE4E2"}],
               "threshold": {"line": {"color": INK, "width": 3}, "thickness": .7, "value": score_100}}))


def render_weighted_score_breakdown(
    product: dict[str, Any], score: float, risk_color: str
) -> None:
    """Render WSM rates, weights, and point contributions without mixing scales."""
    values = weighted_risk_breakdown(product)
    if values.get("method") in {
        "confidence_severity_frequency_bayesian",
        "calibrated_probability_expected_severity",
    }:
        review_count = int(values["review_count"])
        risk_reviews = int(values["risk_reviews"])
        raw_points = float(values["raw_risk_points"])
        evidence_confidence = float(values["evidence_confidence"])
        prior_points = float(values["prior_points"])
        average_probability = float(values["average_risk_probability"])
        average_severity = float(values["average_risk_severity"])
        st.markdown(
            '<div class="analysis-panel-title">Evidence-weighted risk breakdown</div>'
            '<div class="analysis-panel-subtitle">Calibrated risk probability is multiplied by expected severity for each review, averaged, then adjusted for sample size.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<div class="wsm-row">
              <div class="wsm-name"><span class="wsm-dot" style="background:#F04438"></span><div>
              <b>Review risk evidence</b><small>{risk_reviews:,} of {review_count:,} reviews flagged</small></div></div>
              <div class="wsm-equation">
                <span><small>Avg probability</small>{average_probability:.1%}</span><i>+</i>
                <span><small>Avg severity</small>{average_severity:.2f}/3</span><i>=</i>
                <strong><small>Raw score</small>{raw_points:.1f}</strong>
              </div>
            </div>
            <div class="wsm-row">
              <div class="wsm-name"><span class="wsm-dot" style="background:#7F56D9"></span><div>
              <b>Sample confidence</b><small>More unique reviews reduce prior adjustment</small></div></div>
              <div class="wsm-equation">
                <span><small>Evidence</small>{evidence_confidence:.1%}</span><i>+</i>
                <span><small>Prior points</small>{prior_points:.1f}</span><i>=</i>
                <strong><small>Final</small>{score:.1f}</strong>
              </div>
            </div>
            <div class="wsm-total"><div class="wsm-total-head"><div><div class="wsm-total-label">FINAL PRODUCT RISK</div>
            <div class="wsm-total-note">Sentiment does not override detected product risk.</div></div><div class="wsm-total-score" style="color:{risk_color}">{score:.1f}/100</div></div>
            <div class="wsm-score-track"><span style="width:{score:.2f}%;background:{risk_color}"></span></div></div>""",
            unsafe_allow_html=True,
        )
        return
    review_count = int(values["review_count"])
    negative_reviews = int(values["negative_reviews"])
    risk_reviews = int(values["risk_reviews"])
    negative_rate = float(values["negative_ratio"])
    defect_rate = float(values["defect_review_ratio"])
    negative_points = float(values["negative_points"])
    defect_points = float(values["defect_points"])
    calculated_score = float(values["total_points"])
    score_is_current = abs(calculated_score - score) <= 0.11
    total_note = (
        f"{negative_points:.1f} + {defect_points:.1f} contribution points"
        if score_is_current
        else f"Saved result shown; re-analyze to refresh the component totals"
    )

    st.markdown(
        '<div class="analysis-panel-title">Weighted score breakdown</div>'
        '<div class="analysis-panel-subtitle">Observed rate × configured weight = contribution to the final score.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="wsm-row">
          <div class="wsm-name"><span class="wsm-dot" style="background:#F79009"></span><div>
          <b>Negative sentiment</b><small>{negative_reviews:,} of {review_count:,} sampled reviews</small></div></div>
          <div class="wsm-equation">
            <span><small>Rate</small>{negative_rate:.1%}</span><i>&times;</i>
            <span><small>Weight</small>30%</span><i>=</i>
            <strong><small>Points</small>{negative_points:.1f}</strong>
          </div>
        </div>
        <div class="wsm-row">
          <div class="wsm-name"><span class="wsm-dot" style="background:#F04438"></span><div>
          <b>Detected product risk</b><small>{risk_reviews:,} of {review_count:,} sampled reviews</small></div></div>
          <div class="wsm-equation">
            <span><small>Rate</small>{defect_rate:.1%}</span><i>&times;</i>
            <span><small>Weight</small>70%</span><i>=</i>
            <strong><small>Points</small>{defect_points:.1f}</strong>
          </div>
        </div>
        <div class="wsm-total"><div class="wsm-total-head"><div><div class="wsm-total-label">FINAL WEIGHTED RISK</div>
        <div class="wsm-total-note">{total_note}</div></div><div class="wsm-total-score" style="color:{risk_color}">{score:.1f}/100</div></div>
        <div class="wsm-score-track"><span style="width:{score:.2f}%;background:{risk_color}"></span></div></div>""",
        unsafe_allow_html=True,
    )


def render_term_or_signal_panel(product: dict[str, Any]) -> None:
    """Render compact risk-term evidence or a neutral language fallback."""
    values = weighted_risk_breakdown(product)
    summary = product.get("sentiment_summary") or {}
    review_count = int(values["review_count"])
    risk_reviews = int(values["risk_reviews"])
    try:
        duplicate_count = max(0, int(summary.get("duplicate_review_count")))
    except (TypeError, ValueError):
        duplicate_count = sum(
            1 for review in product.get("comments") or []
            if isinstance(review, dict) and review.get("is_duplicate")
        )

    keywords = risk_keyword_counts(product)
    signals = review_signal_counts(product) if not keywords else {}
    if keywords:
        title = "Detected defect terms"
        subtitle = "Active, non-negated matches in unique sampled reviews."
        status_class = "flagged"
        status_icon = "!"
        status_title = f"{risk_reviews:,} review{'s' if risk_reviews != 1 else ''} contained an active risk term"
        status_copy = "Open the review evidence below to inspect each match in context."
        heading = "Matched terms"
        heading_note = "Included in the weighted risk score"
        chip_class = " risk"
        entries = list(keywords.items())[:8]
    else:
        title = "Common review language"
        subtitle = "Useful context when no defect or fraud phrase is matched."
        status_class = ""
        status_icon = "0"
        status_title = "No active risk terms in this sample"
        status_copy = f"{review_count:,} unique written review{'s' if review_count != 1 else ''} checked."
        heading = "Other language found"
        heading_note = "Context only - not included in the risk score"
        chip_class = ""
        entries = list(signals.items())[:8]

    chips = "".join(
        f'<span class="signal-chip{chip_class}">{html.escape(term)}<b>{count:,}</b></span>'
        for term, count in entries
    )
    content = (
        f'<div class="signal-chips">{chips}</div>'
        if chips
        else '<div class="keyword-empty"><div><b>No additional language signals were matched.</b><br>'
             'This describes only the collected review sample.</div></div>'
    )
    st.markdown(
        f'<div class="term-panel-head"><div><div class="analysis-panel-title">{title}</div>'
        f'<div class="analysis-panel-subtitle" style="margin-bottom:0">{subtitle}</div></div>'
        f'<span class="term-status {"flagged" if keywords else "clear"}">{"RISK EVIDENCE" if keywords else "CONTEXT ONLY"}</span></div>'
        f'<div class="term-state {status_class}"><div class="term-state-icon">{status_icon}</div><div>'
        f'<b>{status_title}</b><small>{status_copy}</small></div></div>'
        f'<div class="signal-heading"><b>{heading}</b><span>{heading_note}</span></div>{content}'
        f'<div class="term-coverage"><div><b>{review_count:,}</b>unique reviews checked</div>'
        f'<div><b>{risk_reviews:,}</b>risk-term reviews</div><div><b>{duplicate_count:,}</b>duplicates excluded</div></div>',
        unsafe_allow_html=True,
    )


def highlight_review(review: dict[str, Any]) -> str:
    text = html.escape(str(review.get("content") or "No written review"))
    evidence = ((review.get("sentiment_analysis") or {}).get("risk") or {}).get("evidence") or []
    terms = sorted({str(item.get("term", "")) for item in evidence if item.get("term") and not item.get("negated")}, key=len, reverse=True)
    import re
    for term in terms:
        text = re.sub(f"(?i)({re.escape(html.escape(term))})", r"<mark>\1</mark>", text)
    return text


def inline_name_results(query: str, products: list[dict[str, Any]]) -> None:
    """Render product-name search matches within the public homepage."""
    platforms = sorted({platform_name(product) for product in products})
    if len(platforms) == 1:
        source_text = platforms[0]
    elif len(platforms) == 2:
        source_text = " and ".join(platforms)
    else:
        source_text = ", ".join(platforms[:-1]) + f", and {platforms[-1]}"

    with st.container(key="inline_search_results"):
        st.markdown(
            f'<div class="search-results-head"><div><div class="search-results-count">{len(products):,} results</div>'
            f'<h2 class="search-results-title">Results for “{html.escape(query)}”</h2>'
            f'<div class="search-results-source">Collected from {html.escape(source_text)} · choose a product to analyze</div></div></div>',
            unsafe_allow_html=True,
        )

        with st.container(key="search_result_filters"):
            platform_col, price_col, risk_col, sort_col = st.columns([1.05, 1.05, 1.2, 1.35])
            selected_platforms = platform_col.multiselect(
                "Platform", platforms, default=platforms, key="inline-platform-filter"
            )
            price_band = price_col.selectbox(
                "Price",
                ["All prices", "Under ₱500", "₱500–₱1,000", "₱1,000–₱2,500", "₱2,500+"],
                key="inline-price-filter",
            )
            selected_risks = risk_col.multiselect(
                "Risk level", ["Low", "Moderate", "High"], default=["Low", "Moderate", "High"], key="inline-risk-filter"
            )
            sort_order = sort_col.selectbox(
                "Sort",
                ["Reliability score", "Price: low to high", "Price: high to low", "Review count"],
                key="inline-sort",
            )

        visible = [
            product for product in products
            if platform_name(product) in selected_platforms
            and product_risk_level(product) in selected_risks
        ]

        def in_price_band(product: dict[str, Any]) -> bool:
            price = price_value(product.get("price"))
            if price_band == "All prices":
                return True
            if price is None:
                return False
            if price_band == "Under ₱500":
                return price < 500
            if price_band == "₱500–₱1,000":
                return 500 <= price <= 1000
            if price_band == "₱1,000–₱2,500":
                return 1000 < price <= 2500
            return price > 2500

        visible = [product for product in visible if in_price_band(product)]
        if sort_order == "Reliability score":
            visible.sort(key=product_reliability, reverse=True)
        elif sort_order == "Price: low to high":
            visible.sort(key=lambda product: price_value(product.get("price")) or float("inf"))
        elif sort_order == "Price: high to low":
            visible.sort(key=lambda product: price_value(product.get("price")) or -1, reverse=True)
        else:
            visible.sort(key=lambda product: int((product.get("sentiment_summary") or {}).get("review_count") or 0), reverse=True)

        if not visible:
            st.info("No collected products match these filters.")
            return

        page_size = 6
        total_pages = max(1, (len(visible) + page_size - 1) // page_size)
        page = min(max(0, int(st.session_state.get("search_result_page", 0))), total_pages - 1)
        st.session_state.search_result_page = page
        page_products = visible[page * page_size:(page + 1) * page_size]
        st.caption(f"Showing {page * page_size + 1}–{page * page_size + len(page_products)} of {len(visible)} matching products")

        for row_start in range(0, len(page_products), 3):
            with st.container(key=f"search_result_row_{page}_{row_start // 3}"):
                columns = st.columns(3, gap="large")
                for offset, (column, product) in enumerate(zip(columns, page_products[row_start:row_start + 3])):
                    result_index = page * page_size + row_start + offset
                    summary = product.get("sentiment_summary") or {}
                    level = product_risk_level(product)
                    reliability = product_reliability(product)
                    name = html.escape(str(product.get("name") or "Unnamed product"))
                    raw_image_url = str(product.get("img") or "").strip()
                    if raw_image_url.startswith("//"):
                        raw_image_url = "https:" + raw_image_url
                    parsed_image_url = urlsplit(raw_image_url)
                    image_url = html.escape(raw_image_url, quote=True) if parsed_image_url.scheme in {"http", "https"} else ""
                    image_markup = (
                        f'<img class="search-card-image" src="{image_url}" alt="{name}">'
                        if image_url else '<div class="search-card-placeholder" aria-label="Product image unavailable">◇</div>'
                    )
                    rating = html.escape(str(product.get("rating") or "—"))
                    raw_review_count = summary.get("review_count")
                    try:
                        sampled_reviews = max(0, int(str(raw_review_count).replace(",", "")))
                    except (TypeError, ValueError):
                        sampled_reviews = len(product.get("comments") or [])
                    with column:
                        with st.container(key=f"search_result_card_{result_index}"):
                            st.markdown(
                                f'{image_markup}<div class="search-card-body"><div class="search-card-flags">'
                                f'<span class="search-card-platform">{html.escape(platform_name(product))}</span>'
                                f'<span class="search-card-reliability">{reliability:.0f}% reliability</span></div>'
                                f'<div class="search-card-name">{name}</div>'
                                f'<div class="search-card-rating">★ {rating}<span>{sampled_reviews:,} sampled reviews</span></div>'
                                f'<div class="search-card-bottom"><span class="search-card-price">{html.escape(money(product.get("price")))}</span>'
                                f'<span class="search-card-risk" style="color:{RISK_COLORS[level]}">{level} risk</span></div></div>',
                                unsafe_allow_html=True,
                            )
                            key_hash = hashlib.sha256(
                                f"{result_index}|{product.get('link') or ''}".encode("utf-8")
                            ).hexdigest()[:12]
                            if st.button("Analyze reviews →", key=f"inline-view-{key_hash}", width="stretch"):
                                st.session_state.selected_link = product.get("link")
                                st.session_state.inline_result_link = product.get("link")
                                st.rerun()

        previous_col, page_col, next_col = st.columns([1, 1.4, 1])
        if previous_col.button("← Previous", disabled=page == 0, key="inline-results-previous", width="stretch"):
            st.session_state.search_result_page = page - 1
            st.rerun()
        page_col.markdown(f'<div class="search-pagination">Page {page + 1} of {total_pages}</div>', unsafe_allow_html=True)
        if next_col.button("Next →", disabled=page >= total_pages - 1, key="inline-results-next", width="stretch"):
            st.session_state.search_result_page = page + 1
            st.rerun()


def inline_product_analysis(product: dict[str, Any], products: list[dict[str, Any]]) -> None:
    """Render a completed search report directly within the public homepage."""
    summary = product.get("sentiment_summary") or {}
    score = product_risk_percent(product)
    level = product_risk_level(product)
    level_class = level.casefold()
    review_count = int(summary.get("review_count") or len(product.get("comments") or []))
    name = html.escape(str(product.get("name") or "Unnamed product"))
    description = html.escape(str(product.get("description") or "No product description was collected."))
    image_url = html.escape(product_image_url(product), quote=True)
    listing_url = html.escape(str(product.get("link") or "#"), quote=True)
    rating = html.escape(str(product.get("rating") or "—"))
    alert_text = {
        "High": "Not recommended — strong counterfeit or defect indicators detected",
        "Moderate": "Review carefully — moderate risk indicators detected",
        "Low": "Lower risk — fewer defect indicators detected in this sample",
    }[level]
    risk_color = RISK_COLORS[level]

    with st.container(key="inline_analysis"):
        st.markdown(
            f'<div class="analysis-overline">{len(products):,} products in this result set</div>'
            f'<h2 class="analysis-heading">Analysis for “{name}”</h2>',
            unsafe_allow_html=True,
        )
        image_markup = (
            f'<img class="analysis-product-image" src="{image_url}" alt="{name}" loading="lazy" referrerpolicy="no-referrer">'
            if image_url
            else '<div class="analysis-product-image analysis-product-placeholder" role="img" aria-label="Listing image unavailable">'
            '<svg viewBox="0 0 48 48" fill="none" aria-hidden="true"><path d="M14 16h20l-2 23H16l-2-23Z" '
            'stroke="currentColor" stroke-width="2.5" stroke-linejoin="round"/><path d="M19 18v-5a5 5 0 0 1 10 0v5" '
            'stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/><path d="m19 31 4-4 3 3 3-3 4 4" '
            'stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            '<span>Listing image unavailable</span></div>'
        )
        st.markdown(
            f"""<div class="analysis-product"><div class="analysis-product-grid">
            {image_markup}
            <div><span class="platform-pill">{platform_name(product)}</span><h3>{name}</h3>
            <div class="analysis-meta"><b><span class="rating-star" aria-hidden="true">&#9733;</span>{rating} star rating</b> &nbsp;·&nbsp; {review_count:,} unique reviews analyzed</div>
            <div class="analysis-description">{description}</div><div class="analysis-price">{html.escape(money(product.get('price')))}</div></div>
            <div><a class="learn-pill" href="{listing_url}" target="_blank" rel="noopener">Original listing &#8599;</a></div>
            </div></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="analysis-alert {level_class}"><span class="analysis-alert-icon">!</span><span>{html.escape(alert_text)}</span><small>Risk score {score:.0f}/100</small></div>',
            unsafe_allow_html=True,
        )

        score_col, breakdown_col = st.columns([1, 1.6], gap="large")
        with score_col:
            with st.container(key="inline_risk_panel"):
                st.markdown('<div class="analysis-panel-title">Risk score</div><div class="analysis-panel-subtitle">Calculated from the unique sampled reviews.</div>', unsafe_allow_html=True)
                risk_figure = gauge(score)
                risk_figure.update_layout(height=255, margin=dict(l=5, r=5, t=15, b=5), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(risk_figure, width="stretch", config={"displayModeBar": False}, key="inline-risk-gauge")
                st.markdown(
                    '<div class="formula-note"><span>Low 0–30<br>Moderate 31–60<br>High 61–100</span><span>Decision support only.<br>Inspect the evidence below.</span></div>',
                    unsafe_allow_html=True,
                )
        with breakdown_col:
            with st.container(key="inline_breakdown_panel"):
                render_weighted_score_breakdown(product, score, risk_color)

        sentiment_col, keyword_col = st.columns(2, gap="large")
        counts = summary.get("sentiment_counts") or {"positive": 0, "uncertain": 0, "negative": 0}
        with sentiment_col:
            with st.container(key="inline_sentiment_panel"):
                st.markdown('<div class="analysis-panel-title">Sentiment summary</div>', unsafe_allow_html=True)
                sentiment_frame = pd.DataFrame(
                    {"Sentiment": ["Positive", "Uncertain", "Negative"], "Reviews": [counts.get("positive", 0), counts.get("uncertain", counts.get("neutral", 0)), counts.get("negative", 0)]}
                )
                sentiment_chart = px.pie(
                    sentiment_frame,
                    values="Reviews",
                    names="Sentiment",
                    hole=.64,
                    color="Sentiment",
                    color_discrete_map={"Positive": GREEN, "Uncertain": "#98A2B3", "Negative": RED},
                )
                sentiment_chart.update_traces(marker=dict(line=dict(color="white", width=3)))
                sentiment_chart.add_annotation(
                    x=.5,
                    y=.5,
                    text=f"<b>{sum(int(value or 0) for value in counts.values()):,}</b><br><span style='font-size:10px'>reviews</span>",
                    showarrow=False,
                    font=dict(size=18, color=INK),
                )
                sentiment_chart.update_layout(height=330, margin=dict(l=5, r=5, t=10, b=10), legend=dict(orientation="h", y=-.05), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(sentiment_chart, width="stretch", config={"displayModeBar": False}, key="inline-sentiment")
        with keyword_col:
            with st.container(key="inline_terms_panel"):
                render_term_or_signal_panel(product)

        reviews = [item for item in product.get("comments") or [] if isinstance(item, dict) and not item.get("is_duplicate")][:6]
        st.markdown('<div class="analysis-overline" style="margin-top:2.5rem">Product reviews</div>', unsafe_allow_html=True)
        if reviews:
            review_cards = []
            for review in reviews:
                analysis = review.get("sentiment_analysis") or {}
                sentiment = str(analysis.get("label") or "uncertain").casefold()
                author = html.escape(str(review.get("author") or "Verified shopper"))
                initial = html.escape(author[:1].upper() or "S")
                stars = "★" * max(0, min(5, int(review.get("rating") or 0)))
                review_cards.append(
                    f'<article class="inline-review"><div class="inline-review-head"><span class="review-avatar">{initial}</span><div><div class="review-author">{author}</div><div class="review-stars">{stars}</div></div></div><div class="review-text">{highlight_review(review)}</div><span class="review-sentiment {sentiment}">{html.escape(sentiment)}</span></article>'
                )
            st.markdown(f'<div class="review-grid">{"".join(review_cards)}</div>', unsafe_allow_html=True)
        else:
            st.info("No written reviews were collected for this product.")

        render_recommendations(product, products, limit=3, key_prefix="inline")


def product_page(products: list[dict[str, Any]]) -> None:
    product = selected_product(products)
    if not product:
        st.info("Choose a product from Search results first.")
        return
    summary = product.get("sentiment_summary") or {}
    score = product_risk_percent(product)
    level = product_risk_level(product)
    top_left, details, gauge_col = st.columns([1.1, 2.3, 1.4])
    with top_left:
        image = product_image_url(product)
        if image:
            st.image(image, use_container_width=True)
    with details:
        st.markdown(f'<span class="platform-pill">{platform_name(product)}</span>', unsafe_allow_html=True)
        st.header(product.get("name") or "Unnamed product")
        st.markdown(f"## {money(product.get('price'))}")
        st.caption(f"★ {product.get('rating') or '—'} · Ships from {product.get('location') or 'Unknown'} · {summary.get('review_count', 0)} unique reviews analyzed")
        if product.get("link"): st.link_button("View original listing ↗", product["link"])
    with gauge_col:
        figure = gauge(score)
        figure.update_layout(height=245, margin=dict(l=15, r=15, t=35, b=5), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
        st.caption(f"Reliability score · {product_reliability(product):.0f}/100")
    if level == "High":
        st.markdown('<div class="alert-high"><strong>High-risk review pattern detected</strong><br>This product shows strong counterfeit or defect indicators. Review the evidence and compare safer alternatives below.</div>', unsafe_allow_html=True)
    st.write("")
    st.subheader("Why this score?")
    with st.container(key="product_breakdown_panel"):
        render_weighted_score_breakdown(product, score, RISK_COLORS[level])
    with st.expander("How is this calculated?"):
        st.write("A trained marketplace-review classifier labels each unique review as positive, negative, or uncertain and separately detects whether it reports product risk. VADER and the reviewable lexicons provide supporting signals, but do not override the trained decisions. The product risk score is 30% of the negative-review ratio plus 70% of the detected-risk ratio. Each review counts at most once in the risk ratio, severity is not used, and duplicate reviews are excluded. This is decision support, not proof that a listing is counterfeit.")
    counts = summary.get("sentiment_counts") or {"positive": 0, "uncertain": 0, "negative": 0}
    chart1, chart2 = st.columns([1, 1.5])
    with chart1:
        st.subheader("Sentiment summary")
        frame = pd.DataFrame({"Sentiment": ["Positive", "Uncertain", "Negative"], "Reviews": [counts.get("positive", 0), counts.get("uncertain", counts.get("neutral", 0)), counts.get("negative", 0)]})
        fig = px.pie(frame, values="Reviews", names="Sentiment", hole=.68, color="Sentiment", color_discrete_map={"Positive": GREEN, "Uncertain": "#98A2B3", "Negative": RED})
        fig.update_layout(height=290, margin=dict(l=5, r=5, t=5, b=5), legend=dict(orientation="h", y=-.08), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with chart2:
        with st.container(key="product_terms_panel"):
            render_term_or_signal_panel(product)
    st.subheader("Review evidence")
    sentiment_filter = st.segmented_control("Sentiment", ["All", "Positive", "Uncertain", "Negative", "Flagged"], default="All")
    reviews = [
        item
        for item in product.get("comments") or []
        if not item.get("is_duplicate") and item.get("included_in_analysis") is not False
    ]
    if sentiment_filter != "All":
        reviews = [r for r in reviews if (((r.get("sentiment_analysis") or {}).get("risk") or {}).get("detected") if sentiment_filter == "Flagged" else (r.get("sentiment_analysis") or {}).get("label") == sentiment_filter.lower())]
    for index, review in enumerate(reviews[:30]):
        analysis = review.get("sentiment_analysis") or {}
        label = analysis.get("label", "uncertain").title()
        color = {"Positive": GREEN, "Uncertain": "#667085", "Negative": RED}.get(label, "#667085")
        stars = "★" * int(review.get("rating") or 0)
        st.markdown(f'<div class="review"><div class="review-top"><span><b>{html.escape(review.get("author") or "Verified shopper")}</b> · {stars}</span><span style="color:{color};font-weight:700">{label}</span></div>{highlight_review(review)}</div>', unsafe_allow_html=True)
    render_recommendations(product, products, limit=4, key_prefix="product-page")


def database_page(products: list[dict[str, Any]]) -> None:
    st.markdown('<div class="eyebrow">Administration</div>', unsafe_allow_html=True)
    st.title("Product database")
    counts = database_counts()
    st.caption(f"Central SQLite store · {counts['products']:,} products · {counts['reviews']:,} reviews")
    rows = legacy_product_rows(products)
    if not rows:
        st.info("No products have been collected.")
        return
    frame = pd.DataFrame(rows).drop(columns=["Image"], errors="ignore")
    st.dataframe(frame, use_container_width=True, hide_index=True, column_config={"Link": st.column_config.LinkColumn("Listing"), "Risk score": st.column_config.NumberColumn(format="%.2f"), "Reliability score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%")})
    st.download_button("Export product CSV", csv_bytes(rows), "defaketive_products.csv", "text/csv")


def overview_page(products: list[dict[str, Any]]) -> None:
    back_to_home_button("dashboard-back-home")
    st.markdown('<div class="eyebrow">Administration</div>', unsafe_allow_html=True)
    st.title("System overview")
    reviews = review_rows(products)
    levels = [product_risk_level(product) for product in products]
    high = levels.count("High")
    cols = st.columns(5)
    values = [("Products analyzed", str(len(products))), ("Reviews scraped", f"{len(reviews):,}"), ("High-risk flags", str(high)), ("Model F1-score", "—"), ("Avg. SUS score", "—")]
    for col, (label, value) in zip(cols, values):
        with col: metric_card(label, value, "Live local dataset" if value != "—" else "Awaiting evaluation data")
    left, right = st.columns([1, 1.5])
    with left:
        st.subheader("Risk distribution")
        distribution = pd.DataFrame({"Risk": ["Low", "Moderate", "High"], "Products": [levels.count("Low"), levels.count("Moderate"), high]})
        fig = px.pie(distribution, values="Products", names="Risk", hole=.62, color="Risk", color_discrete_map=RISK_COLORS)
        fig.update_layout(height=330, margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.subheader("Recent high-risk flags")
        flagged = [row for row in legacy_product_rows(products) if row["Risk level"] == "High"]
        st.dataframe(pd.DataFrame(flagged)[["Product", "Risk score", "Reviews", "Link"]] if flagged else pd.DataFrame(columns=["Product", "Risk score", "Reviews", "Link"]), use_container_width=True, hide_index=True)


def scraper_page() -> None:
    st.markdown('<div class="eyebrow">Administration · Objective 1</div>', unsafe_allow_html=True)
    st.title("Scraper manager")
    st.caption("Create Shopee, Lazada, or Temu Philippines collection jobs and inspect locally saved runs.")
    st.warning("Marketplace sites may block automated collection. Never retry a CAPTCHA or verification failure repeatedly. Wait for the marketplace cooldown or import an authorized dataset instead.")
    with st.expander("＋ New scrape job", expanded=True):
        with st.form("admin-scrape"):
            c0, c1, c2, c3 = st.columns([1.2, 3, 1, 1])
            platform = c0.selectbox("Platform", ["shopee", "lazada", "temu"], format_func=lambda value: {"shopee": "Shopee PH", "lazada": "Lazada PH (beta)", "temu": "Temu PH (beta)"}[value])
            keyword = c1.text_input("Target product or keyword")
            count = c2.number_input("Products", 1, 20, 5)
            reviews = c3.number_input("Reviews each", 5, 100, 30, step=5)
            start = st.form_submit_button("Run scraper", type="primary")
        if start and keyword:
            products = run_live_scrape(keyword, count, reviews, platform=platform)
            if products: st.session_state.uploaded_products = products
    jobs = [{"File": path.name, "Status": "Completed", "Updated": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"), "Size": f"{path.stat().st_size / 1024:.1f} KB"} for path in result_files()]
    st.subheader("Scraping jobs")
    st.dataframe(pd.DataFrame(jobs), use_container_width=True, hide_index=True)
    st.subheader("Live log console")
    logs = sorted((ROOT / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True) if (ROOT / "logs").exists() else []
    st.code(logs[0].read_text(encoding="utf-8", errors="replace")[-6000:] if logs else "No scraper logs yet.", language="text")


def lexicon_page() -> None:
    st.markdown('<div class="eyebrow">Administration · Objective 2</div>', unsafe_allow_html=True)
    st.title("Lexicon dictionary")
    sentiment = ROOT / "sentiment-analysis" / "defaketive_sentiment" / "lexicons" / "sentiment_lexicon.tsv"
    risk = sentiment.with_name("risk_lexicon.tsv")
    tab1, tab2 = st.tabs(["Sentiment terms", "High-risk terms"])
    with tab1:
        frame = pd.read_csv(sentiment, sep="\t", comment="#", names=["Term", "Weight", "Language", "Category"])
        st.dataframe(frame, use_container_width=True, hide_index=True)
    with tab2:
        frame = pd.read_csv(risk, sep="\t", comment="#", names=["Term", "Category", "Severity", "Language"])
        st.dataframe(frame, use_container_width=True, hide_index=True)
    st.caption("The lexicons are reviewable source files. Editing is intentionally disabled in the dashboard until role-based authentication and an audit trail are added.")


def model_page() -> None:
    st.markdown('<div class="eyebrow">Administration · Objective 5</div>', unsafe_allow_html=True)
    st.title("Model evaluation")
    st.info("Upload a human-verified dataset to calculate production evaluation metrics. Placeholder values are never presented as measured results.")
    uploaded = st.file_uploader(
        "Upload verified test dataset", type=["csv"], key="evaluation",
        help="Include actual_label plus either predicted_label or review_text.",
    )
    if uploaded:
        try:
            evaluation_frame = pd.read_csv(uploaded)
            actual_column = next((name for name in ("actual_label", "true_label", "label") if name in evaluation_frame), None)
            predicted_column = next((name for name in ("predicted_label", "prediction") if name in evaluation_frame), None)
            text_column = next((name for name in ("review_text", "review", "text", "content") if name in evaluation_frame), None)
            if not actual_column or not (predicted_column or text_column):
                st.error("CSV needs actual_label and either predicted_label or review_text columns.")
            elif st.button("Run verified evaluation", type="primary"):
                if predicted_column:
                    predictions = evaluation_frame[predicted_column].astype(str).tolist()
                else:
                    sentiment_root = ROOT / "sentiment-analysis"
                    if str(sentiment_root) not in sys.path:
                        sys.path.insert(0, str(sentiment_root))
                    from defaketive_sentiment.manuscript import HybridManuscriptRiskModel
                    evaluator = HybridManuscriptRiskModel(allow_unqualified=True)
                    predictions = [evaluator.analyze(text)["label"] for text in evaluation_frame[text_column].fillna("")]
                result = evaluate_labels(evaluation_frame[actual_column].astype(str), predictions)
                save_evaluation_run(uploaded.name, result)
                st.session_state.evaluation_result = result
                st.success(f"Evaluated {result['sample_count']:,} human-labeled reviews and saved the run.")
        except Exception as exc:
            st.error(f"Evaluation could not run: {exc}")
    result = st.session_state.get("evaluation_result")
    a, b, c, d = st.columns(4)
    metric_values = [result.get(key) if result else None for key in ("accuracy", "precision", "recall", "f1")]
    for col, label, value in zip([a, b, c, d], ["Accuracy", "Precision", "Recall", "F1-score"], metric_values):
        with col: metric_card(label, f"{value:.1%}" if value is not None else "—", f"n = {result['sample_count']}" if result else "No verified run")
    st.subheader("Confusion matrix")
    labels = result["labels"] if result else ["negative", "neutral", "positive"]
    values = result["matrix"] if result else [[0] * len(labels) for _ in labels]
    matrix = go.Figure(data=go.Heatmap(z=values, x=[f"Predicted {label}" for label in labels], y=[f"Actual {label}" for label in labels], text=values, texttemplate="%{text}", colorscale=[[0, "#F2F4F7"], [1, INDIGO]], showscale=False))
    matrix.update_layout(height=330, margin=dict(l=20, r=20, t=10, b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(matrix, use_container_width=True, config={"displayModeBar": False})
    if uploaded and not result: st.caption(f"Ready to evaluate: {uploaded.name}")


def survey_page() -> None:
    back_to_home_button("survey-back-home")
    st.markdown('<div class="eyebrow">Customer acceptance</div>', unsafe_allow_html=True)
    st.title("SUS / UMUX feedback")
    st.caption("SUS uses a 1–5 scale; UMUX uses its standard 1–7 scale. Scores are calculated and stored without collecting identity data.")
    sus = [
        ("I think that I would like to use this system frequently.", "Sa tingin ko, gusto kong gamitin nang madalas ang sistemang ito."),
        ("I found the system unnecessarily complex.", "Nakita kong hindi kailangang komplikado ang sistema."),
        ("I thought the system was easy to use.", "Madaling gamitin ang sistema para sa akin."),
        ("I think I would need technical support to use this system.", "Kakailanganin ko ng teknikal na tulong upang magamit ito."),
        ("I found the functions well integrated.", "Maayos na pinagsama-sama ang mga function."),
        ("I thought there was too much inconsistency.", "Masyadong hindi pare-pareho ang sistema."),
        ("Most people would learn to use this very quickly.", "Mabilis matututunan ng karamihan ang sistemang ito."),
        ("I found the system cumbersome to use.", "Mabigat o mahirap gamitin ang sistema."),
        ("I felt very confident using the system.", "Kampante akong gamitin ang sistema."),
        ("I needed to learn a lot before using the system.", "Marami muna akong kailangang matutunan bago ito gamitin."),
    ]
    umux = [
        ("This system's capabilities meet my requirements.", "Natutugunan ng kakayahan ng sistema ang aking mga pangangailangan."),
        ("Using this system is a frustrating experience.", "Nakakainis o nakabibigo ang paggamit ng sistemang ito."),
        ("This system is easy to use.", "Madaling gamitin ang sistemang ito."),
        ("I have to spend too much time correcting things with this system.", "Masyado akong gumugugol ng oras sa pagwawasto habang ginagamit ito."),
    ]
    with st.form("survey"):
        frequency = st.selectbox("How often do you shop online?", ["Weekly", "Monthly", "A few times a year", "Rarely"])
        st.markdown("#### System Usability Scale (1–5)")
        sus_responses = []
        for i, (english, tagalog) in enumerate(sus, 1):
            sus_responses.append(st.radio(f"{i}. {english}\n\n_{tagalog}_", [1, 2, 3, 4, 5], horizontal=True, key=f"sus-{i}", index=2))
        st.markdown("#### Usability Metric for User Experience (1–7)")
        umux_responses = []
        for i, (english, tagalog) in enumerate(umux, 1):
            umux_responses.append(st.radio(f"{i}. {english}\n\n_{tagalog}_", [1, 2, 3, 4, 5, 6, 7], horizontal=True, key=f"umux-{i}", index=3))
        sent = st.form_submit_button("Submit feedback", type="primary")
    if sent:
        sus_result = score_sus(sus_responses)
        umux_result = score_umux(umux_responses)
        save_survey_response(frequency, sus_responses, umux_responses, sus_result, umux_result)
        st.success(f"Thank you. Your anonymous response was saved · SUS {sus_result:.1f}/100 · UMUX {umux_result:.1f}/100")


def _sync_page_from_sidebar() -> None:
    selected = st.session_state.get("navigation")
    if selected in NAVIGATION_PAGES:
        st.session_state.page = selected


def sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="brand">DeFaketive<span class="brand-dot">.</span></div>', unsafe_allow_html=True)
        st.caption("REVIEW RISK INTELLIGENCE")
        current = st.session_state.get("page", "Search")
        options = list(NAVIGATION_PAGES)
        pending_target = st.session_state.pop("_navigation_target", None)
        if pending_target in NAVIGATION_PAGES:
            st.session_state.navigation = pending_target
        elif "navigation" not in st.session_state or st.session_state.navigation not in NAVIGATION_PAGES:
            st.session_state.navigation = current if current in NAVIGATION_PAGES else "Search"
        page = st.selectbox(
            "Navigate",
            options,
            key="navigation",
            on_change=_sync_page_from_sidebar,
        )
        st.session_state.page = page
        st.divider()
        uploaded = st.file_uploader("Load scraper JSON", type=["json"], help="Load an existing Shopee scraper result without running Chrome.")
        if uploaded:
            try:
                st.session_state.uploaded_products = analyze_products(load_products(uploaded.getvalue()))
                for key in ("search_mode", "search_result_query", "search_result_page", "inline_result_link", "selected_link"):
                    st.session_state.pop(key, None)
                save_products(st.session_state.uploaded_products, source=uploaded.name)
                st.success("Dataset loaded")
            except Exception as exc:
                st.error(f"Could not load file: {exc}")
        if st.session_state.get("uploaded_products") is not None and st.button("Use latest saved results"):
            st.session_state.pop("uploaded_products", None)
            for key in ("search_mode", "search_result_query", "search_result_page", "inline_result_link", "selected_link"):
                st.session_state.pop(key, None)
            st.rerun()
        st.divider()
        st.caption("Shopee PH · Lazada beta · Temu beta\n\nExplainable Taglish sentiment model")
    return page


inject_theme()
page = sidebar()
products = get_products()

if page == "Search": landing_page(products)
elif page == "Search results": results_page(products)
elif page == "Product analysis": product_page(products)
elif page == "Feedback survey": survey_page()
elif page == "Admin overview": overview_page(products)
elif page == "Scraper manager": scraper_page()
elif page == "Lexicon manager": lexicon_page()
elif page == "Product database": database_page(products)
elif page == "Model evaluation": model_page()
