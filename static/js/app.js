const state = { homeScan: null, catalogLoading: true, currentProduct: null, reportFrom: "history", activity: {}, wishlist: [], catalogError: false, modelUnavailable: "", products: [], catalogProducts: [], toastTimer: null, activeJobId: null, jobStartedAt: null, jobElapsedTimer: null, searchBusy: false };
const MARKETPLACES = { shopee: "Shopee PH", lazada: "Lazada PH", temu: "Temu PH" };
// Lazada scans read a fixed number of reviews from each star level (webapp.LAZADA_REVIEWS_PER_STAR).
const LAZADA_REVIEWS_PER_STAR = 60;
const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
const money = (value) => String(value || "\u2014").replace(/(?:PHP|Php|php)\s*|\u00e2\u201a\u00b1/g, "\u20b1");
// Badges replace the old 0-100 risk score. The server words every verdict and
// badge (dashboard_utils.badge_view) and tests that wording; the browser only
// lays it out. A product without a view was analysed by an older model and is
// shown as not analysed -- never as low risk, which is what reading a missing
// score as zero used to do.
const NO_VIEW = {available: false, state: "not_analyzed", tone: "unknown", headline: "Not analyzed yet", detail: "Open the report to check it with the current review model.", reviews_analyzed: 0, counts: {safety: 0, counterfeit: 0, defect: 0}, badges: []};
const view = (product) => product?.badge_view || NO_VIEW;
const tone = (product) => view(product).tone || "unknown";
const counts = (product) => ({safety: 0, counterfeit: 0, defect: 0, ...(view(product).counts || {})});
const reviewCount = (product) => Number(view(product).reviews_analyzed || product?.comments?.length || 0) || 0;
// Most concerning first. A safety report outranks fake and defect reports
// even when there are too few reviews for a verdict: one fire report on a
// four-review listing matters more than several "peke" reports.
function concernRank(product) { const t = tone(product); if (t === "high") return 0; if (counts(product).safety > 0) return 1; return {moderate: 2, unknown: 3, low: 4}[t] ?? 3; }
function byConcern(a, b) { const ca = counts(a), cb = counts(b); return concernRank(a) - concernRank(b) || cb.safety - ca.safety || (cb.counterfeit + cb.defect) - (ca.counterfeit + ca.defect); }
// A single safety report is worth a look even when there are too few reviews for a verdict.
function needsAttention(product) { return ["high", "moderate"].includes(tone(product)) || counts(product).safety > 0; }
function matchesConcern(product, filter) { if (filter === "attention") return needsAttention(product); if (filter === "safety") return counts(product).safety > 0; if (filter === "clear") return tone(product) === "low"; if (filter === "unknown") return tone(product) === "unknown"; return true; }
function verdictBadge(product) { return `<span class="risk-badge ${esc(tone(product))}">${esc(view(product).headline)}</span>`; }
function reportChips(product) { const c = counts(product); const chips = [["safety", "safety"], ["counterfeit", "fake"], ["defect", "defect"]].filter(([key]) => c[key]).map(([key, word]) => `<span class="report-chip ${key}">${c[key]} ${word} report${c[key] === 1 ? "" : "s"}</span>`); return chips.length ? `<div class="report-chips">${chips.join("")}</div>` : ""; }
// Wrap the matched words of a quote, using the character ranges the server sent.
function highlightSpans(text, spans) { const source = String(text || ""); const merged = []; (spans || []).map(span => [Math.max(0, Number(span?.[0]) || 0), Math.min(source.length, Number(span?.[1]) || 0)]).filter(([start, end]) => end > start).sort((a, b) => a[0] - b[0]).forEach(([start, end]) => { const last = merged[merged.length - 1]; if (last && start <= last[1]) last[1] = Math.max(last[1], end); else merged.push([start, end]); }); let html = "", cursor = 0; merged.forEach(([start, end]) => { html += esc(source.slice(cursor, start)) + "<mark>" + esc(source.slice(start, end)) + "</mark>"; cursor = end; }); return html + esc(source.slice(cursor)); }
function safeImage(product) { const candidate = product?.img || product?.image || product?.image_url || product?.thumbnail; return /^(https?:\/\/|\/static\/)/i.test(String(candidate || "")) ? candidate : ""; }
function safeLink(value) { return /^https?:\/\//i.test(String(value || "")) ? String(value) : ""; }
function toast(message, kind = "info") { const node = $("#toast"); node.textContent = message; node.dataset.kind = kind; node.hidden = false; clearTimeout(state.toastTimer); state.toastTimer = setTimeout(() => { node.hidden = true; }, 5000); }
async function api(url, options = {}) { const response = await fetch(url, options); const body = await response.json().catch(() => ({})); if (!response.ok) { const error = new Error(body.error || "Something went wrong."); error.body = body; throw error; } return body; }
function queryContext(value) { const text = String(value || "").trim(); let parsed; try { parsed = new URL(text); } catch (_) { return { isUrl: false, platform: null }; } if (!/^https?:$/.test(parsed.protocol)) return { isUrl: false, platform: null }; const host = parsed.hostname.toLowerCase(); if (host === "shopee.ph" || host.endsWith(".shopee.ph")) return { isUrl: true, platform: "shopee" }; if (host === "lazada.com.ph" || host.endsWith(".lazada.com.ph")) return { isUrl: true, platform: "lazada" }; if (host === "temu.com" || host.endsWith(".temu.com")) return { isUrl: true, platform: "temu" }; return { isUrl: true, platform: null }; }
function clearSearchFeedback() { const feedback = $("#search-feedback"); feedback.hidden = true; feedback.removeAttribute("data-kind"); }
function showSearchFeedback(message, title = "Scan could not start", kind = "error") { $("#search-feedback-title").textContent = title; $("#search-feedback-message").textContent = message; const feedback = $("#search-feedback"); feedback.dataset.kind = kind; feedback.hidden = false; }
function selectedMarketplace() { return document.querySelector('input[name="marketplace"]:checked')?.value || "shopee"; }
function selectMarketplace(value) { const input = document.querySelector(`input[name="marketplace"][value="${value}"]`); if (input) input.checked = true; }
function productLinkIdentity(value) { try { const url = new URL(String(value || "").trim()); return `${url.hostname.toLowerCase().replace(/^www\./, "")}${url.pathname.replace(/\/+$/, "").toLowerCase()}`; } catch (_) { return ""; } }
function savedProductForQuery(value) { const identity = productLinkIdentity(value); if (!identity) return null; const catalog = state.catalogProducts.length ? state.catalogProducts : state.products; return catalog.find(product => productLinkIdentity(product.link) === identity) || null; }
function setSearchBusy(busy) {
  state.searchBusy = busy;
  updateBell();
  ["#search-query", "#product-count", "#review-count"].forEach(selector => { const node = $(selector); if (node) node.disabled = busy; });
  document.querySelectorAll('input[name="marketplace"]').forEach(input => { input.disabled = busy; });
  updateSearchUI();
}
function updateSearchUI(clearFeedback = false) {
  if (clearFeedback) clearSearchFeedback();
  const query = $("#search-query").value.trim();
  const context = queryContext(query);
  const hasQuery = Boolean(query);
  const unsupported = context.isUrl && !context.platform;
  const productCount = $("#product-count");
  const productField = $("#product-count-field");
  const marketplaceField = $("#marketplace-field");
  const advanced = $("#advanced-options");
  const detection = $("#marketplace-detection");
  const savedMatch = $("#saved-match");
  const primary = $("#search-submit");
  const fresh = $("#fresh-search");
  const summary = $("#scan-summary");
  const match = context.platform ? (savedProductForQuery(query) || workspaceProducts().find(product => productLinkIdentity(product.link) === productLinkIdentity(query))) : null;

  if (context.platform) {
    if (!productCount.dataset.linkMode) {
      productCount.dataset.previousValue = productCount.value;
      productCount.dataset.linkMode = "true";
    }
    selectMarketplace(context.platform);
    productCount.value = "1";
  } else if (productCount.dataset.linkMode) {
    productCount.value = productCount.dataset.previousValue || "6";
    delete productCount.dataset.previousValue;
    delete productCount.dataset.linkMode;
  }

  marketplaceField.hidden = !hasQuery || context.isUrl;
  advanced.hidden = !hasQuery || unsupported;
  productField.hidden = Boolean(context.platform);
  detection.hidden = !context.isUrl;
  detection.dataset.kind = unsupported ? "error" : "success";
  detection.textContent = unsupported
    ? "This website is not supported. Paste a Shopee PH, Lazada PH, or Temu product link."
    : context.platform ? `✓ ${MARKETPLACES[context.platform]} product link detected.` : "";

  savedMatch.hidden = !match;
  savedMatch.textContent = match ? `Saved report found: ${match.name || MARKETPLACES[context.platform] + " product"}` : "";

  if (context.platform && match) {
    primary.dataset.searchIntent = "saved";
    primary.innerHTML = `View saved report <span aria-hidden="true">→</span>`;
    fresh.textContent = "Refresh with new reviews";
    fresh.hidden = false;
  } else if (context.platform) {
    primary.dataset.searchIntent = "live";
    primary.innerHTML = `Analyze this ${MARKETPLACES[context.platform]} product <span aria-hidden="true">→</span>`;
    fresh.hidden = true;
  } else {
    primary.dataset.searchIntent = "saved";
    primary.innerHTML = `Search analyzed products <span aria-hidden="true">→</span>`;
    fresh.textContent = "Collect fresh reviews";
    fresh.hidden = !hasQuery || unsupported;
  }

  primary.disabled = state.searchBusy || !hasQuery || unsupported;
  fresh.disabled = state.searchBusy || !hasQuery || unsupported;
  productCount.disabled = state.searchBusy || Boolean(context.platform);
  const lazadaScan = (context.platform || selectedMarketplace()) === "lazada";
  $("#review-count").disabled = state.searchBusy || lazadaScan;
  const reviewCountField = $("#review-count").closest("label");
  if (reviewCountField) reviewCountField.hidden = lazadaScan;
  document.querySelectorAll('input[name="marketplace"]').forEach(input => { input.disabled = state.searchBusy; });

  if (!hasQuery || unsupported) {
    summary.hidden = true;
  } else {
    const marketplace = context.platform || selectedMarketplace();
    const products = context.platform ? 1 : Number(productCount.value || 1);
    const reviews = Number($("#review-count").value || 1);
    const reviewPlan = lazadaScan ? `up to ${LAZADA_REVIEWS_PER_STAR} reviews from each star level` : `up to ${reviews} reviews each`;
    summary.textContent = `Fresh collection settings: ${MARKETPLACES[marketplace]} · up to ${products} product${products === 1 ? "" : "s"} · ${reviewPlan}`;
    summary.hidden = false;
  }
  $("#search-note").textContent = state.searchBusy
    ? "You can browse your workspace while this scan runs. Use the progress button above to return."
    : "Results are based on a sample of public reviews. Collection time varies by marketplace.";

}
function setJobStage(stage, reportedProgress) { const fallback = { 1: 18, 2: 68, 3: 92 }[stage] || 18; const progress = Number.isFinite(Number(reportedProgress)) ? Math.min(100, Math.max(0, Number(reportedProgress))) : fallback; $("#job-progress").style.width = `${progress}%`; document.querySelectorAll("[data-job-stage]").forEach(item => { const itemStage = Number(item.dataset.jobStage); item.classList.toggle("complete", itemStage < stage); item.classList.toggle("active", itemStage === stage); }); }
function startJobClock(createdAt) { clearInterval(state.jobElapsedTimer); const parsed = Date.parse(createdAt || ""); state.jobStartedAt = Number.isNaN(parsed) ? Date.now() : parsed; const update = () => { const seconds = Math.max(0, Math.floor((Date.now() - state.jobStartedAt) / 1000)); const elapsed = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`; $("#job-elapsed").textContent = elapsed; $("#home-scan-elapsed").textContent = elapsed + " elapsed"; }; update(); state.jobElapsedTimer = setInterval(update, 1000); }
function finishJob() { clearInterval(state.jobElapsedTimer); state.jobElapsedTimer = null; state.activeJobId = null; try { sessionStorage.removeItem("defaketive-active-job"); } catch (_) {} setSearchBusy(false); }
function closeMainNavigation() {
  const nav = $("#main-nav");
  const toggle = $(".nav-toggle");
  if (!nav || !toggle) return;
  nav.classList.remove("open");
  toggle.setAttribute("aria-expanded", "false");
  toggle.setAttribute("aria-label", "Open navigation");
}

function setActiveNavigation(view, section = "") {
  const links = document.querySelectorAll("#main-nav .nav-link");
  links.forEach(link => {
    link.classList.remove("active");
    link.removeAttribute("aria-current");
  });
  if (view !== "landing") return;
  const active = section
    ? document.querySelector(`#main-nav .nav-link[data-scroll="${section}"]`)
    : document.querySelector('#main-nav .nav-link[data-route="landing"]');
  if (active) {
    active.classList.add("active");
    active.setAttribute("aria-current", section ? "location" : "page");
  }
}

function currentView() {
  return document.querySelector("[data-view]:not([hidden])")?.dataset.view || "landing";
}

function updateHeaderState() {
  const header = $(".pub-header");
  if (!header) return;
  header.classList.toggle("scrolled", window.scrollY > 8);
  if (currentView() !== "landing") return;
  const threshold = (header.offsetHeight || 72) + 48;
  const activeSection = ["faq", "about"].filter(id => {
    const section = $("#" + id);
    return section && section.getBoundingClientRect().top <= threshold;
  }).at(-1) || "";
  setActiveNavigation("landing", activeSection);
}

function focusProductSearch() {
  requestAnimationFrame(() => {
    const input = $("#search-query");
    if (!input) return;
    input.focus({preventScroll: true});
    input.scrollIntoView({behavior: "smooth", block: "center"});
  });
}

const WORKSPACE_ROUTES = ["home", "analyze", "history", "wishlist", "settings", "results", "analysis"];
const ROUTE_LABELS = {home: "Home", analyze: "Analyze product", history: "History", wishlist: "Wishlist", settings: "Settings", results: "Search results", analysis: "Product report"};
function go(view, replace = false, preserveUrl = false) {
  const previousView = currentView();
  if (view === "dashboard") view = "home";
  if (view === "admin") view = "history";
  if (view === "login" && state.user) view = "home";
  if (view !== "landing" && view !== "login" && !WORKSPACE_ROUTES.includes(view)) view = "home";
  document.body.dataset.activeView = view;
  document.querySelectorAll("[data-view]").forEach(node => {
    const visible = node.dataset.view === view;
    node.hidden = !visible;
    node.classList.toggle("active", visible);
  });
  $("#workspace-shell").hidden = view === "landing" || view === "login";
  let hash = view;
  if (view === "analysis" && state.currentProduct?.link) {
    hash += "?" + new URLSearchParams({product: state.currentProduct.link, from: state.reportFrom});
  }
  if (!preserveUrl && window.location.hash !== "#" + hash) {
    window.history[replace ? "replaceState" : "pushState"](null, "", "#" + hash);
  }
  const activeRoute = view === "analysis" ? state.reportFrom : view === "results" ? "analyze" : view;
  document.querySelectorAll(".app-rail [data-route]").forEach(link => {
    const active = link.dataset.route === activeRoute;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  setActiveNavigation(view);
  closeMainNavigation();
  closeAccountMenu();
  if (view === "results") renderResults();
  if (view === "home" || view === "history" || view === "wishlist") renderWorkspace();
  if (view === "analysis" && !state.currentProduct) {
    $("#analysis-content").innerHTML = '<div class="flow-empty"><h1>Choose a product report</h1><p>Open an analysis from History, or analyze a product to get started.</p><a href="#history">View history →</a></div>';
  }
  document.title = view === "landing" ? "DeFaketive — Spot fakes and defects before you buy" : view === "login" ? "Sign in · DeFaketive" : ROUTE_LABELS[view] + " · DeFaketive";
  if (view === "login") prepareLoginPage();
  updateLimitNotice();
  window.scrollTo({top: 0, behavior: "instant"});
  if (previousView !== view) {
    const heading = document.querySelector(`[data-view="${view}"] h1, [data-view="${view}"] h2`);
    if (heading) { heading.setAttribute("tabindex", "-1"); heading.focus({preventScroll: true}); }
  }
}
function readWorkspaceStorage(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch (_) { return fallback; }
}
function persistWorkspace(key, value) {
  if (state.user) return;
  try { localStorage.setItem(key, JSON.stringify(value)); }
  catch (_) { toast("Browser storage is unavailable. Your changes will last for this visit only.", "warning"); }
}
function cacheWorkspaceProducts(products, opened = false) {
  products.forEach(product => {
    if (!safeLink(product?.link)) return;
    // Keep compact summaries locally; the complete review evidence stays in the app's saved reports.
    const {comments, ...summary} = product;
    state.activity[product.link] = {product: summary, openedAt: opened ? Date.now() : state.activity[product.link]?.openedAt || 0};
  });
  state.activity = Object.fromEntries(Object.entries(state.activity).sort((a, b) => b[1].openedAt - a[1].openedAt).filter(([link], index) => index < 100 || state.wishlist.includes(link)));
  persistWorkspace("defaketive-activity", state.activity);
  if (state.user && opened) products.forEach(product => {
    const entry = state.activity[product?.link];
    if (entry) saveToAccount("/api/me/activity", "PUT", {link: product.link, product: entry.product, openedAt: entry.openedAt});
  });
}
function rememberProduct(product) { cacheWorkspaceProducts([product], true); }
function workspaceProducts() {
  const products = new Map();
  Object.values(state.activity).forEach(entry => {
    if (entry?.product && safeLink(entry.product.link)) products.set(entry.product.link, entry.product);
  });
  state.catalogProducts.forEach(product => {
    if (safeLink(product.link)) products.set(product.link, product);
  });
  return [...products.values()].sort((a, b) => {
    const date = p => state.activity[p.link]?.openedAt || Date.parse(p.analyzed_at || p.analysis_date || p.updated_at || "") || 0;
    return date(b) - date(a);
  });
}
function toggleWishlist(link) {
  if (!safeLink(link)) return;
  const saved = state.wishlist.includes(link);
  state.wishlist = saved ? state.wishlist.filter(item => item !== link) : [...state.wishlist, link];
  const product = workspaceProducts().find(item => item.link === link) || state.currentProduct;
  if (!saved && product?.link === link) cacheWorkspaceProducts([product]);
  persistWorkspace("defaketive-wishlist", state.wishlist);
  if (state.user) {
    const entry = state.activity[link];
    // The account needs the product's summary before it can list it as saved.
    const ready = !saved && entry ? saveToAccount("/api/me/activity", "PUT", {link, product: entry.product, openedAt: entry.openedAt}) : Promise.resolve();
    ready.then(() => saveToAccount("/api/me/wishlist", saved ? "DELETE" : "PUT", {link}));
  }
  syncReportSave();
  renderWorkspace();
  toast(saved ? "Removed from wishlist." : state.user ? "Saved to the wishlist in your account." : "Saved to your wishlist on this browser.");
}
function reportDate(product) {
  const value = product.analysis_date || product.analyzed_at || product.updated_at;
  const date = value && !Number.isNaN(Date.parse(value)) ? new Date(value) : null;
  return date ? {iso: date.toISOString(), label: new Intl.DateTimeFormat("en-PH", {month: "short", day: "numeric", year: "numeric"}).format(date)} : null;
}
function workspaceRows(products, emptyTitle, emptyCopy) {
  if (!products.length) return '<div class="flow-empty"><h3>' + esc(emptyTitle) + '</h3><p>' + esc(emptyCopy) + '</p><a href="#analyze">Analyze a product →</a></div>';
  return '<div class="flow-report-list">' + products.map(product => {
    const saved = state.wishlist.includes(product.link), image = safeImage(product), date = reportDate(product);
    return `<article class="flow-report-row"><div class="flow-product-thumb">${image ? `<img src="${esc(image)}" alt="" loading="lazy">` : '<span aria-hidden="true">▤</span>'}</div><div class="flow-product-info"><h3 title="${esc(product.name || "Unnamed product")}">${esc(product.name || "Unnamed product")}</h3><p><span class="flow-market-label">${esc(platform(product))}</span><span>${reviewCount(product).toLocaleString()} reviews</span><span class="flow-analysis-date">${date ? `Analyzed <time datetime="${esc(date.iso)}">${esc(date.label)}</time>` : "Analysis date unavailable"}</span></p></div>${verdictBadge(product)}<div class="flow-row-actions"><button type="button" class="flow-save" data-save-link="${esc(product.link)}" aria-label="${saved ? "Remove from" : "Save to"} wishlist: ${esc(product.name)}" aria-pressed="${saved}">${saved ? "♥" : "♡"}</button><button type="button" class="flow-open" data-product-link="${esc(product.link)}">View report →</button></div></article>`;
  }).join("") + "</div>";
}
function persistHomeScan(scan) {
  state.homeScan = scan;
  try { sessionStorage.setItem("defaketive-home-scan", JSON.stringify(scan)); } catch (_) {}
  renderHomeScan();
}
function renderHomeScan() {
  updateBell();
  const scan = state.homeScan, panel = $("#home-scan-status");
  panel.hidden = !scan;
  if (!scan) return;
  const complete = scan.state === "complete", stopped = scan.state === "error", attention = scan.state === "attention";
  panel.dataset.state = scan.state;
  const stage = Math.min(3, Math.max(1, Number(scan.stage) || 1));
  $("#home-scan-platform").textContent = MARKETPLACES[scan.platform] || "Product analysis";
  $("#home-scan-title").textContent = complete ? "Your report is ready" : stopped ? "This scan couldn't finish" : attention ? "Scan progress needs checking" : ["Collecting product reviews", "Analyzing review signals", "Preparing your report"][stage - 1];
  const product = workspaceProducts().find(item => item.link === scan.productLink);
  $("#home-scan-product").textContent = product?.name || scan.query || "Your product";
  $("#home-scan-message").textContent = complete ? "Your analysis is saved. Open the report whenever you're ready." : scan.message || "You can keep browsing while we work.";
  $("#home-scan-action").textContent = complete ? (scan.productLink ? "Open report →" : "View results →") : stopped ? "View scan details →" : attention ? "Check progress →" : "View progress →";
  const progress = Number.isFinite(Number(scan.progress)) ? Math.min(100, Math.max(0, Number(scan.progress))) : {1: 18, 2: 68, 3: 92}[stage];
  $("#home-scan-progress").hidden = complete || stopped || attention;
  $("#home-scan-progress").setAttribute("aria-valuenow", String(progress));
  $("#home-scan-progress-fill").style.width = progress + "%";
  $("#home-scan-elapsed").hidden = complete || stopped;
}
async function openHomeScan() {
  const scan = state.homeScan;
  if (!scan) return;
  if (scan.state === "complete") {
    if (scan.resultsPending) {
      try {
        await loadProducts();
        const direct = queryContext(scan.query).isUrl;
        const product = direct ? state.products.find(item => productLinkIdentity(item.link) === productLinkIdentity(scan.query)) || (state.products.length === 1 ? state.products[0] : null) : null;
        persistHomeScan({...scan, productLink: product?.link || scan.productLink, resultLinks: state.products.map(item => item.link).filter(safeLink), resultsPending: false});
        return openHomeScan();
      } catch (_) { return toast("Saved results are still unavailable. Please try opening the report again shortly.", "error"); }
    }
    if (safeLink(scan.productLink)) return openProduct(scan.productLink);
    const links = new Set(scan.resultLinks || []);
    state.products = workspaceProducts().filter(product => links.has(product.link));
    go("results");
  } else {
    go("analyze");
    if (scan.state === "attention") pollJob(scan.id, scan);
    if (scan.state === "error") showSearchFeedback(scan.message || "Try another product link.", "Live scan stopped");
    $("#job-panel").scrollIntoView({behavior: "smooth", block: "center"});
  }
}
// --- Home ------------------------------------------------------------------
// A chip says what reviewers reported, never "safe" or "avoid". A single
// safety report shows as one even when there are too few reviews for a verdict.
function chipFor(product) {
  const t = tone(product), safety = counts(product).safety;
  if (t === "high") return {kind: "safety", label: "Safety reports"};
  if (safety > 0) return {kind: "safety", label: safety === 1 ? "Safety report" : "Safety reports"};
  if (t === "moderate") return {kind: "caution", label: "Caution"};
  if (t === "low") return {kind: "clear", label: "No warnings"};
  return {kind: "unknown", label: view(product).state === "not_analyzed" ? "Not analyzed" : "Too few reviews"};
}
function marketKey(product) {
  const name = String(product?.platform || "").toLowerCase();
  return ["shopee", "lazada", "temu"].find(key => name.includes(key)) || "other";
}
function smallStars(value) {
  const rating = Math.max(0, Math.min(5, Number(value) || 0));
  if (!rating) return '<span class="app-rating"><small>No rating</small></span>';
  const stars = [1, 2, 3, 4, 5].map(position => `<img src="/static/assets/app/star-${rating >= position - .5 ? "filled" : "empty"}.svg" width="8.8" height="8.47" alt="">`).join("");
  return `<span class="app-rating" role="img" aria-label="${rating.toFixed(1)} out of 5 stars"><span class="app-stars" aria-hidden="true">${stars}</span><small aria-hidden="true">${rating.toFixed(1)}/5</small></span>`;
}
// One product card, used by Home, Wishlist and the Products list.
function appCard(product, badge = "") {
  const link = safeLink(product.link), image = safeImage(product), chip = chipFor(product);
  const saved = state.wishlist.includes(link), name = product.name || "Unnamed product";
  return `<article class="app-product"><div class="app-product-image">${image ? `<img src="${esc(image)}" alt="" loading="lazy">` : '<span>No photo</span>'}<span class="app-chip ${chip.kind}">${esc(chip.label)}</span>${badge}</div><div class="app-product-body"><div class="app-product-top"><span class="app-market ${marketKey(product)}">${esc(platform(product))}</span><button type="button" class="app-heart" data-save-link="${esc(link)}" aria-pressed="${saved}" aria-label="${saved ? "Remove from" : "Save to"} wishlist: ${esc(name)}"></button></div><button type="button" class="app-product-name" data-product-link="${esc(link)}" title="${esc(name)}">${esc(name)}</button><b class="app-price">${esc(money(product.price))}</b>${smallStars(product.rating)}</div></article>`;
}
function updateRowArrow(track) {
  const next = track.parentElement?.querySelector("[data-row-next]");
  if (next) next.hidden = track.scrollWidth <= track.clientWidth + 4 || track.scrollLeft + track.clientWidth >= track.scrollWidth - 4;
}
function fillRow(key, products, emptyHtml = "") {
  const section = document.querySelector(`[data-home-row="${key}"]`), track = $("#home-" + key);
  if (!section || !track) return;
  section.hidden = !products.length && !emptyHtml;
  track.innerHTML = products.length ? products.map(product => appCard(product)).join("") : emptyHtml;
  track.setAttribute("aria-busy", "false");
  requestAnimationFrame(() => updateRowArrow(track));
}
function renderHomeBoxes(products) {
  const total = products.length;
  const groups = [["No warnings", "clear", .7], ["Caution", "caution", .45], ["Safety reports", "safety", .28], ["Too few reviews", "unknown", .14]]
    .map(([label, kind, strength]) => ({label, strength, count: products.filter(product => chipFor(product).kind === kind).length}));
  const where = state.user ? "in your account" : "on this browser";
  $("#home-overview").innerHTML = `<h2 class="app-eyebrow quiet">Overview</h2><div class="app-bars">${groups.map(group => {
    const percent = total ? Math.round(group.count / total * 100) : 0;
    return `<div><p><span>${group.label} <small>${group.count} product${group.count === 1 ? "" : "s"}</small></span><b>${percent}%</b></p><div class="app-bar" role="img" aria-label="${group.label}: ${percent}%"><i style="width:${percent}%;opacity:${group.strength}"></i></div></div>`;
  }).join("")}</div><p class="app-footnote">${total ? `Based on ${total} product${total === 1 ? "" : "s"} ${where}.` : "Check a product to see your overview."}</p>`;
  const stores = Object.entries(MARKETPLACES).map(([key, label]) => {
    const mine = products.filter(product => marketKey(product) === key);
    return {label, total: mine.length, flagged: mine.filter(needsAttention).length};
  }).filter(store => store.total);
  $("#home-stores").innerHTML = '<h2 class="app-eyebrow quiet">Warnings by marketplace</h2>' + (stores.length
    ? `<div class="app-store-rows">${stores.map(store => `<div><span>${store.label}</span><div class="app-bar" role="img" aria-label="${store.flagged} of ${store.total} products with warnings"><i style="width:${Math.round(store.flagged / store.total * 100)}%;opacity:.5"></i></div><b>${store.flagged} of ${store.total}</b><small>with warnings</small></div>`).join("")}</div><p class="app-footnote">Counted from the products you checked.</p>`
    : '<p class="app-footnote">Check a product to see how each marketplace compares.</p>');
}
function renderHomeProducts(products) {
  renderHomeScan();
  const stats = $("#home-stats");
  if (state.catalogLoading && !products.length) {
    stats.innerHTML = '<p class="app-loading" role="status">Loading your activity…</p>';
    return;
  }
  const safety = products.filter(product => chipFor(product).kind === "safety").sort(byConcern);
  const caution = products.filter(product => chipFor(product).kind === "caution").sort(byConcern);
  stats.innerHTML = [
    [products.length, "Products analyzed"],
    [products.reduce((total, product) => total + reviewCount(product), 0), "Reviews analyzed"],
    [safety.length + caution.length, "With warnings"],
    [state.wishlist.length, "In your wishlist"],
  ].map(([value, label]) => `<div><b>${value.toLocaleString()}</b><span>${label}</span></div>`).join("");
  const alert = $("#home-alert");
  alert.hidden = !(safety.length || caution.length);
  alert.dataset.kind = safety.length ? "danger" : "caution";
  alert.innerHTML = safety.length
    ? `<div><b>${safety.length} product${safety.length === 1 ? " has" : "s have"} safety reports</b><p>Reviewers described fire, smoke, shocks or overheating.</p></div><button type="button" data-show-risk="safety">View them</button>`
    : `<div><b>${caution.length} product${caution.length === 1 ? " needs" : "s need"} a closer look</b><p>Reviewers reported fakes or defects.</p></div><button type="button" data-show-risk="attention">View them</button>`;
  const emptyTitle = state.modelUnavailable ? "Review analysis is unavailable" : state.catalogError ? "Saved reports could not be loaded" : "Your first analysis starts here";
  const emptyCopy = state.modelUnavailable || (state.catalogError ? "Reload the page to try again, or start a new analysis above." : "Paste a product link above. Your completed report will appear here.");
  fillRow("recent", products.slice(0, 10), `<div class="app-empty"><h3>${esc(emptyTitle)}</h3><p>${esc(emptyCopy)}</p></div>`);
  fillRow("caution", caution.slice(0, 10));
  fillRow("safety", safety.slice(0, 10));
  renderHomeBoxes(products);
}
function showHistoryFilter(kind) {
  $("#history-search").value = "";
  $("#history-risk").value = kind;
  go("history");
}
// The bell stands for the background scan: a dot while one runs or has finished.
function updateBell() {
  const bell = $("#active-scan-link");
  if (!bell) return;
  const scan = state.homeScan;
  const label = state.searchBusy ? "A scan is running. Open it" : scan?.state === "complete" ? "Your report is ready. Open it" : scan ? "Your last scan needs a look. Open it" : "No scans running";
  bell.classList.toggle("has-dot", state.searchBusy || Boolean(scan));
  bell.setAttribute("aria-label", label);
  bell.title = label;
}
function closeAccountMenu() {
  const menu = $("#account-menu"), toggle = $("[data-account-toggle]");
  if (menu) menu.hidden = true;
  if (toggle) toggle.setAttribute("aria-expanded", "false");
}
// --- History and wishlist in the account -----------------------------------
// Guests keep their history in this browser. Signed-in users keep it in their
// account, so it follows them to any browser.
const jsonRequest = (method, body) => ({method, headers: {"Content-Type": "application/json"}, body: JSON.stringify(body || {})});
function saveToAccount(url, method, body) {
  return api(url, jsonRequest(method, body)).catch(() => toast("That couldn't be saved to your account. Check your connection and try again.", "error"));
}
async function loadLibrary() {
  if (!state.user) return;
  const localWishlist = readWorkspaceStorage("defaketive-wishlist", []);
  const wishlist = Array.isArray(localWishlist) ? localWishlist.filter(link => safeLink(link)) : [];
  const localActivity = readWorkspaceStorage("defaketive-activity", {});
  // Bring over what this browser's guest actually opened or saved, once.
  const opened = Object.fromEntries(Object.entries(localActivity && typeof localActivity === "object" ? localActivity : {})
    .filter(([link, entry]) => safeLink(link) && (entry?.openedAt > 0 || wishlist.includes(link))));
  let library;
  try {
    library = Object.keys(opened).length || wishlist.length
      ? await api("/api/me/import", jsonRequest("POST", {activity: opened, wishlist}))
      : await api("/api/me/library");
  } catch (_) {
    return toast("Your saved history couldn't be loaded. Reload the page to try again.", "error");
  }
  // The next guest on a shared computer must not see this user's history.
  try { localStorage.removeItem("defaketive-activity"); localStorage.removeItem("defaketive-wishlist"); } catch (_) {}
  state.activity = library.activity || {};
  state.wishlist = library.wishlist || [];
  cacheWorkspaceProducts(state.catalogProducts);
  renderWorkspace();
  syncReportSave();
}
function forgetLibrary() {
  state.activity = {};
  state.wishlist = [];
  cacheWorkspaceProducts(state.catalogProducts);
  renderWorkspace();
  syncReportSave();
}
function renderWorkspace() {
  const products = workspaceProducts();
  renderHomeProducts(products);
  const query = $("#history-search").value.trim().toLowerCase(), risk = $("#history-risk").value;
  const filtered = products.filter(product => (!query || `${product.name} ${platform(product)}`.toLowerCase().includes(query)) && matchesConcern(product, risk));
  $("#history-count").textContent = `${filtered.length} of ${products.length} saved reports`;
  $("#history-list").innerHTML = workspaceRows(filtered, products.length ? "No reports match these filters" : "No analyses yet", products.length ? "Try another product name or show all reports." : "Analyze a product to start your history.");
  $("#wishlist-list").innerHTML = workspaceRows(products.filter(product => state.wishlist.includes(product.link)), "Keep a product in mind", "Use the heart beside a saved analysis or save a product from its report. It will appear here.");
  ["#history-list", "#wishlist-list"].forEach(selector => {
    const holder = $(selector);
    bindProductButtons(holder);
    holder.querySelectorAll("[data-save-link]").forEach(button => button.addEventListener("click", () => toggleWishlist(button.dataset.saveLink)));
  });
}
async function routeFromHash() {
  const [route, query = ""] = window.location.hash.slice(1).split("?");
  const params = new URLSearchParams(query);
  if (["about", "faq", "how", "platforms"].includes(route)) {
    go("landing", true);
    requestAnimationFrame(() => $("#" + route)?.scrollIntoView({behavior: "smooth"}));
    return;
  }
  if (route === "analysis" && params.get("product")) {
    const from = params.get("from");
    state.reportFrom = ["home", "history", "wishlist", "results", "analyze"].includes(from) ? from : "history";
    if (state.currentProduct?.link !== params.get("product")) {
      await openProduct(params.get("product"), true, true);
      return;
    }
  }
  go(route || "landing", true);
}
function platform(product) { return product?.platform ? (MARKETPLACES[String(product.platform).toLowerCase()] || product.platform) : "Marketplace"; }
function productCard(product) { const image = safeImage(product); const link = safeLink(product.link); return `<article class="product-card"><div class="product-image">${image ? `<img src="${esc(image)}" alt="">` : "Listing image unavailable"}</div><div class="product-card-body"><span class="platform-badge">${esc(platform(product))}</span><h3>${esc(product.name || "Unnamed product")}</h3><div class="product-price">${esc(money(product.price))}</div><div class="card-meta">★ ${esc(product.rating || "—")} · ${reviewCount(product).toLocaleString()} reviews analyzed</div>${verdictBadge(product)}${reportChips(product)}<button class="card-button" data-product-link="${esc(link)}">Open report →</button></div></article>`; }
function bindProductButtons(root = document) {
  root.querySelectorAll("[data-product-link]").forEach(button => button.addEventListener("click", () => openProduct(button.dataset.productLink)));
}
function filteredProducts() { const platformFilter = $("#filter-platform")?.value || "all"; const concernFilter = $("#filter-risk")?.value || "all"; const sort = $("#filter-sort")?.value || "fewest"; const visible = state.products.filter(product => (platformFilter === "all" || platform(product) === platformFilter) && matchesConcern(product, concernFilter)); return visible.sort((a, b) => sort === "reviews" ? reviewCount(b) - reviewCount(a) : sort === "most" ? byConcern(a, b) : byConcern(b, a)); }
function renderResults() { const products = filteredProducts(); $("#results-summary").textContent = products.length ? `${products.length} analyzed product${products.length === 1 ? "" : "s"} shown.` : "No products match your selected filters."; $("#product-grid").innerHTML = products.map(productCard).join(""); bindProductButtons($("#product-grid")); }
async function loadProducts() { const data = await api("/api/products"); state.products = data.products || []; state.catalogProducts = [...state.products]; state.catalogError = false; state.catalogLoading = false; cacheWorkspaceProducts(state.catalogProducts); if ($("#search-form")) updateSearchUI(); renderWorkspace(); }
// Highlight only words from categories the check actually tagged, so a heat
// word the filter set aside as clothing or weather talk is not marked.
function reviewTerms(analysis) { const found = new Set(); (analysis.sentences || []).forEach(sentence => { const tags = new Set(sentence.tags || []); (sentence.terms || []).forEach(term => { if (term.negated || !tags.has(term.category) || !Array.isArray(term.span)) return; const text = String(sentence.text_raw || "").slice(term.span[0], term.span[1]).trim(); if (text) found.add(text); }); }); return [...found].sort((a, b) => b.length - a.length); }
function highlightReview(review) { const terms = reviewTerms(review.sentiment_analysis || {}); const content = esc(review.content || "No written review was collected."); if (!terms.length) return content; const pattern = terms.map(term => esc(term).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"); return content.replace(new RegExp(`(${pattern})`, "ig"), "<mark>$1</mark>"); }
async function openProduct(link, replace = false, restoring = false) {
  if (!safeLink(link)) return toast("This product has no valid listing link.", "error");
  if (!restoring && currentView() !== "analysis") state.reportFrom = ["home", "history", "wishlist", "results", "analyze"].includes(currentView()) ? currentView() : "history";
  if (restoring) {
    state.currentProduct = null;
    go("analysis", true, true);
    $("#analysis-content").innerHTML = '<div class="flow-empty" role="status">Loading product report…</div>';
  }
  const originHash = window.location.hash;
  const requestId = (state.reportRequestId || 0) + 1;
  state.reportRequestId = requestId;
  try {
    const data = await api(`/api/products/detail?${new URLSearchParams({link})}`);
    if (state.reportRequestId !== requestId || window.location.hash !== originHash) return;
    renderAnalysis(data.product, data.recommendations || []);
    go("analysis", replace);
  } catch (error) {
    toast(error.message, "error");
    if (restoring && window.location.hash === originHash) go("history", true);
  }
}
async function openSignedInPage() {
  if (state.user) go("home");
  else openAuth("signin");
}
// --- Sign in, create account, and the terms pop-up ------------------------
const AUTH_DIALOGS = {signin: "#signin-dialog", signup: "#signup-dialog", terms: "#terms-dialog"};
function closeDialogs() { document.querySelectorAll("dialog[open]").forEach(dialog => dialog.close()); }
function clearAuthErrors(root) {
  root.querySelectorAll("[data-error-for], [data-form-error]").forEach(node => { node.hidden = true; node.textContent = ""; });
  root.querySelectorAll('[aria-invalid="true"]').forEach(node => node.removeAttribute("aria-invalid"));
}
function openAuth(kind) {
  if (state.user) return go("home");
  const dialog = $(AUTH_DIALOGS[kind]);
  if (!dialog) return;
  closeDialogs();
  clearAuthErrors(dialog);
  dialog.showModal();
  dialog.querySelector("input")?.focus();
}
function openTerms(section) {
  const dialog = $("#terms-dialog");
  if (!dialog.open) dialog.showModal();
  const target = $(section === "privacy" ? "#privacy-policy" : "#terms-of-use");
  target?.scrollIntoView({block: "start"});
  target?.querySelector("h3")?.setAttribute("tabindex", "-1");
  target?.querySelector("h3")?.focus();
}
function showAuthErrors(form, error) {
  const fields = error.body?.fields || {};
  let shown = false;
  Object.entries(fields).forEach(([name, message]) => {
    const slot = form.querySelector(`[data-error-for="${name}"]`);
    const input = form.querySelector(`[name="${name}"]`);
    if (slot) { slot.textContent = message; slot.hidden = false; shown = true; }
    if (input) input.setAttribute("aria-invalid", "true");
  });
  if (!shown) {
    const general = form.querySelector("[data-form-error]");
    general.textContent = error.message;
    general.hidden = false;
  }
  (form.querySelector('[aria-invalid="true"]') || form.querySelector("input"))?.focus();
}
async function submitAuth(event) {
  event.preventDefault();
  const form = event.currentTarget, kind = form.dataset.authForm;
  clearAuthErrors(form);
  const field = name => form.elements[name]?.value || "";
  const payload = kind === "signup"
    ? {full_name: field("full_name"), email: field("email"), password: field("password"), confirm: field("confirm"), agree: Boolean(form.elements.agree?.checked)}
    : {email: field("email"), password: field("password")};
  const button = form.querySelector(".auth-submit"), label = button.textContent;
  button.disabled = true;
  button.textContent = kind === "signup" ? "Creating account…" : "Signing in…";
  try {
    const me = await api(kind === "signup" ? "/api/auth/signup" : "/api/auth/login", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    form.reset();
    await afterSignIn(me, kind);
  } catch (error) {
    showAuthErrors(form, error);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}
async function afterSignIn(me, kind) {
  applyAuth(me);
  closeDialogs();
  state.loginNotice = "";
  await loadLibrary();
  toast(kind === "signup" ? `Welcome, ${firstName(me.user?.name)}. Your account is ready.` : `Signed in as ${me.user?.name}.`);
  const here = currentView();
  const next = state.afterLogin || (here === "landing" || here === "login" ? "home" : here);
  state.afterLogin = null;
  if (next !== here) go(next);
}
async function signOut() {
  try {
    applyAuth(await api("/api/auth/logout", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"}));
  } catch (error) {
    return toast(error.message, "error");
  }
  forgetLibrary();
  toast("You're signed out.");
  go("landing");
}
function continueAsGuest() {
  closeDialogs();
  go("analyze");
  focusProductSearch();
}
function showLoginPage(message, after = "") {
  state.loginNotice = message || "";
  if (after) state.afterLogin = after;
  go("login");
}
async function prepareLoginPage() {
  const notice = $("[data-login-notice]");
  if (notice) {
    notice.textContent = state.loginNotice || "";
    notice.hidden = !state.loginNotice;
  }
  if (state.publicStats) return;
  try {
    state.publicStats = await api("/api/stats/public");
    document.querySelectorAll("[data-public-stat]").forEach(node => {
      const value = Number(state.publicStats[node.dataset.publicStat]);
      if (Number.isFinite(value)) node.textContent = value.toLocaleString();
    });
  } catch (_) { /* The numbers are a nicety; the form still works without them. */ }
}
// Who is signed in, and how many free scans a guest has left today.
const firstName = name => String(name || "").trim().split(/\s+/)[0] || "";
function guestNote() {
  if (state.user) return "Unlimited scans with your account";
  const left = Number(state.guest?.left ?? 3);
  return left > 0 ? `${left} free scan${left === 1 ? "" : "s"} left today as a guest · sign in for unlimited` : "No free scans left today · sign in for unlimited scans";
}
function applyAuth(me) {
  state.user = me?.user || null;
  state.guest = me?.guest || null;
  document.body.dataset.auth = state.user ? "user" : "guest";
  document.querySelectorAll("[data-auth-guest]").forEach(node => { node.hidden = Boolean(state.user); });
  document.querySelectorAll("[data-auth-user]").forEach(node => { node.hidden = !state.user; });
  document.querySelectorAll("[data-user-greeting]").forEach(node => { node.textContent = state.user ? `Hi, ${firstName(state.user.name)}` : ""; });
  document.querySelectorAll("[data-guest-note]").forEach(node => { node.textContent = guestNote(); });
  const parts = String(state.user?.name || "").trim().split(/\s+/).filter(Boolean);
  const initials = parts.length ? (parts[0][0] + (parts.length > 1 ? parts[parts.length - 1][0] : "")).toUpperCase() : "";
  document.querySelectorAll("[data-user-initials]").forEach(node => { node.textContent = initials; });
  document.querySelectorAll("[data-user-name]").forEach(node => { node.textContent = state.user?.name || ""; });
  const lead = $("[data-welcome-lead]"), who = $("[data-welcome-name]");
  if (lead) lead.textContent = state.user ? "Welcome back," : "Welcome,";
  if (who) who.textContent = state.user ? `${firstName(state.user.name)}.` : "shopper.";
  closeAccountMenu();
  updateLimitNotice();
  const left = Number(state.guest?.left ?? 3);
  document.querySelectorAll("[data-continue-guest]").forEach(node => {
    node.textContent = left > 0 ? `Continue as guest — ${left} free scan${left === 1 ? "" : "s"} left today` : "Continue as guest — no free scans left today";
  });
}
async function refreshAuth() {
  try { applyAuth(await api("/api/auth/me")); } catch (_) { return; /* Keep the guest view if the check fails. */ }
  if (state.user) await loadLibrary();
}
function sentimentData(product) { const counts = product.sentiment_summary?.sentiment_counts || {}; return [{label:"Positive", value:Number(counts.positive || 0), color:"#12b76a"},{label:"Uncertain", value:Number(counts.uncertain ?? counts.neutral ?? 0), color:"#94a3b8"},{label:"Negative", value:Number(counts.negative || 0), color:"#f04438"}]; }

function sentimentPercentages(product) {
  const data = sentimentData(product);
  const total = data.reduce((sum, item) => sum + item.value, 0);
  const shares = product.sentiment_summary?.sentiment_shares;
  // Per-star samples: percentages come from the weighted shares, counts stay counts.
  if (shares) return data.map(item => ({...item, percent: Number(shares[item.label.toLowerCase()] || 0) * 100}));
  return data.map(item => ({...item, percent: total ? item.value / total * 100 : 0}));
}



// --- Product report --------------------------------------------------------
const REVIEW_TAG_LABELS = {safety: "Safety report", counterfeit: "Possible fake", defect: "Defect report"};
const sentimentLabel = review => { const label = String(review?.sentiment_analysis?.label || "unanalyzed").toLowerCase(); return label === "neutral" ? "uncertain" : label; };
// The words the check matched in a review: only words from a category the
// sentence was actually tagged with, and never a negated one ("hindi sira").
function matchedWords(review) {
  const found = new Map();
  (review?.sentiment_analysis?.sentences || []).forEach(sentence => {
    const tags = new Set(sentence.tags || []);
    (sentence.terms || []).forEach(term => {
      if (term.negated || !tags.has(term.category)) return;
      const raw = Array.isArray(term.span) ? String(sentence.text_raw || "").slice(term.span[0], term.span[1]) : "";
      const word = String(term.ngram || raw).trim().toLowerCase();
      if (word && !found.has(word)) found.set(word, term.category);
    });
  });
  return found;
}
// How many reviews each warning word appears in, most common first.
function warningWords(reviews) {
  const tally = new Map();
  reviews.forEach(review => matchedWords(review).forEach((category, word) => {
    const entry = tally.get(word) || {word, category, count: 0};
    entry.count += 1;
    tally.set(word, entry);
  }));
  return [...tally.values()].sort((a, b) => b.count - a.count || a.word.localeCompare(b.word));
}
// One plain sentence for the summary: how the reviews lean, and what was flagged.
function summaryHeadline(positiveShare, negativeShare, reportCounts) {
  const lean = positiveShare >= 70 ? "Mostly positive reviews" : negativeShare >= 50 ? "Mostly negative reviews" : "Mixed reviews";
  const parts = [["safety", "safety report"], ["counterfeit", "fake report"], ["defect", "defect report"]]
    .filter(([key]) => reportCounts[key] > 0)
    .map(([key, noun]) => `${reportCounts[key]} ${noun}${reportCounts[key] === 1 ? "" : "s"}`);
  if (!parts.length) return `${lean}, and nothing flagged.`;
  const list = parts.length > 1 ? parts.slice(0, -1).join(", ") + " and " + parts[parts.length - 1] : parts[0];
  return `${lean}, with ${list}.`;
}
function reviewCard(review) {
  const label = sentimentLabel(review);
  const reported = (review.sentiment_analysis?.review_tags || []).map(tag => REVIEW_TAG_LABELS[tag]).filter(Boolean);
  const words = [...matchedWords(review).keys()].slice(0, 4);
  const when = review.time || review.date || "";
  const tags = [...reported.map(text => `<span class="flag">${esc(text)}</span>`), ...words.map(word => `<span>${esc(word)}</span>`)].join("");
  return `<article class="rep-review"><header>${smallStars(review.rating)}${when ? `<time>${esc(when)}</time>` : ""}</header><p><i class="rep-dot ${esc(label)}" role="img" aria-label="${esc(label)} review"></i><span>${highlightReview(review)}</span></p>${tags ? `<div class="rep-tags">${tags}</div>` : ""}</article>`;
}

// What each verdict means, in the order the server tests them.
const VERDICT_GUIDE = [
  ["safety_warning", "high", "Reviewers report safety problems", "Serious or repeated reports of fire, smoke, shocks or overheating"],
  ["possible_safety_issue", "moderate", "Some reviewers mention safety concerns", "Fewer or less severe safety reports"],
  ["possible_counterfeit_or_defect", "moderate", "Reviewers report fakes or defects", "More fake or breakage reports than is typical for the product type"],
  ["reviews_may_be_unreliable", "unknown", "These reviews may not be trustworthy", "Many reviews look duplicated or generic"],
  ["no_warning_signs", "low", "No warning signs found", "Enough reviews, none flagged. This is not proof that a product is safe"],
  ["insufficient_evidence", "unknown", "Not enough reviews to judge", "Too few reviews for an overall verdict; any reports are still shown"],
];
// The three labels a shopper sees on cards, and what each one means.
const VERDICT_GROUPS = [
  ["clear", "No warnings", "Enough reviews, none flagged. Not proof it's safe."],
  ["caution", "Caution", "Reviewers report fakes, defects or milder safety concerns."],
  ["safety", "Safety reports", "Serious or repeated reports of fire, smoke, shocks or overheating."],
];
const CHECK_LABELS = {safety: "Safety", counterfeit: "Fakes", defect: "Defects"};

function renderAnalysis(product, recommendations) {
  state.currentProduct = product;
  rememberProduct(product);
  const summary = product.sentiment_summary || {};
  const verdict = view(product);
  const chip = chipFor(product);
  const accuracy = verdict.safety_check_accuracy || null;
  const allReviews = Array.isArray(product.comments) ? product.comments.filter(item => item && typeof item === "object") : [];
  const reviews = allReviews.filter(item => !item.is_duplicate && item.included_in_analysis !== false);
  const coverage = summary.coverage || {};
  const uniqueCount = Number(verdict.reviews_analyzed || reviews.length) || reviews.length;
  const availableCount = Math.max(uniqueCount, Number(coverage.reviews_listed || coverage.reviews_available || uniqueCount) || uniqueCount);
  const coveragePercent = availableCount ? uniqueCount / availableCount * 100 : 0;
  // A per-star sample is weighted to the listing, so "coverage" would mislead.
  const sampleNote = coverage.sampling === "stratified_by_star"
    ? `${uniqueCount.toLocaleString()} reviews read, up to ${Number(coverage.star_quota || LAZADA_REVIEWS_PER_STAR)} from each star level${coverage.weighted ? `, weighted to match the listing's ${availableCount.toLocaleString()} ratings` : ""}`
    : `${uniqueCount.toLocaleString()} of ${availableCount.toLocaleString()} reviews analyzed (${coveragePercent.toFixed(1)}% coverage)`;
  const duplicateCount = Number(summary.duplicate_review_count ?? allReviews.filter(item => item.is_duplicate).length) || 0;
  const flagged = reviews.filter(item => (item.sentiment_analysis?.review_tags || []).length);
  const image = safeImage(product), link = safeLink(product.link), name = product.name || "Unnamed product";
  const customerRating = Math.max(0, Math.min(5, Number(product.rating || 0)));
  const sentiment = sentimentPercentages(product);
  const share = label => sentiment.find(item => item.label === label) || {value: 0, percent: 0};
  const positive = share("Positive"), uncertain = share("Uncertain"), negative = share("Negative");
  const date = reportDate(product);
  // Lazada scans read a set number of reviews per star level, so low-star reviews
  // are over-represented in the raw counts. Shares are weighted back to the listing.
  const weighted = Boolean(summary.sentiment_shares);
  const reportCounts = counts(product);
  const words = warningWords(reviews);
  const saved = state.wishlist.includes(link);
  const status = $("#report-status");
  if (status) status.textContent = verdict.available ? `Analysis complete · ${uniqueCount.toLocaleString()} reviews processed` : "Not analyzed yet";

  const guideRows = VERDICT_GUIDE.map(([stateName, rowTone, label, meaning]) => `<tr class="${verdict.state === stateName ? "current" : ""}"><td><span class="${rowTone}">${esc(label)}</span>${verdict.state === stateName ? '<em>This product</em>' : ""}</td><td>${esc(meaning)}</td></tr>`).join("");
  const accuracyCopy = accuracy
    ? `It is not perfect. When we tested it on ${Number(accuracy.sentences_checked).toLocaleString()} review sentences that two people had checked by hand, it caught ${accuracy.reports_caught} of ${accuracy.reports_confirmed} real safety reports, and ${accuracy.flags_raised - accuracy.flags_confirmed} of the ${accuracy.flags_raised} sentences it flagged turned out to be harmless. A flagged review is a reason to read, not proof of danger, and no flags is not proof of safety.`
    : "A flagged review is a reason to read, not proof of danger, and no flags is not proof of safety.";
  const meta = [
    ["Platform", platform(product)], ["Category", product.category || "Not specified"], ["Price", money(product.price)],
    ["Positive reviews", weighted ? `${positive.percent.toFixed(0)}% (weighted)` : `${positive.value} / ${uniqueCount}`], ["Negative reviews", weighted ? `${negative.percent.toFixed(0)}% (weighted)` : `${negative.value} / ${uniqueCount}`], ["Analysis date", date?.label || "Unavailable"],
  ].map(([term, value]) => `<div><dt>${term}</dt><dd>${esc(value)}</dd></div>`).join("");

  const checks = (verdict.badges || []).map(badge => {
    const quote = (badge.evidence || [])[0];
    const notes = quote ? [quote.hazard ? `Reports ${esc(quote.hazard)}` : "", quote.rating ? `${Number(quote.rating).toFixed(0)}-star review` : ""].filter(Boolean).join(" · ") : "";
    const width = uniqueCount ? Math.min(100, Math.max(badge.mentions ? 3 : 0, badge.mentions / uniqueCount * 100)) : 0;
    return `<div class="rep-check ${esc(badge.category)}"><p><span>${esc(CHECK_LABELS[badge.category] || badge.label)}</span><b>${badge.mentions} review${badge.mentions === 1 ? "" : "s"}</b></p><div class="rep-bar" role="img" aria-label="${badge.mentions} of ${uniqueCount} reviews"><i style="width:${width}%"></i></div><p class="rep-check-note">${esc(badge.summary)}</p>${quote ? `<blockquote>${highlightSpans(quote.text, quote.highlights)}${notes ? `<small>${notes}</small>` : ""}</blockquote><button type="button" class="rep-link" data-review-tag="${esc(badge.category)}">Read all ${badge.mentions} →</button>` : ""}</div>`;
  }).join("") || '<p class="rep-empty">This product has not been checked with the current review model.</p>';

  const topWords = words.slice(0, 6), maxWord = topWords[0]?.count || 1;
  const wordRows = topWords.length
    ? topWords.map(item => `<div class="rep-word"><button type="button" class="${esc(item.category)}" data-review-word="${esc(item.word)}" title="Show reviews that mention this">${esc(item.word)}</button><div class="rep-bar"><i style="width:${item.count / maxWord * 100}%"></i></div><b>${item.count}×</b></div>`).join("")
    : '<p class="rep-empty">The check matched no warning words in these reviews.</p>';

  const lookFor = [
    [reportCounts.safety, "Safety", "Fire, smoke, shocks, overheating", "safety"],
    [reportCounts.counterfeit, "Genuine product", "Fakes, replicas, “not original”", "counterfeit"],
    [reportCounts.defect, "Works as expected", "Broken, dead or faulty", "defect"],
  ].map(([count, title, text, kind]) => `<div class="rep-tile"><p><b>${count}</b><small> review${count === 1 ? "" : "s"}</small></p><h3>${title}</h3><p class="rep-tile-note"><i class="rep-dot ${count ? kind : "none"}"></i>${text}</p></div>`).join("")
    + `<div class="rep-tile"><p><b>${positive.percent.toFixed(0)}%</b><small> positive</small></p><h3>Sentiment</h3><p class="rep-tile-note"><i class="rep-dot none"></i>Context only. It does not change the verdict.</p></div>`;

  const alternatives = recommendations.slice(0, 3);
  const alternativeCards = alternatives.length
    ? `<div class="rep-alts">${alternatives.map(item => {
        const itemImage = safeImage(item), itemLink = safeLink(item.link), itemCounts = counts(item);
        const itemPositive = sentimentPercentages(item).find(entry => entry.label === "Positive")?.percent || 0;
        return `<article class="rep-alt"><header><div class="rep-alt-image">${itemImage ? `<img src="${esc(itemImage)}" width="64" height="64" alt="" loading="lazy">` : "<span>No photo</span>"}</div><h3>${esc(item.name || "Unnamed product")}</h3></header><b class="rep-alt-price">${esc(money(item.price))}</b><dl><div><dt>Positive</dt><dd>${itemPositive.toFixed(0)}%</dd></div><div><dt>Safety reports</dt><dd>${itemCounts.safety}</dd></div><div><dt>Fake / defect</dt><dd>${itemCounts.counterfeit + itemCounts.defect}</dd></div></dl><footer><span class="rep-pill clear">✓ No warning signs</span><span><button type="button" class="rep-link recommendation-open" data-product-link="${esc(itemLink)}">Open report →</button>${itemLink ? `<a class="rep-link" target="_blank" rel="noreferrer" href="${esc(itemLink)}">Go to product <img class="app-ink-icon" src="/static/assets/app/external.svg" width="10" height="10" alt=""></a>` : ""}</span></footer></article>`;
      }).join("")}</div>`
    : '<p class="rep-empty">No similar product with enough reviews and no warning signs is in the analyzed collection yet.</p>';

  const filterCounts = {all: reviews.length, reported: flagged.length, negative: 0, uncertain: 0, positive: 0};
  reviews.forEach(review => { const label = sentimentLabel(review); if (label in filterCounts) filterCounts[label] += 1; });
  const defaultReviewFilter = flagged.length ? "reported" : "all";

  $("#analysis-content").innerHTML = `<section class="rep-card rep-product">
      <div class="rep-product-image">${image ? `<img src="${esc(image)}" width="168" height="168" alt="">` : "<span>Listing image unavailable</span>"}</div>
      <div class="rep-product-main">
        <div class="rep-product-head"><p class="rep-eyebrow tone-green">Analyzed product</p><div class="rep-actions"><button type="button" id="report-save" class="rep-outline" data-report-save aria-pressed="${saved}">${saved ? "♥ Saved" : "♡ Save"}</button>${link ? `<a class="rep-outline" target="_blank" rel="noreferrer" href="${esc(link)}">Go to product <img class="app-ink-icon" src="/static/assets/app/external.svg" width="11" height="11" alt=""></a>` : ""}</div></div>
        <h1 tabindex="-1">${esc(name)}</h1>
        <div class="rep-rating">${smallStars(customerRating)}<span>${customerRating ? `${customerRating.toFixed(1)} out of 5 · ` : ""}${uniqueCount.toLocaleString()} reviews analyzed</span></div>
        <dl class="rep-meta">${meta}</dl>
        <p class="rep-sample">${esc(sampleNote)}</p>
        <div class="rep-verdict"><span class="rep-pill ${chip.kind}">${esc(chip.label)}</span><strong>${esc(verdict.headline)}</strong><span class="rep-verdict-label">What reviewers report</span></div>
        <p class="rep-verdict-detail">${esc(verdict.detail)} Review signals are a guide, not proof.</p>
      </div>
    </section>
    <section class="rep-card rep-verdicts">
      <header class="rep-head"><div><p class="rep-eyebrow">Verdicts</p><h2>How DeFaketive Labels a Product</h2><p class="rep-sub">Each product gets a plain verdict from what its reviewers reported. There is no score.</p></div><button type="button" class="rep-outline rep-toggle" data-rep-toggle="report-verdict-detail" aria-expanded="false"><span>View full detail</span><img class="app-ink-icon" src="/static/assets/app/chevron.svg" width="11" height="11" alt=""></button></header>
      <div class="rep-scale" aria-hidden="true"><i class="clear"></i><i class="caution"></i><i class="safety"></i></div>
      <div class="rep-scale-labels" aria-hidden="true"><span class="clear">No warnings</span><span class="caution">Caution</span><span class="safety">Safety reports</span></div>
      <div class="rep-groups">${VERDICT_GROUPS.map(([kind, label, meaning]) => `<div class="rep-group ${kind}${chip.kind === kind ? " current" : ""}"><p><i class="rep-dot ${kind}"></i><b>${label}</b>${chip.kind === kind ? "<em>This product</em>" : ""}</p><p>${meaning}</p></div>`).join("")}</div>
      <div id="report-verdict-detail" class="rep-table" hidden><table><thead><tr><th>Verdict</th><th>What it means</th></tr></thead><tbody>${guideRows}</tbody></table><p>This product: <b>${esc(verdict.headline)}</b>. ${esc(verdict.detail)}</p></div>
    </section>
    <details id="report-details" class="rep-card rep-how"><summary><span><span class="rep-eyebrow">How the check works</span><b>Why you see reviews, not a score</b></span><img class="app-ink-icon" src="/static/assets/app/chevron.svg" width="14" height="14" alt=""></summary>
      <p>DeFaketive reads every review sentence and looks for the words people use when a product catches fire, smokes, gives a shock, turns out to be fake, or breaks, in English, Tagalog and Taglish. It shows you the sentences it found instead of a single score, because a score hides what reviewers actually said.</p>
      <p>${esc(accuracyCopy)}</p>
      <p class="rep-how-stats"><span>${flagged.length} of ${uniqueCount} reviews flagged</span><span>${duplicateCount} duplicates excluded</span></p>
    </details>
    <div class="rep-divider"><span>Our analysis</span><i></i><button type="button" class="rep-outline rep-toggle" data-rep-toggle="report-summary" aria-expanded="false"><span>View full detail</span><img class="app-ink-icon" src="/static/assets/app/chevron.svg" width="11" height="11" alt=""></button></div>
    <section id="report-summary" class="rep-summary" hidden>
      <div><p class="rep-eyebrow">Analysis summary</p><h2>${esc(summaryHeadline(positive.percent, negative.percent, reportCounts))}</h2><p>DeFaketive read ${uniqueCount.toLocaleString()} unique reviews (${duplicateCount} duplicates excluded): ${weighted ? `about ${positive.percent.toFixed(0)}% positive, ${uncertain.percent.toFixed(0)}% uncertain and ${negative.percent.toFixed(0)}% negative once weighted to the listing's star mix` : `${positive.value} positive, ${uncertain.value} uncertain and ${negative.value} negative`}. The check flagged ${flagged.length} review${flagged.length === 1 ? "" : "s"}${topWords.length ? `, led by ${topWords.slice(0, 2).map(item => `“${esc(item.word)}”`).join(" and ")}` : ""}.</p></div>
      <dl><div><dt>Positive</dt><dd>${positive.percent.toFixed(1)}%</dd></div><div><dt>Negative</dt><dd>${negative.percent.toFixed(1)}%</dd></div><div><dt>Flagged reviews</dt><dd>${flagged.length}</dd></div><div><dt>Duplicates excluded</dt><dd>${duplicateCount}</dd></div></dl>
    </section>
    <div class="rep-grid">
      <section class="rep-card"><p class="rep-eyebrow tone-green">Sentiment distribution</p><h2>Review Sentiment</h2><p class="rep-sub">${weighted ? "Shares are weighted to the listing's star mix, because low-star reviews were read on purpose." : "How the reviews lean. Shown for context; it does not change the verdict."}</p>
        <div class="rep-sentiment"><div class="rep-donut" style="--positive:${positive.percent}%;--uncertain:${positive.percent + uncertain.percent}%" role="img" aria-label="${positive.percent.toFixed(0)}% positive, ${uncertain.percent.toFixed(0)}% uncertain, ${negative.percent.toFixed(0)}% negative"><span><b>${uniqueCount}</b><small>reviews</small></span></div>
          <ul>${[["positive", "Positive reviews", positive], ["uncertain", "Uncertain reviews", uncertain], ["negative", "Negative reviews", negative]].map(([kind, label, item]) => `<li><i class="rep-dot ${kind}"></i><span>${label}<small>${weighted ? `${item.value} of ${uniqueCount} read` : `${item.percent.toFixed(0)}% of reviews`}</small></span><b>${weighted ? item.percent.toFixed(0) + "%" : item.value}</b></li>`).join("")}</ul></div>
      </section>
      <section id="report-findings" class="rep-card" tabindex="-1"><p class="rep-eyebrow tone-green">What reviewers reported</p><h2>Three checks, in their own words</h2><p class="rep-sub">The sentences the check flagged, with the words it matched highlighted.</p><div class="rep-checks">${checks}</div></section>
      <section class="rep-card"><p class="rep-eyebrow tone-red">Warning words</p><h2>Warning words found</h2><p class="rep-sub">Words the check matched in these reviews. Pick one to see its reviews.</p><div class="rep-words">${wordRows}</div></section>
      <section class="rep-card"><p class="rep-eyebrow tone-green">What we check</p><h2>What the check looks for</h2><p class="rep-sub">How many reviews mentioned each kind of problem.</p><div class="rep-tiles">${lookFor}</div></section>
    </div>
    <section id="report-evidence" class="rep-card rep-evidence" tabindex="-1">
      <header class="rep-head"><div><p class="rep-eyebrow">Review evidence</p><p id="review-result-count" class="rep-sub" aria-live="polite"></p></div><label class="rep-search"><img src="/static/assets/app/search.svg" width="12" height="12" alt=""><span class="sr-only">Filter reviews by word</span><input id="review-search" type="search" placeholder="Filter by word..." autocomplete="off"></label></header>
      <div class="rep-filters"><div role="group" aria-label="Filter reviews"><button type="button" data-review-filter="all">All · ${filterCounts.all}</button><button type="button" data-review-filter="reported">Flagged · ${filterCounts.reported}</button><button type="button" data-review-filter="negative">Negative · ${filterCounts.negative}</button><button type="button" data-review-filter="uncertain">Uncertain · ${filterCounts.uncertain}</button><button type="button" data-review-filter="positive">Positive · ${filterCounts.positive}</button></div>${topWords.length ? `<i aria-hidden="true"></i><div role="group" aria-label="Filter by warning word">${topWords.slice(0, 5).map(item => `<button type="button" class="word" data-review-word="${esc(item.word)}">${esc(item.word)}</button>`).join("")}</div>` : ""}</div>
      <div id="analysis-review-list" class="rep-reviews"></div>
      <button id="show-more-reviews" class="rep-more" type="button"></button>
    </section>
    <section class="rep-card"><p class="rep-eyebrow tone-green">Other options</p><p class="rep-sub lead">Similar products with no warning signs: enough reviews, and not a single safety report.</p>${alternativeCards}</section>
    <p class="rep-disclaimer">This report analyzes a sample of public reviews. It supports shopping decisions but does not prove that a listing or product is authentic or counterfeit.</p>
    <button type="button" class="rep-back-top" data-back-top>Back to top</button>`;

  let activeFilter = defaultReviewFilter, activeTag = "", activeWord = "", visibleLimit = 6;
  const content = $("#analysis-content");
  const drawReviews = () => {
    const query = $("#review-search").value.trim().toLowerCase();
    const filtered = reviews.filter(review => {
      const tags = review.sentiment_analysis?.review_tags || [];
      const matchesType = activeFilter === "all" || (activeFilter === "reported" ? Boolean(tags.length) : sentimentLabel(review) === activeFilter);
      const matchesTag = !activeTag || tags.includes(activeTag);
      const matchesWord = !activeWord || matchedWords(review).has(activeWord);
      return matchesType && matchesTag && matchesWord && (!query || String(review.content || "").toLowerCase().includes(query));
    });
    const visible = filtered.slice(0, visibleLimit);
    const narrowed = [activeTag ? REVIEW_TAG_LABELS[activeTag].toLowerCase() + "s" : "", activeWord ? `“${activeWord}”` : ""].filter(Boolean).join(", ");
    $("#review-result-count").textContent = `Showing ${visible.length} of ${filtered.length} matching review${filtered.length === 1 ? "" : "s"}${narrowed ? ` for ${narrowed}` : ""}.`;
    $("#analysis-review-list").innerHTML = visible.length ? visible.map(reviewCard).join("") : '<p class="rep-empty">No reviews match these filters.</p>';
    const showMore = $("#show-more-reviews");
    showMore.hidden = visible.length >= filtered.length;
    showMore.textContent = `View ${Math.min(6, filtered.length - visible.length)} more reviews`;
    document.querySelectorAll("[data-review-filter]").forEach(item => item.classList.toggle("active", item.dataset.reviewFilter === activeFilter));
    document.querySelectorAll("[data-review-word]").forEach(item => { const on = item.dataset.reviewWord === activeWord; item.classList.toggle("active", on); item.setAttribute("aria-pressed", String(on)); });
  };
  const showEvidence = () => { const panel = $("#report-evidence"); panel?.scrollIntoView({behavior: "smooth", block: "start"}); panel?.focus({preventScroll: true}); };
  // One listener for the whole report: it is re-rendered for every product.
  content.onclick = event => {
    const pick = selector => event.target.closest(selector);
    const filter = pick("[data-review-filter]"), word = pick("[data-review-word]"), tag = pick("[data-review-tag]"), toggle = pick("[data-rep-toggle]"), open = pick(".recommendation-open");
    if (filter) {
      activeFilter = filter.dataset.reviewFilter;
      activeTag = "";
      // "All" clears every narrowing, including a picked word and the search box.
      if (activeFilter === "all") { activeWord = ""; $("#review-search").value = ""; }
      visibleLimit = 6;
      drawReviews();
    }
    else if (word) { activeWord = activeWord === word.dataset.reviewWord ? "" : word.dataset.reviewWord; activeFilter = "all"; activeTag = ""; visibleLimit = 6; drawReviews(); if (!word.closest("#report-evidence")) showEvidence(); }
    else if (tag) { activeTag = tag.dataset.reviewTag; activeFilter = "reported"; activeWord = ""; visibleLimit = 6; drawReviews(); showEvidence(); }
    else if (toggle) {
      const target = document.getElementById(toggle.dataset.repToggle), show = target.hidden;
      target.hidden = !show;
      toggle.setAttribute("aria-expanded", String(show));
      toggle.querySelector("span").textContent = show ? "Hide full detail" : "View full detail";
    }
    else if (pick("#show-more-reviews")) { visibleLimit += 6; drawReviews(); }
    else if (pick("[data-report-save]")) toggleWishlist(state.currentProduct?.link);
    else if (open) openProduct(open.dataset.productLink);
    else if (pick("[data-back-top]")) window.scrollTo({top: 0, behavior: "smooth"});
  };
  $("#review-search").addEventListener("input", () => { visibleLimit = 6; drawReviews(); });
  drawReviews();
  updateLimitNotice();
}
function syncReportSave() {
  const button = $("#report-save");
  if (!button) return;
  const saved = state.wishlist.includes(safeLink(state.currentProduct?.link));
  button.setAttribute("aria-pressed", String(saved));
  button.textContent = saved ? "♥ Saved" : "♡ Save";
}
// A guest who is out of free scans sees this over the campus photo on a report.
function updateLimitNotice() {
  const notice = $("#guest-limit-notice");
  if (!notice) return;
  let dismissed = false;
  try { dismissed = sessionStorage.getItem("defaketive-limit-dismissed") === "1"; } catch (_) {}
  const show = !state.user && Number(state.guest?.left) === 0 && currentView() === "analysis" && !dismissed;
  notice.hidden = !show;
  document.body.classList.toggle("has-limit-notice", show);
}

async function search(event) {
  event.preventDefault();
  if (state.searchBusy) return;
  const query = $("#search-query").value.trim();
  if (!query) return;
  const context = queryContext(query);
  if (context.isUrl && !context.platform) {
    showSearchFeedback("Only Shopee PH, Lazada PH, and Temu product links are supported.", "Unsupported product link");
    return;
  }
  const intent = event.submitter?.dataset.searchIntent || $("#search-submit").dataset.searchIntent || "saved";
  const live = intent === "live";
  clearSearchFeedback();
  if (!live && context.platform) {
    const saved = workspaceProducts().find(product => productLinkIdentity(product.link) === productLinkIdentity(query));
    if (saved) return openProduct(saved.link);
  }
  setSearchBusy(true);
  let jobStarted = false;
  try {
    const data = await api("/api/search", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        query,
        platform: context.platform || selectedMarketplace(),
        live,
        product_count: context.platform ? 1 : $("#product-count").value,
        review_count: $("#review-count").value,
      }),
    });
    if (data.guest) applyAuth({user: null, guest: data.guest});
    if (data.job) {
      jobStarted = true;
      state.activeJobId = data.job.id;
      try { sessionStorage.setItem("defaketive-active-job", JSON.stringify(data.job)); } catch (_) {}
      pollJob(data.job.id, data.job);
    } else {
      state.products = data.products || [];
      if (!state.products.length) {
        showSearchFeedback("No saved analysis matched this search. Choose Collect fresh reviews to scan the marketplace, or try another name.", "No saved match", "warning");
        return;
      }
      if (context.isUrl && state.products.length === 1) await openProduct(state.products[0].link);
      else go("results");
    }
  } catch (error) {
    if (error.body?.limit_reached) {
      applyAuth({user: null, guest: error.body.guest});
      showSearchFeedback(error.message, "Out of free scans for today", "warning");
      // After signing in, come back to Analyze with the same link still filled in.
      showLoginPage(error.message, "analyze");
    } else showSearchFeedback(error.message);
  } finally {
    if (!jobStarted) setSearchBusy(false);
  }
}
async function pollJob(id, initialJob = {}) {
  const panel = $("#job-panel");
  const pollToken = (state.pollToken || 0) + 1;
  state.pollToken = pollToken;
  state.activeJobId = id;
  setSearchBusy(true);
  panel.hidden = false;
  const initial = {...initialJob, id, state: "running"};
  persistHomeScan(initial);
  $("#job-title").textContent = initialJob.platform ? `${MARKETPLACES[initialJob.platform] || initialJob.platform} live scan` : "Live analysis running";
  $("#job-message").textContent = initialJob.message || "Opening the marketplace and preparing collection…";
  setJobStage(Number(initialJob.stage || 1), initialJob.progress);
  startJobClock(initialJob.created_at);
  async function tick() {
    try {
      const update = await api(`/api/jobs/${id}`);
      if (state.pollToken !== pollToken) return;
      const job = {...initial, ...update, id};
      $("#job-message").textContent = job.message || "Analysis in progress…";
      setJobStage(Number(job.stage || 1), job.progress);
      if (job.state === "complete") {
        panel.hidden = true;
        finishJob();
        // Keep the terminal result even if refreshing the catalog temporarily fails.
        let loaded = false;
        try { await loadProducts(); loaded = true; }
        catch (_) { toast("The scan finished, but saved results could not be refreshed. Reopen Home to try again.", "warning"); }
        if (state.pollToken !== pollToken) return;
        const directLink = queryContext(job.query).isUrl;
        const product = directLink && loaded ? state.products.find(item => productLinkIdentity(item.link) === productLinkIdentity(job.query)) || (state.products.length === 1 ? state.products[0] : null) : null;
        persistHomeScan({...job, productLink: product?.link || (directLink ? safeLink(job.query) : ""), resultLinks: loaded ? state.products.map(item => item.link).filter(safeLink) : [], resultsPending: !loaded});
        toast("Your report is ready.");
        // A visitor who stayed on the analysis screen continues straight to the report.
        // Other workspace pages keep their place and expose the completed scan card.
        if (currentView() === "analyze" && loaded) {
          if (product) await openProduct(product.link);
          else go("results");
        }
        return;
      }
      if (job.state === "error") {
        panel.hidden = true;
        finishJob();
        persistHomeScan(job);
        showSearchFeedback(job.message || "The marketplace couldn't complete this scan.", "Live scan stopped");
        updateBell();
        return;
      }
      persistHomeScan(job);
      setTimeout(tick, 2500);
    } catch (error) {
      if (state.pollToken !== pollToken) return;
      persistHomeScan({...state.homeScan, state: "attention", message: error.message});
      showSearchFeedback(error.message, "Could not read scan progress");
      updateBell();
    }
  }
  return tick();
}
async function uploadDataset(event) { const file = event.target.files[0]; if (!file) return; const body = new FormData(); body.append("file", file); try { const data = await api("/api/upload", {method:"POST",body}); state.products = data.products || []; state.catalogProducts = [...state.products]; cacheWorkspaceProducts(state.catalogProducts); updateSearchUI(); toast(`${data.count} product${data.count === 1 ? "" : "s"} imported and analyzed.`); go("results"); } catch (error) { toast(error.message, "error"); } finally { event.target.value = ""; } }
function setup() {
  WORKSPACE_ROUTES.forEach(route => $("#workspace-pages").append($("#" + route)));
  const activity = readWorkspaceStorage("defaketive-activity", {});
  state.activity = activity && typeof activity === "object" && !Array.isArray(activity)
    ? Object.fromEntries(Object.entries(activity).filter(([link, entry]) => safeLink(link) && entry?.product?.link === link && Number.isFinite(entry.openedAt))) : {};
  const wishlist = readWorkspaceStorage("defaketive-wishlist", []);
  state.wishlist = Array.isArray(wishlist) ? [...new Set(wishlist.filter(link => safeLink(link)))] : [];
  document.querySelectorAll("[data-start-analysis]").forEach(form => form.addEventListener("submit", event => {
    event.preventDefault();
    if (!state.searchBusy) $("#search-query").value = form.querySelector("input").value.trim();
    go("analyze");
    updateSearchUI(true);
    focusProductSearch();
  }));
  try {
    const scan = JSON.parse(sessionStorage.getItem("defaketive-home-scan") || "null");
    if (scan && typeof scan === "object" && ["running", "complete", "error", "attention"].includes(scan.state)) state.homeScan = scan;
  } catch (_) {}
  $("#home-scan-action").addEventListener("click", openHomeScan);
  $("#history-search").addEventListener("input", renderWorkspace);
  $("#history-risk").addEventListener("change", renderWorkspace);
  $("#report-back").addEventListener("click", () => { go("analyze"); focusProductSearch(); });
  document.querySelectorAll("[data-limit-dismiss]").forEach(button => button.addEventListener("click", () => {
    try { sessionStorage.setItem("defaketive-limit-dismissed", "1"); } catch (_) {}
    updateLimitNotice();
  }));
  $("#active-scan-link").addEventListener("click", () => {
    if (state.homeScan) openHomeScan();
    else if (state.searchBusy) go("analyze");
    else toast("No scans running.");
  });
  // Home's cards are re-rendered often, so their clicks are handled here once.
  $("#home").addEventListener("click", event => {
    const save = event.target.closest("[data-save-link]"), risk = event.target.closest("[data-show-risk]"), next = event.target.closest("[data-row-next]");
    const open = event.target.closest("[data-product-link]") || event.target.closest(".app-product")?.querySelector("[data-product-link]");
    if (save) toggleWishlist(save.dataset.saveLink);
    else if (risk) showHistoryFilter(risk.dataset.showRisk);
    else if (next) { const track = next.parentElement.querySelector(".app-row-track"); track.scrollBy({left: track.clientWidth * .8, behavior: "smooth"}); }
    else if (open) openProduct(open.dataset.productLink);
  });
  document.querySelectorAll(".app-row-track").forEach(track => track.addEventListener("scroll", () => updateRowArrow(track), {passive: true}));
  const accountToggle = $("[data-account-toggle]");
  if (accountToggle) accountToggle.addEventListener("click", () => {
    const menu = $("#account-menu"), open = menu.hidden;
    menu.hidden = !open;
    accountToggle.setAttribute("aria-expanded", String(open));
    if (open) menu.querySelector("a, button")?.focus();
  });
  window.addEventListener("hashchange", routeFromHash);

  document.querySelectorAll("[data-route]").forEach(link => link.addEventListener("click", event => {
    event.preventDefault();
    go(link.dataset.route);
    if (link.hasAttribute("data-focus-search")) focusProductSearch();
  }));
  document.querySelectorAll("[data-signin]").forEach(button => button.addEventListener("click", openSignedInPage));
  document.querySelectorAll("[data-auth-open]").forEach(button => button.addEventListener("click", () => openAuth(button.dataset.authOpen)));
  document.querySelectorAll("[data-terms-open]").forEach(button => button.addEventListener("click", () => openTerms(button.dataset.termsOpen)));
  document.querySelectorAll("[data-auth-form]").forEach(form => form.addEventListener("submit", submitAuth));
  document.querySelectorAll("[data-continue-guest]").forEach(button => button.addEventListener("click", continueAsGuest));
  document.querySelectorAll("[data-signout]").forEach(button => button.addEventListener("click", signOut));
  document.querySelectorAll("[data-dialog-close]").forEach(button => button.addEventListener("click", () => button.closest("dialog")?.close()));
  // A click on the dimmed backdrop lands on the <dialog> itself, not its card.
  document.querySelectorAll("dialog.auth-dialog").forEach(dialog => dialog.addEventListener("click", event => { if (event.target === dialog) dialog.close(); }));
  document.querySelectorAll("[data-toggle-password]").forEach(button => button.addEventListener("click", () => {
    const input = button.parentElement.querySelector("input");
    const show = input.type === "password";
    input.type = show ? "text" : "password";
    button.setAttribute("aria-pressed", String(show));
    button.setAttribute("aria-label", show ? "Hide password" : "Show password");
  }));

  const searchForm = $("#search-form");
  if (searchForm) searchForm.addEventListener("submit", search);
  const searchQuery = $("#search-query");
  if (searchQuery) searchQuery.addEventListener("input", () => updateSearchUI(true));
  document.querySelectorAll('input[name="marketplace"]').forEach(input => input.addEventListener("change", () => updateSearchUI()));
  ["#product-count", "#review-count"].forEach(id => {
    const node = $(id);
    if (node) node.addEventListener("input", () => updateSearchUI());
  });
  const retryButton = $("#retry-search");
  if (retryButton) retryButton.addEventListener("click", () => {
    clearSearchFeedback();
    if (searchForm) searchForm.requestSubmit();
  });
  const datasetUpload = $("#dataset-upload");
  if (datasetUpload) datasetUpload.addEventListener("change", uploadDataset);
  ["#filter-platform", "#filter-risk", "#filter-sort"].forEach(id => {
    const node = $(id);
    if (node) node.addEventListener("change", renderResults);
  });
  const navToggle = $(".nav-toggle");
  if (navToggle) navToggle.addEventListener("click", event => {
    const nav = $("#main-nav");
    if (!nav) return;
    const open = nav.classList.toggle("open");
    event.currentTarget.setAttribute("aria-expanded", String(open));
    event.currentTarget.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
    if (open) nav.querySelector(".nav-link")?.focus();
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") {
      if ($("#main-nav")?.classList.contains("open")) navToggle?.focus();
      closeMainNavigation();
      closeAccountMenu();
    }
  });
  document.addEventListener("click", event => {
    const header = $(".pub-header");
    if (header && !header.contains(event.target)) closeMainNavigation();
    if (!event.target.closest?.(".app-account")) closeAccountMenu();
  });
  window.addEventListener("scroll", updateHeaderState, {passive: true});
  const desktopNavigation = window.matchMedia?.("(min-width: 761px)");
  desktopNavigation?.addEventListener("change", event => {
    if (event.matches) closeMainNavigation();
  });

  document.querySelectorAll("[data-scroll]").forEach(link => link.addEventListener("click", event => {
    event.preventDefault();
    go("landing");
    closeMainNavigation();
    requestAnimationFrame(() => {
      const section = $(`#${link.dataset.scroll || "how"}`);
      if (section) {
        section.scrollIntoView({behavior: "smooth"});
        const heading = section.querySelector("h2") || section;
        heading.setAttribute("tabindex", "-1");
        heading.focus({preventScroll: true});
      }
      setActiveNavigation("landing", link.dataset.scroll);
    });
  }));

  document.querySelectorAll("[data-focus-search]:not([data-route])").forEach(link => link.addEventListener("click", event => {
    event.preventDefault();
    go("analyze");
    focusProductSearch();
  }));

  document.querySelectorAll("[data-landing-scroll]").forEach(link => link.addEventListener("click", event => {
    event.preventDefault();
    go("landing");
    requestAnimationFrame(() => $(`#${link.dataset.landingScroll}`)?.scrollIntoView({behavior: "smooth"}));
  }));

  const themeToggles = [$("#theme-toggle"), $("#workspace-theme-toggle")].filter(Boolean);
  const syncTheme = theme => {
    document.documentElement.dataset.theme = theme;
    $("#appearance-select").value = theme;
    themeToggles.forEach(button => {
      button.setAttribute("aria-pressed", String(theme === "dark"));
      button.setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} theme`);
      button.setAttribute("title", `Switch to ${theme === "dark" ? "light" : "dark"} theme`);
      button.querySelector("span").textContent = button.id === "workspace-theme-toggle" ? (theme === "dark" ? "Dark" : "Light") : (theme === "dark" ? "Light" : "Dark");
    });
  };
  syncTheme((() => { try { return localStorage.getItem("defaketive-theme"); } catch (_) { return "light"; } })() === "dark" ? "dark" : "light");
  $("#appearance-select").addEventListener("change", event => {
    const theme = event.target.value;
    try { localStorage.setItem("defaketive-theme", theme); } catch (_) {}
    syncTheme(theme);
  });
  themeToggles.forEach(button => button.addEventListener("click", () => {
    const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    try { localStorage.setItem("defaketive-theme", nextTheme); } catch (_) {}
    syncTheme(nextTheme);
  }));
  updateSearchUI();
  updateHeaderState();
}
document.addEventListener("DOMContentLoaded", async () => {
  setup();
  applyAuth(null);
  refreshAuth();
  // Render routes immediately, so a slow catalog request never strands navigation on the landing page.
  const routeReady = routeFromHash();
  try { await loadProducts(); }
  catch (error) {
    // A missing model is not a transient loading problem: say so, and why.
    state.modelUnavailable = error.body?.model_unavailable ? `${error.message} ${error.body.detail || ""}`.trim() : "";
    state.catalogError = true; state.catalogLoading = false; renderWorkspace();
    toast(state.modelUnavailable ? error.message : "Saved results could not be loaded. You can still start a new analysis.", "error");
  }
  await routeReady;
  try {
    const job = JSON.parse(sessionStorage.getItem("defaketive-active-job") || "null");
    if (job?.id && !state.activeJobId) {
      state.activeJobId = job.id;
      $("#search-query").value = job.query || "";
      setSearchBusy(true);
      pollJob(job.id, job);
    }
  } catch (_) { /* Storage may be disabled; the rest of the workspace remains usable. */ }
});
