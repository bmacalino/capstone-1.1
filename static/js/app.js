const state = { homeScan: null, catalogLoading: true, currentProduct: null, reportFrom: "history", activity: {}, wishlist: [], catalogError: false, products: [], catalogProducts: [], toastTimer: null, activeJobId: null, jobStartedAt: null, jobElapsedTimer: null, searchBusy: false };
const MARKETPLACES = { shopee: "Shopee PH", lazada: "Lazada PH", temu: "Temu PH" };
const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
const money = (value) => String(value || "—").replace(/Ã¢â€šÂ±/g, "₱");
const score = (product) => Number(product?.sentiment_summary?.risk_score || 0);
const reliability = (product) => Math.round(Number(product?.reliability_score ?? product?.sentiment_summary?.reliability_score ?? (100 - score(product))) || 0);
const reviewCount = (product) => Number(product?.sentiment_summary?.review_count ?? product?.comments?.length ?? 0) || 0;
function riskLevel(product) { const risk = score(product); return risk <= 30 ? "Low" : risk <= 60 ? "Moderate" : "High"; }
function safeImage(product) { const candidate = product?.img || product?.image || product?.image_url || product?.thumbnail; return /^(https?:\/\/|\/static\/)/i.test(String(candidate || "")) ? candidate : ""; }
function safeLink(value) { return /^https?:\/\//i.test(String(value || "")) ? String(value) : ""; }
function toast(message, kind = "info") { const node = $("#toast"); node.textContent = message; node.dataset.kind = kind; node.hidden = false; clearTimeout(state.toastTimer); state.toastTimer = setTimeout(() => { node.hidden = true; }, 5000); }
async function api(url, options = {}) { const response = await fetch(url, options); const body = await response.json().catch(() => ({})); if (!response.ok) throw new Error(body.error || "Something went wrong."); return body; }
function queryContext(value) { const text = String(value || "").trim(); let parsed; try { parsed = new URL(text); } catch (_) { return { isUrl: false, platform: null }; } if (!/^https?:$/.test(parsed.protocol)) return { isUrl: false, platform: null }; const host = parsed.hostname.toLowerCase(); if (host === "shopee.ph" || host.endsWith(".shopee.ph")) return { isUrl: true, platform: "shopee" }; if (host === "lazada.com.ph" || host.endsWith(".lazada.com.ph")) return { isUrl: true, platform: "lazada" }; if (host === "temu.com" || host.endsWith(".temu.com")) return { isUrl: true, platform: "temu" }; return { isUrl: true, platform: null }; }
function clearSearchFeedback() { const feedback = $("#search-feedback"); feedback.hidden = true; feedback.removeAttribute("data-kind"); }
function showSearchFeedback(message, title = "Scan could not start", kind = "error") { $("#search-feedback-title").textContent = title; $("#search-feedback-message").textContent = message; const feedback = $("#search-feedback"); feedback.dataset.kind = kind; feedback.hidden = false; }
function selectedMarketplace() { return document.querySelector('input[name="marketplace"]:checked')?.value || "shopee"; }
function selectMarketplace(value) { const input = document.querySelector(`input[name="marketplace"][value="${value}"]`); if (input) input.checked = true; }
function productLinkIdentity(value) { try { const url = new URL(String(value || "").trim()); return `${url.hostname.toLowerCase().replace(/^www\./, "")}${url.pathname.replace(/\/+$/, "").toLowerCase()}`; } catch (_) { return ""; } }
function savedProductForQuery(value) { const identity = productLinkIdentity(value); if (!identity) return null; const catalog = state.catalogProducts.length ? state.catalogProducts : state.products; return catalog.find(product => productLinkIdentity(product.link) === identity) || null; }
function setSearchBusy(busy) {
  state.searchBusy = busy;
  $("#active-scan-link").hidden = !busy;
  $("#active-scan-link").textContent = busy ? "Scan in progress →" : "Return to scan →";
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
  $("#review-count").disabled = state.searchBusy;
  document.querySelectorAll('input[name="marketplace"]').forEach(input => { input.disabled = state.searchBusy; });

  if (!hasQuery || unsupported) {
    summary.hidden = true;
  } else {
    const marketplace = context.platform || selectedMarketplace();
    const products = context.platform ? 1 : Number(productCount.value || 1);
    const reviews = Number($("#review-count").value || 1);
    summary.textContent = `Fresh collection settings: ${MARKETPLACES[marketplace]} · up to ${products} product${products === 1 ? "" : "s"} · up to ${reviews} reviews each`;
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
  const header = $(".site-header");
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
function closeWorkspaceNavigation() {
  $("#workspace-shell").classList.remove("sidebar-open");
  $(".workspace-mobile-menu").setAttribute("aria-expanded", "false");
  $(".workspace-backdrop").hidden = true;
}
function go(view, replace = false, preserveUrl = false) {
  const previousView = currentView();
  if (view === "dashboard") view = "home";
  if (view === "admin") view = "history";
  if (view !== "landing" && !WORKSPACE_ROUTES.includes(view)) view = "home";
  document.body.dataset.activeView = view;
  document.querySelectorAll("[data-view]").forEach(node => {
    const visible = node.dataset.view === view;
    node.hidden = !visible;
    node.classList.toggle("active", visible);
  });
  $("#workspace-shell").hidden = view === "landing";
  $("#workspace-page-label").textContent = ROUTE_LABELS[view] || "Home";
  let hash = view;
  if (view === "analysis" && state.currentProduct?.link) {
    hash += "?" + new URLSearchParams({product: state.currentProduct.link, from: state.reportFrom});
  }
  if (!preserveUrl && window.location.hash !== "#" + hash) {
    window.history[replace ? "replaceState" : "pushState"](null, "", "#" + hash);
  }
  const activeRoute = view === "analysis" ? state.reportFrom : view === "results" ? "analyze" : view;
  document.querySelectorAll(".workspace-nav [data-route]").forEach(link => {
    const active = link.dataset.route === activeRoute;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  setActiveNavigation(view);
  closeMainNavigation();
  closeWorkspaceNavigation();
  if (view === "results") renderResults();
  if (view === "home" || view === "history" || view === "wishlist") renderWorkspace();
  if (view === "analysis" && !state.currentProduct) {
    $("#analysis-content").innerHTML = '<div class="flow-empty"><h1>Choose a product report</h1><p>Open an analysis from History, or analyze a product to get started.</p><a href="#history">View history →</a></div>';
    $("#report-save").hidden = true;
  }
  $("#report-back").textContent = "← Back to " + (ROUTE_LABELS[state.reportFrom] || "History").toLowerCase();
  document.title = (view === "landing" ? "DeFaketive — Review Risk Intelligence" : ROUTE_LABELS[view] + " · DeFaketive");
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
  syncReportSave();
  renderWorkspace();
  toast(saved ? "Removed from wishlist." : "Saved to your wishlist on this browser.");
}
function syncReportSave() {
  const button = $("#report-save");
  const link = safeLink(state.currentProduct?.link);
  const saved = state.wishlist.includes(link);
  button.hidden = !link;
  button.setAttribute("aria-pressed", String(saved));
  button.textContent = saved ? "♥ Saved to wishlist" : "♡ Save to wishlist";
}
function reportDate(product) {
  const value = product.analysis_date || product.analyzed_at || product.updated_at;
  const date = value && !Number.isNaN(Date.parse(value)) ? new Date(value) : null;
  return date ? {iso: date.toISOString(), label: new Intl.DateTimeFormat("en-PH", {month: "short", day: "numeric", year: "numeric"}).format(date)} : null;
}
function workspaceRows(products, emptyTitle, emptyCopy) {
  if (!products.length) return '<div class="flow-empty"><h3>' + esc(emptyTitle) + '</h3><p>' + esc(emptyCopy) + '</p><a href="#analyze">Analyze a product →</a></div>';
  return '<div class="flow-report-list">' + products.map(product => {
    const level = riskLevel(product), saved = state.wishlist.includes(product.link), image = safeImage(product), date = reportDate(product);
    return `<article class="flow-report-row"><div class="flow-product-thumb">${image ? `<img src="${esc(image)}" alt="" loading="lazy">` : '<span aria-hidden="true">▤</span>'}</div><div class="flow-product-info"><h3 title="${esc(product.name || "Unnamed product")}">${esc(product.name || "Unnamed product")}</h3><p><span class="flow-market-label">${esc(platform(product))}</span><span>${reviewCount(product).toLocaleString()} reviews</span><span class="flow-analysis-date">${date ? `Analyzed <time datetime="${esc(date.iso)}">${esc(date.label)}</time>` : "Analysis date unavailable"}</span></p></div><span class="risk-badge ${level.toLowerCase()}">${level} risk · ${(score(product) / 100).toFixed(2)}</span><div class="flow-row-actions"><button type="button" class="flow-save" data-save-link="${esc(product.link)}" aria-label="${saved ? "Remove from" : "Save to"} wishlist: ${esc(product.name)}" aria-pressed="${saved}">${saved ? "♥" : "♡"}</button><button type="button" class="flow-open" data-product-link="${esc(product.link)}">View report →</button></div></article>`;
  }).join("") + "</div>";
}
function persistHomeScan(scan) {
  state.homeScan = scan;
  try { sessionStorage.setItem("defaketive-home-scan", JSON.stringify(scan)); } catch (_) {}
  renderHomeScan();
}
function renderHomeScan() {
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
function renderHomeProducts(products, highRisk) {
  renderHomeScan();
  const lastOpened = Object.values(state.activity).filter(entry => entry.openedAt > 0 && safeLink(entry.product?.link)).sort((a, b) => b.openedAt - a.openedAt)[0];
  $("#home-continue").hidden = !lastOpened;
  $("#home-continue").dataset.productLink = lastOpened?.product.link || "";
  $("#home-continue-name").textContent = lastOpened?.product.name || "Your last analysis";
  $("#home-continue").title = lastOpened?.product.name || "";
  $("#home-recent").setAttribute("aria-busy", String(state.catalogLoading));
  $("#home-wishlist-preview").setAttribute("aria-busy", String(state.catalogLoading));
  if (state.catalogLoading && !products.length) {
    $("#home-metrics").innerHTML = '<span class="home-list-caption">Loading your activity…</span>';
    return;
  }
  $("#home-metrics").innerHTML = [
    [products.length, "Saved analyses"],
    [products.reduce((total, product) => total + reviewCount(product), 0), "Reviews analyzed"],
    [state.wishlist.length, "In your wishlist"],
  ].map(([value, label]) => `<article><b>${value.toLocaleString()}</b><span>${label}</span></article>`).join("");
  const query = $("#home-recent-search").value.trim().toLowerCase();
  const matched = products.filter(product => !query || `${product.name} ${platform(product)}`.toLowerCase().includes(query));
  $("#home-clear-search").hidden = !query;
  $("#home-recent-count").textContent = query ? `${matched.length} matching report${matched.length === 1 ? "" : "s"} · showing ${Math.min(4, matched.length)}` : products.length ? `Showing ${Math.min(4, products.length)} recent reports · ${products.length} saved` : "";
  $("#home-recent").innerHTML = workspaceRows(matched.slice(0, 4),
    query ? "No matching analyses" : state.catalogError ? "Saved reports could not be loaded" : "Your first analysis starts here",
    query ? "Try another name or marketplace, or clear your search." : state.catalogError ? "Reload the page to try again, or start a new analysis above." : "Paste a product link above. Your completed report will appear here.");
  const byLink = new Map(products.map(product => [product.link, product]));
  const saved = [...state.wishlist].reverse().map(link => byLink.get(link)).filter(Boolean);
  $("#home-wishlist-count").textContent = String(saved.length);
  $("#home-wishlist-preview").innerHTML = saved.length ? saved.slice(0, 3).map(product => {
    const image = safeImage(product), level = riskLevel(product);
    return `<article class="home-saved-product"><div class="home-saved-thumb">${image ? `<img src="${esc(image)}" alt="" loading="lazy">` : '<img src="/static/assets/figma/home-report.svg" width="20" height="20" alt="">'}</div><div><button type="button" data-product-link="${esc(product.link)}">${esc(product.name || "Unnamed product")}</button><p>${esc(platform(product))}</p><span class="risk-badge ${level.toLowerCase()}">${level} risk</span></div></article>`;
  }).join("") : '<div class="home-wishlist-empty"><span aria-hidden="true">♡</span><h3>Keep your options together</h3><p>Save a product using the heart beside its report. Your shortlist will appear here.</p></div>';
  $("#home-risk-alert").innerHTML = highRisk.length ? `<header><div><p class="flow-eyebrow">BEFORE YOU BUY</p><h2>Worth a closer look</h2><p>High risk signals in your saved analyses.</p></div><button class="home-text-button" type="button" data-show-risk>View all ${highRisk.length} →</button></header><div class="home-risk-products">${highRisk.slice(0, 2).map(product => {
    const concern = analysisMainIssue(product.sentiment_summary || {});
    return `<article><span class="risk-badge high">High risk · ${(score(product) / 100).toFixed(2)}</span><h3>${esc(product.name || "Unnamed product")}</h3><p>${concern === "No dominant issue" ? "High measured risk across " + reviewCount(product) + " sampled reviews. Open the report to inspect the evidence." : "Most reported concern: " + esc(concern) + ". Check what reviewers actually said."}</p><button type="button" class="home-text-button" data-home-evidence="${esc(product.link)}">Read the evidence →</button></article>`;
  }).join("")}</div>` : "";
  bindProductButtons($("#home-wishlist-preview"));
  $("#home-risk-alert").querySelectorAll("[data-home-evidence]").forEach(button => button.addEventListener("click", async () => {
    const link = button.dataset.homeEvidence;
    await openProduct(link);
    if (currentView() !== "analysis" || state.currentProduct?.link !== link) return;
    document.querySelector('[data-review-filter="risk"]')?.click();
    const section = $("#report-evidence");
    section?.scrollIntoView({behavior: "smooth", block: "start"});
    section?.focus({preventScroll: true});
  }));
}
function renderWorkspace() {
  const products = workspaceProducts();
  const highRisk = products.filter(product => riskLevel(product) === "High");
  renderHomeProducts(products, highRisk);
  const query = $("#history-search").value.trim().toLowerCase(), risk = $("#history-risk").value;
  const filtered = products.filter(product => (!query || `${product.name} ${platform(product)}`.toLowerCase().includes(query)) && (risk === "all" || riskLevel(product) === risk));
  $("#history-count").textContent = `${filtered.length} of ${products.length} saved reports`;
  $("#history-list").innerHTML = workspaceRows(filtered, products.length ? "No reports match these filters" : "No analyses yet", products.length ? "Try another product name or select all risk levels." : "Analyze a product to start your history.");
  $("#wishlist-list").innerHTML = workspaceRows(products.filter(product => state.wishlist.includes(product.link)), "Keep a product in mind", "Use the heart beside a saved analysis or save a product from its report. It will appear here.");
  ["#home-recent", "#history-list", "#wishlist-list"].forEach(selector => {
    const holder = $(selector);
    bindProductButtons(holder);
    holder.querySelectorAll("[data-save-link]").forEach(button => button.addEventListener("click", () => toggleWishlist(button.dataset.saveLink)));
  });
  $("[data-show-risk]")?.addEventListener("click", () => {
    $("#history-search").value = "";
    $("#history-risk").value = "High";
    go("history");
  });
}
async function routeFromHash() {
  const [route, query = ""] = window.location.hash.slice(1).split("?");
  const params = new URLSearchParams(query);
  if (["about", "faq", "how"].includes(route)) {
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
function productCard(product) { const level = riskLevel(product); const image = safeImage(product); const link = safeLink(product.link); return `<article class="product-card"><div class="product-image">${image ? `<img src="${esc(image)}" alt="">` : "Listing image unavailable"}</div><div class="product-card-body"><span class="platform-badge">${esc(platform(product))}</span><h3>${esc(product.name || "Unnamed product")}</h3><div class="product-price">${esc(money(product.price))}</div><div class="card-meta">★ ${esc(product.rating || "—")} · ${reviewCount(product).toLocaleString()} reviews analyzed</div><span class="risk-badge ${level.toLowerCase()}">${level} risk · ${score(product).toFixed(0)}% score</span><div class="reliability"><i style="width:${Math.min(100, Math.max(0, reliability(product)))}%"></i></div><div class="card-meta">Review reliability: ${reliability(product)}%</div><button class="card-button" data-product-link="${esc(link)}">Open report →</button></div></article>`; }
function bindProductButtons(root = document) {
  root.querySelectorAll("[data-product-link]").forEach(button => button.addEventListener("click", () => openProduct(button.dataset.productLink)));
}
function filteredProducts() { const platformFilter = $("#filter-platform")?.value || "all"; const riskFilter = $("#filter-risk")?.value || "all"; const sort = $("#filter-sort")?.value || "all"; const visible = state.products.filter(product => (platformFilter === "all" || platform(product) === platformFilter) && (riskFilter === "all" || riskLevel(product) === riskFilter)); return visible.sort((a,b) => { if (sort === "risk") return score(a)-score(b); if (sort === "reviews") return reviewCount(b)-reviewCount(a); return reliability(b)-reliability(a); }); }
function renderResults() { const products = filteredProducts(); $("#results-summary").textContent = products.length ? `${products.length} analyzed product${products.length === 1 ? "" : "s"} shown.` : "No products match your selected filters."; $("#product-grid").innerHTML = products.map(productCard).join(""); bindProductButtons($("#product-grid")); }
async function loadProducts() { const data = await api("/api/products"); state.products = data.products || []; state.catalogProducts = [...state.products]; state.catalogError = false; state.catalogLoading = false; cacheWorkspaceProducts(state.catalogProducts); if ($("#search-form")) updateSearchUI(); renderWorkspace(); }
function highlightReview(review) { const analysis = review.sentiment_analysis || {}; const terms = (analysis.risk?.evidence || []).filter(item => !item.negated).map(item => item.term).filter(Boolean); let content = esc(review.content || "No written review was collected."); terms.forEach(term => { const escaped = esc(term).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); if (escaped) content = content.replace(new RegExp(`(${escaped})`, "ig"), "<mark>$1</mark>"); }); return content; }
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
  go("home");
}
function sentimentData(product) { const counts = product.sentiment_summary?.sentiment_counts || {}; return [{label:"Positive", value:Number(counts.positive || 0), color:"#12b76a"},{label:"Neutral", value:Number(counts.neutral || 0), color:"#94a3b8"},{label:"Negative", value:Number(counts.negative || 0), color:"#f04438"}]; }
function analysisMainIssue(summary) {
  const issues = Object.entries(summary.risk_categories || {}).filter(([, count]) => Number(count) > 0).sort((a, b) => Number(b[1]) - Number(a[1]));
  return issues.length ? String(issues[0][0]).replaceAll("_", " ").replace(/\b\w/g, letter => letter.toUpperCase()) : "No dominant issue";
}

function sentimentStack(product) {
  const data = sentimentData(product);
  const total = data.reduce((sum, item) => sum + item.value, 0);
  if (!total) return '<p class="analysis-empty">No sentiment labels are available.</p>';
  const segments = data.map(item => `<span class="sentiment-segment ${item.label.toLowerCase()}" style="width:${item.value / total * 100}%" title="${item.label}: ${item.value}" aria-label="${item.label}: ${item.value} reviews"></span>`).join("");
  const legend = data.map(item => `<div><i class="${item.label.toLowerCase()}"></i><span>${item.label}</span><b>${item.value}</b><small>${Math.round(item.value / total * 100)}%</small></div>`).join("");
  return `<div class="sentiment-stack" role="img" aria-label="Sentiment distribution across ${total} reviews">${segments}</div><div class="sentiment-legend">${legend}</div>`;
}

function issueBars(summary, reviewTotal) {
  const issues = Object.entries(summary.risk_categories || {}).sort((a, b) => Number(b[1]) - Number(a[1])).slice(0, 6);
  if (!issues.length) return '<p class="analysis-empty">No active product-risk issues were detected in this review sample.</p>';
  const highest = Math.max(...issues.map(([, count]) => Number(count || 0)), 1);
  return `<div class="issue-bars">${issues.map(([issue, count]) => {
    const label = String(issue).replaceAll("_", " ").replace(/\b\w/g, letter => letter.toUpperCase());
    const numericCount = Number(count || 0);
    const percentage = reviewTotal ? Math.round(numericCount / reviewTotal * 100) : 0;
    return `<div class="issue-row"><div><span>${esc(label)}</span><b>${numericCount} review${numericCount === 1 ? "" : "s"} · ${percentage}%</b></div><div class="issue-track"><i style="width:${numericCount / highest * 100}%"></i></div></div>`;
  }).join("")}</div>`;
}

const REPORT_ASSET_BASE = "/static/assets/figma/report";
function reportIcon(file, alt = "", width = 16, height = width, className = "") {
  return `<img${className ? ` class="${esc(className)}"` : ""} src="${REPORT_ASSET_BASE}/${esc(file)}" width="${width}" height="${height}" alt="${esc(alt)}">`;
}

function starMarkup(value) {
  const rating = Math.max(0, Math.min(5, Number(value || 0)));
  const stars = [1, 2, 3, 4, 5].map(position => {
    const icon = rating >= position - .25 ? "icon-07.svg" : rating >= position - .75 ? "icon-08.svg" : "icon-02.svg";
    return reportIcon(icon, "", 20, 20);
  }).join("");
  return `<span class="report-stars" role="img" aria-label="${rating.toFixed(1)} out of 5 stars">${stars}</span>`;
}

function sentimentPercentages(product) {
  const data = sentimentData(product);
  const total = data.reduce((sum, item) => sum + item.value, 0);
  return data.map(item => ({...item, percent: total ? item.value / total * 100 : 0}));
}

function reportKeywords(reviews, summary) {
  const terms = new Map();
  reviews.forEach(review => {
    (review.sentiment_analysis?.risk?.evidence || []).filter(item => !item.negated && item.term).forEach(item => {
      const key = String(item.term).trim().toLowerCase();
      if (key) terms.set(key, (terms.get(key) || 0) + 1);
    });
  });
  if (!terms.size) Object.entries(summary.risk_categories || {}).forEach(([term, count]) => terms.set(String(term).replaceAll("_", " "), Number(count || 0)));
  const ranked = [...terms.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
  return ranked.length
    ? ranked.map(([term, count]) => `<span class="report-keyword"><b>${esc(term.replace(/\b\w/g, letter => letter.toUpperCase()))}</b><small>${count} mention${count === 1 ? "" : "s"}</small></span>`).join("")
    : '<p class="report-empty-copy">No active risk terms were detected.</p>';
}

function riskRecommendation(level) {
  if (level === "Low") return {title: "Generally safe to buy", copy: "Review signals indicate a low likelihood of product-quality issues."};
  if (level === "Moderate") return {title: "Buy with caution", copy: "Compare recent reviews and seller policies before purchasing."};
  return {title: "Not recommended", copy: "Strong review signals suggest choosing a lower-risk alternative."};
}

function reviewCard(review) {
  const analysis = review.sentiment_analysis || {};
  const label = String(analysis.label || "unanalyzed").toLowerCase();
  const activeRisk = Boolean(analysis.risk?.detected);
  const stars = Math.max(0, Math.min(5, Number(review.rating || 0)));
  const meta = [review.time || review.date, review.variation].filter(Boolean).map(esc).join(" · ");
  const evidence = (analysis.risk?.evidence || []).filter(item => !item.negated && item.term).slice(0, 3);
  return `<article class="report-review-card"><div class="report-review-head"><div>${starMarkup(stars)}<small>${stars ? stars.toFixed(1) : "No rating"}</small></div><span class="sentiment-pill ${esc(label)}">${esc(label)}</span></div><p>${highlightReview(review)}</p>${evidence.length ? `<div class="report-evidence-tags">${evidence.map(item => `<span>${esc(item.term)}</span>`).join("")}</div>` : ""}<footer><b>${esc(review.author || "Verified shopper")}</b>${meta ? `<span>${meta}</span>` : ""}${activeRisk ? '<span class="report-risk-evidence">Risk evidence</span>' : ""}</footer></article>`;
}

function renderAnalysis(product, recommendations) {
  state.currentProduct = product;
  rememberProduct(product);
  syncReportSave();
  const summary = product.sentiment_summary || {};
  const allReviews = Array.isArray(product.comments) ? product.comments.filter(item => item && typeof item === "object") : [];
  const reviews = allReviews.filter(item => !item.is_duplicate);
  const uniqueCount = Number(summary.review_count || reviews.length) || reviews.length;
  const duplicateCount = Number(summary.duplicate_review_count ?? allReviews.filter(item => item.is_duplicate).length) || 0;
  const riskReviewCount = Number(summary.risk_review_count || reviews.filter(item => item.sentiment_analysis?.risk?.detected).length) || 0;
  const negativeCount = Number(summary.sentiment_counts?.negative || reviews.filter(item => item.sentiment_analysis?.label === "negative").length) || 0;
  const negativeRate = uniqueCount ? negativeCount / uniqueCount : 0;
  const riskReviewRate = uniqueCount ? riskReviewCount / uniqueCount : Number(summary.keyword_failure_rate || 0);
  const negativePoints = negativeRate * 30;
  const riskPoints = riskReviewRate * 70;
  const productRisk = score(product);
  const productReliability = reliability(product);
  const productRiskLevel = riskLevel(product);
  const riskClass = productRiskLevel.toLowerCase();
  const mainIssue = analysisMainIssue(summary);
  const image = safeImage(product);
  const link = safeLink(product.link);
  const customerRating = Math.max(0, Math.min(5, Number(product.rating || 0)));
  const adjustedRating = Math.max(0, Math.min(5, Number(product.adjusted_rating ?? product.authenticity_adjusted_rating ?? product.rating ?? 0)));
  const sentiment = sentimentPercentages(product);
  const positive = sentiment.find(item => item.label === "Positive") || {value: 0, percent: 0};
  const neutral = sentiment.find(item => item.label === "Neutral") || {value: 0, percent: 0};
  const negative = sentiment.find(item => item.label === "Negative") || {value: 0, percent: 0};
  const saferRecommendations = recommendations.slice(0, 3);
  const recommendationMarkup = saferRecommendations.length
    ? `<div class="report-alternative-grid">${saferRecommendations.map(item => {
        const itemLevel = riskLevel(item);
        const itemImage = safeImage(item);
        const itemSentiment = sentimentPercentages(item).find(entry => entry.label === "Positive")?.percent || Math.max(0, 100 - score(item));
        const isSafer = score(item) < productRisk;
        const statusIcon = isSafer ? "icon-14.svg" : "icon-13.svg";
        const statusLabel = isSafer ? "Recommended" : "Consider";
        return `<article class="report-alternative"><header><div class="report-alt-image">${itemImage ? `<img src="${esc(itemImage)}" width="64" height="64" alt="">` : '<span>No image</span>'}</div><div><h4>${esc(item.name || "Unnamed product")}</h4><div class="report-alt-rating">${starMarkup(item.rating)}<span>${Number(item.rating || 0).toFixed(1)}</span></div></div></header><dl><div><dt>Positive</dt><dd>${itemSentiment.toFixed(0)}%</dd></div><div><dt>Risk</dt><dd>${score(item).toFixed(0)}%</dd></div><div><dt>Reliability</dt><dd>${reliability(item).toFixed(0)}%</dd></div></dl><button class="recommendation-open report-alt-action" data-product-link="${esc(safeLink(item.link))}">${reportIcon(statusIcon, "", 14, 14)}${statusLabel}<span>View report →</span></button></article>`;
      }).join("")}</div>`
    : '<p class="report-empty-copy">No sufficiently similar product with a lower measured risk is available in the analyzed collection.</p>';
  const defaultReviewFilter = negativeCount ? "negative" : "all";
  const brand = product.brand || product.seller || "Not specified";
  const category = product.category || "Not specified";
  const date = reportDate(product);
  const formattedDate = date?.label || "Unavailable";
  const findings = [
    {title: "Risk signals", value: uniqueCount ? `${riskReviewCount} of ${uniqueCount} reviews` : "No review sample", copy: uniqueCount ? "Contain recorded product-risk signals." : "There are no sampled reviews to assess."},
    {title: "Most reported concern", value: mainIssue === "No dominant issue" ? "None recorded" : mainIssue, copy: mainIssue === "No dominant issue" ? "No leading concern is recorded in this sample." : "The most frequent issue in the analyzed sample."},
    {title: "Negative experiences", value: uniqueCount ? `${negativeCount} of ${uniqueCount} reviews` : "Not available", copy: uniqueCount ? "Were classified as negative. Read the context below." : "Sentiment needs a usable review sample."},
  ];
  $("#analysis-content").innerHTML = `<section class="report-card report-overview">
    <div class="report-overview-product"><div class="report-product-image">${image ? `<img src="${esc(image)}" width="112" height="112" alt="">` : '<span>Listing image unavailable</span>'}</div><div><p class="report-overview-meta"><span>${esc(platform(product))}</span><span>${esc(money(product.price))}</span></p><h1>${esc(product.name || "Unnamed product")}</h1><p class="report-sample-note">${uniqueCount.toLocaleString()} reviews analyzed <span aria-hidden="true">·</span> ${date ? `<time datetime="${esc(date.iso)}">${esc(date.label)}</time>` : "Analysis date unavailable"}</p></div></div>
    <div class="report-verdict ${uniqueCount ? riskClass : "unknown"}"><span class="report-verdict-label">${uniqueCount ? productRiskLevel + " risk" : "Insufficient review data"}</span><strong>${uniqueCount ? (productRisk / 100).toFixed(2) : "—"}<small>${uniqueCount ? "/ 1.00" : ""}</small></strong><p>${uniqueCount ? "Lower is better. Based on the sampled reviews." : "Analyze reviews before making a risk assessment."}</p></div>
    <div class="report-overview-bottom"><p>Review signals are a guide, not proof of authenticity.</p><div><button type="button" data-report-jump="report-evidence">Read the evidence ↓</button><button type="button" data-report-jump="report-details">How is this calculated?</button></div></div>
  </section>
  <section class="report-findings" aria-labelledby="report-findings-title"><header><span class="report-eyebrow">AT A GLANCE</span><h2 id="report-findings-title">Three things to know</h2></header><div>${findings.map((finding, index) => `<article><span class="report-finding-index">0${index + 1}</span><h3>${esc(finding.title)}</h3><strong>${esc(finding.value)}</strong><p>${esc(finding.copy)}</p></article>`).join("")}</div></section>
  <section id="report-evidence" class="report-card report-evidence-panel" tabindex="-1"><header class="report-section-head"><div><span class="report-eyebrow">Review evidence</span><h2>Top Negative Reviews</h2><p>Risk terms are highlighted only when the model recorded active evidence.</p></div><span id="review-result-count" class="report-result-count"></span></header><div class="report-tools"><div class="review-filter-chips" role="group" aria-label="Filter reviews"><button class="${defaultReviewFilter === "all" ? "active" : ""}" data-review-filter="all">All</button><button data-review-filter="risk">Risk evidence</button><button class="${defaultReviewFilter === "negative" ? "active" : ""}" data-review-filter="negative">Negative</button><button data-review-filter="neutral">Neutral</button><button data-review-filter="positive">Positive</button></div><div class="review-search-row"><label>Search reviews<input id="review-search" type="search" placeholder="Keyword or reviewer"></label><label>Rating<select id="review-rating"><option value="all">All ratings</option><option value="5">5 stars</option><option value="4">4 stars</option><option value="3">3 stars</option><option value="2">2 stars</option><option value="1">1 star</option></select></label></div></div><div id="analysis-review-list" class="report-review-grid"></div><button id="show-more-reviews" class="report-show-more" type="button">${reportIcon("icon-05.svg", "", 16, 16)}View more reviews</button></section>

  <details id="report-details" class="report-details"><summary><span><b>Score details &amp; methodology</b><small>Ratings, review breakdown, and how the score is calculated</small></span><span class="report-details-chevron" aria-hidden="true">＋</span></summary><div class="report-details-body">
    <section class="report-card report-listing-details"><header><h2>Listing details</h2></header><dl class="report-product-meta"><div><dt>Brand</dt><dd>${esc(brand)}</dd></div><div><dt>Category</dt><dd>${esc(category)}</dd></div></dl><div class="report-ratings"><article><span>Customer rating</span><div>${starMarkup(customerRating)}<b>${customerRating ? customerRating.toFixed(1) : "—"}</b></div><small>From ${uniqueCount.toLocaleString()} sampled reviews</small></article><article><span>Adjusted rating</span><div>${starMarkup(adjustedRating)}<b>${adjustedRating ? adjustedRating.toFixed(1) : "—"}</b></div><small>After review-risk adjustment</small></article></div>${link ? `<a class="report-listing-link" target="_blank" rel="noreferrer" href="${esc(link)}">View original listing ↗</a>` : ""}</section>
  <section class="report-card report-classification"><div><span class="report-eyebrow">Risk classification</span><h2>${productRiskLevel} risk detected</h2><p>Defaketive measured a <b>${productRisk.toFixed(1)}% weighted risk score</b> across ${uniqueCount.toLocaleString()} unique reviews. ${mainIssue === "No dominant issue" ? "No dominant issue was recorded." : esc(mainIssue) + " is the most frequent reported issue."}</p><div class="report-key-stats"><span>${reportIcon("icon-09.svg", "", 14, 14)} ${riskReviewCount} reviews containing risk signals</span><span>${reportIcon("icon-11.svg", "", 14, 14)} ${duplicateCount} Duplicates excluded</span></div></div><div class="report-thresholds"><table><thead><tr><th>Score</th><th>Classification</th><th>Guidance</th></tr></thead><tbody><tr class="${productRiskLevel === "Low" ? "active" : ""}"><td>0.00–0.30</td><td><span class="low">Low risk</span></td><td>Fewer measured risk signals</td></tr><tr class="${productRiskLevel === "Moderate" ? "active" : ""}"><td>0.31–0.60</td><td><span class="moderate">Moderate risk</span></td><td>Review with caution</td></tr><tr class="${productRiskLevel === "High" ? "active" : ""}"><td>0.61–1.00</td><td><span class="high">High risk</span></td><td>Consider alternatives</td></tr></tbody></table><div class="report-model-note"><b>How the score is calculated</b><span>30% negative sentiment + 70% active product-risk signals. Higher is better applies to the separate ${productReliability.toFixed(0)}% reliability score.</span></div></div></section>
  <section class="report-card report-breakdown"><header><span class="report-eyebrow">Analysis breakdown</span><h2>Review Analysis Summary</h2><p>Signals from unique reviews after duplicate exclusion.</p></header><div class="report-breakdown-grid"><article><h3>Sentiment distribution</h3><div class="report-donut-row"><div class="report-donut" style="--positive-stop:${positive.percent}%;--neutral-stop:${positive.percent + neutral.percent}%">${reportIcon("icon-04.svg", "", 120, 120, "report-donut-art")}<span><b>${uniqueCount}</b><small>reviews</small></span></div><div class="report-sentiment-legend">${sentiment.map(item => `<p class="${item.label.toLowerCase()}"><i></i><span>${item.label}</span><b>${item.percent.toFixed(0)}%</b><small>${item.value}</small></p>`).join("")}</div></div></article><article><h3>Risk score breakdown</h3><div class="report-score-lines"><p><span>Negative-review rate</span><b>${(negativeRate * 100).toFixed(1)}%</b><i><em style="width:${negativeRate * 100}%"></em></i><small>Contributes ${negativePoints.toFixed(1)} points</small></p><p><span>Reviews containing risk signals</span><b>${(riskReviewRate * 100).toFixed(1)}%</b><i><em style="width:${riskReviewRate * 100}%"></em></i><small>Contributes ${riskPoints.toFixed(1)} points</small></p></div><footer><span>Final weighted risk</span><b>${productRisk.toFixed(1)}%</b></footer></article><article><h3>Detected keywords</h3><div class="report-keywords">${reportKeywords(reviews, summary)}</div><small class="report-card-note">Top reported issues from active, non-negated evidence.</small></article><article><h3>Aspects analyzed</h3><div class="report-aspects"><p>${reportIcon("icon-09.svg", "", 14, 14)}<span><b>Durability</b><small>Breakage and wear</small></span></p><p>${reportIcon("icon-10.svg", "", 14, 14)}<span><b>Quality</b><small>Materials and finish</small></span></p><p>${reportIcon("icon-11.svg", "", 14, 14)}<span><b>Performance</b><small>Function and reliability</small></span></p><p>${reportIcon("icon-14.svg", "", 14, 14)}<span><b>Overall</b><small>Sentiment and value</small></span></p></div></article></div></section>
  </div></details>
  <section class="report-card report-alternatives"><header class="report-section-head"><div><span class="report-eyebrow">Safer choices</span><h2>Recommended Alternatives</h2><p>Similar analyzed products with a lower measured risk score.</p></div></header>${recommendationMarkup}</section><p class="analysis-disclaimer report-disclaimer">This report analyzes a sample of public reviews. It supports shopping decisions but does not prove that a listing or product is authentic or counterfeit.</p>`;

  document.querySelectorAll("[data-report-jump]").forEach(button => button.addEventListener("click", () => {
    const section = document.getElementById(button.dataset.reportJump);
    if (!section) return;
    if (section.tagName === "DETAILS") section.open = true;
    section.scrollIntoView({behavior: "smooth", block: "start"});
    (section.querySelector("summary") || section).focus({preventScroll: true});
  }));
  const dateNode = $("#report-analysis-date");
  if (dateNode) dateNode.textContent = `Analysis date: ${formattedDate}`;
  document.querySelectorAll(".recommendation-open").forEach(button => button.addEventListener("click", () => openProduct(button.dataset.productLink)));

  let activeFilter = defaultReviewFilter;
  let visibleLimit = 3;
  const drawReviews = () => {
    const query = $("#review-search").value.trim().toLowerCase();
    const rating = $("#review-rating").value;
    const filtered = reviews.filter(review => {
      const analysis = review.sentiment_analysis || {};
      const label = String(analysis.label || "unanalyzed").toLowerCase();
      const matchesType = activeFilter === "all" || (activeFilter === "risk" ? Boolean(analysis.risk?.detected) : label === activeFilter);
      const matchesRating = rating === "all" || Number(review.rating || 0) === Number(rating);
      const searchable = `${review.author || ""} ${review.content || ""} ${review.variation || ""}`.toLowerCase();
      return matchesType && matchesRating && (!query || searchable.includes(query));
    });
    const visible = filtered.slice(0, visibleLimit);
    $("#review-result-count").textContent = `${filtered.length} review${filtered.length === 1 ? "" : "s"}`;
    $("#analysis-review-list").innerHTML = visible.length ? visible.map(reviewCard).join("") : '<p class="report-empty-copy">No reviews match these filters.</p>';
    const showMore = $("#show-more-reviews");
    showMore.hidden = visible.length >= filtered.length;
    showMore.innerHTML = `${reportIcon("icon-05.svg", "", 16, 16)}View more reviews (${filtered.length - visible.length})`;
  };
  document.querySelectorAll("[data-review-filter]").forEach(button => button.addEventListener("click", () => {
    activeFilter = button.dataset.reviewFilter;
    visibleLimit = 3;
    document.querySelectorAll("[data-review-filter]").forEach(item => item.classList.toggle("active", item === button));
    drawReviews();
  }));
  $("#review-search").addEventListener("input", () => { visibleLimit = 3; drawReviews(); });
  $("#review-rating").addEventListener("change", () => { visibleLimit = 3; drawReviews(); });
  $("#show-more-reviews").addEventListener("click", () => { visibleLimit += 6; drawReviews(); });
  drawReviews();
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
    showSearchFeedback(error.message);
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
        $("#active-scan-link").hidden = false;
        $("#active-scan-link").textContent = "Scan needs attention →";
        return;
      }
      persistHomeScan(job);
      setTimeout(tick, 2500);
    } catch (error) {
      if (state.pollToken !== pollToken) return;
      persistHomeScan({...state.homeScan, state: "attention", message: error.message});
      showSearchFeedback(error.message, "Could not read scan progress");
      $("#active-scan-link").hidden = false;
      $("#active-scan-link").textContent = "Check scan progress →";
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
  $("#home-continue").addEventListener("click", () => openProduct($("#home-continue").dataset.productLink));
  $("#home-scan-action").addEventListener("click", openHomeScan);
  $("#home-recent-search").addEventListener("input", renderWorkspace);
  $("#home-clear-search").addEventListener("click", () => {
    $("#home-recent-search").value = "";
    renderWorkspace();
    $("#home-recent-search").focus();
  });
  $("#history-search").addEventListener("input", renderWorkspace);
  $("#history-risk").addEventListener("change", renderWorkspace);
  $("#report-back").addEventListener("click", () => go(state.reportFrom));
  $("#report-save").addEventListener("click", () => toggleWishlist(state.currentProduct?.link));
  $("#active-scan-link").addEventListener("click", () => { if (state.homeScan?.state === "attention") openHomeScan(); else go("analyze"); });
  $(".workspace-backdrop").addEventListener("click", closeWorkspaceNavigation);
  window.addEventListener("hashchange", routeFromHash);

  document.querySelectorAll("[data-route]").forEach(link => link.addEventListener("click", event => {
    event.preventDefault();
    go(link.dataset.route);
    if (link.hasAttribute("data-focus-search")) focusProductSearch();
  }));
  document.querySelectorAll("[data-signin]").forEach(button => button.addEventListener("click", openSignedInPage));

  const activateExploreTab = name => {
    document.querySelectorAll("[data-explore-tab]").forEach(button => {
      const active = button.dataset.exploreTab === name;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
    });
    document.querySelectorAll("[data-explore-panel]").forEach(panel => {
      const active = panel.dataset.explorePanel === name;
      panel.hidden = !active;
      panel.classList.toggle("active", active);
    });
  };
  document.querySelectorAll("[data-explore-tab]").forEach(button => {
    button.addEventListener("click", () => activateExploreTab(button.dataset.exploreTab));
    button.addEventListener("keydown", event => {
      if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
      const tabs = [...document.querySelectorAll("[data-explore-tab]")];
      const step = event.key === "ArrowRight" ? 1 : -1;
      const next = tabs[(tabs.indexOf(button) + step + tabs.length) % tabs.length];
      activateExploreTab(next.dataset.exploreTab);
      next.focus();
    });
  });
  document.querySelectorAll("[data-explore-link]").forEach(link => link.addEventListener("click", event => {
    event.preventDefault();
    go("landing");
    activateExploreTab(link.dataset.exploreLink);
    requestAnimationFrame(() => $("#how")?.scrollIntoView({behavior: "smooth"}));
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
      closeWorkspaceNavigation();
    }
  });
  document.addEventListener("click", event => {
    const header = $(".site-header");
    if (header && !header.contains(event.target)) closeMainNavigation();
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

  const workspaceMenu = $(".workspace-mobile-menu");
  workspaceMenu.addEventListener("click", () => {
    const open = $("#workspace-shell").classList.toggle("sidebar-open");
    workspaceMenu.setAttribute("aria-expanded", String(open));
    $(".workspace-backdrop").hidden = !open;
  });

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
  // Render routes immediately, so a slow catalog request never strands navigation on the landing page.
  const routeReady = routeFromHash();
  try { await loadProducts(); }
  catch (error) { state.catalogError = true; state.catalogLoading = false; renderWorkspace(); toast("Saved results could not be loaded. You can still start a new analysis.", "error"); }
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
