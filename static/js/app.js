const state = { products: [], catalogProducts: [], adminProducts: [], admin: "overview", toastTimer: null, activeJobId: null, jobStartedAt: null, jobElapsedTimer: null, searchBusy: false };
const MARKETPLACES = { shopee: "Shopee PH", lazada: "Lazada PH", temu: "Temu PH" };
const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
const money = (value) => String(value || "—").replace(/Ã¢â€šÂ±/g, "₱");
const score = (product) => Number(product?.sentiment_summary?.risk_score || 0);
const reliability = (product) => Math.round(Number(product?.reliability_score ?? product?.sentiment_summary?.reliability_score ?? (100 - score(product))) || 0);
const reviewCount = (product) => Number(product?.sentiment_summary?.review_count ?? product?.comments?.length ?? 0) || 0;
function riskLevel(product) { const risk = score(product); return risk <= 30 ? "Low" : risk <= 60 ? "Moderate" : "High"; }
function safeImage(product) { const candidate = product?.img || product?.image || product?.image_url || product?.thumbnail; return /^https?:\/\//i.test(String(candidate || "")) ? candidate : ""; }
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
  const match = context.platform ? savedProductForQuery(query) : null;

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
  $("#search-note").textContent = hasQuery && !unsupported
    ? "Fresh collection may open Chrome for marketplace login or CAPTCHA verification."
    : "We’ll suggest the best next action after you enter a product.";
}
function setJobStage(stage, reportedProgress) { const fallback = { 1: 18, 2: 68, 3: 92 }[stage] || 18; const progress = Number.isFinite(Number(reportedProgress)) ? Math.min(100, Math.max(0, Number(reportedProgress))) : fallback; $("#job-progress").style.width = `${progress}%`; document.querySelectorAll("[data-job-stage]").forEach(item => { const itemStage = Number(item.dataset.jobStage); item.classList.toggle("complete", itemStage < stage); item.classList.toggle("active", itemStage === stage); }); }
function startJobClock(createdAt) { clearInterval(state.jobElapsedTimer); const parsed = Date.parse(createdAt || ""); state.jobStartedAt = Number.isNaN(parsed) ? Date.now() : parsed; const update = () => { const seconds = Math.max(0, Math.floor((Date.now() - state.jobStartedAt) / 1000)); $("#job-elapsed").textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`; }; update(); state.jobElapsedTimer = setInterval(update, 1000); }
function finishJob() { clearInterval(state.jobElapsedTimer); state.jobElapsedTimer = null; state.activeJobId = null; setSearchBusy(false); }
function closeMainNavigation() {
  const nav = $("#main-nav");
  const toggle = $(".nav-toggle");
  if (!nav || !toggle) return;
  nav.classList.remove("open");
  toggle.setAttribute("aria-expanded", "false");
  toggle.setAttribute("aria-label", "Open navigation");
}

function setActiveNavigation(view, howActive = false) {
  const links = document.querySelectorAll("#main-nav .nav-link");
  links.forEach(link => {
    link.classList.remove("active");
    link.removeAttribute("aria-current");
  });
  const targetView = view === "analysis" ? "results" : view;
  const active = howActive
    ? document.querySelector('#main-nav .nav-link[data-scroll="how"]')
    : document.querySelector(`#main-nav .nav-link[data-route="${targetView}"]`);
  if (active) {
    active.classList.add("active");
    active.setAttribute("aria-current", "page");
  }
}

function currentView() {
  return document.querySelector("[data-view]:not([hidden])")?.dataset.view || "home";
}

function updateHeaderState() {
  const header = $(".site-header");
  if (!header) return;
  header.classList.toggle("scrolled", window.scrollY > 8);
  if (currentView() !== "home") return;
  const how = $("#how");
  const threshold = header.offsetHeight + 80;
  const howActive = Boolean(how && how.getBoundingClientRect().top <= threshold && how.getBoundingClientRect().bottom > threshold);
  setActiveNavigation("home", howActive);
}

function focusProductSearch() {
  requestAnimationFrame(() => {
    const input = $("#search-query");
    if (!input) return;
    input.focus({preventScroll: true});
    input.scrollIntoView({behavior: "smooth", block: "center"});
  });
}

function go(view) {
  document.querySelectorAll("[data-view]").forEach(node => {
    const visible = node.dataset.view === view || (view === "home" && node.dataset.view === "how");
    node.hidden = !visible;
    node.classList.toggle("active", visible);
  });
  window.location.hash = view;
  setActiveNavigation(view);
  closeMainNavigation();
  if (view === "results") renderResults();
  if (view === "admin") renderAdmin();
  window.scrollTo({top: 0, behavior: "smooth"});
}
function platform(product) { return product?.platform ? (MARKETPLACES[String(product.platform).toLowerCase()] || product.platform) : "Marketplace"; }
function productCard(product) { const level = riskLevel(product); const image = safeImage(product); const link = safeLink(product.link); return `<article class="product-card"><div class="product-image">${image ? `<img src="${esc(image)}" alt="">` : "Listing image unavailable"}</div><div class="product-card-body"><span class="platform-badge">${esc(platform(product))}</span><h3>${esc(product.name || "Unnamed product")}</h3><div class="product-price">${esc(money(product.price))}</div><div class="card-meta">★ ${esc(product.rating || "—")} · ${reviewCount(product).toLocaleString()} reviews analyzed</div><span class="risk-badge ${level.toLowerCase()}">${level} risk · ${score(product).toFixed(0)}% score</span><div class="reliability"><i style="width:${Math.min(100, Math.max(0, reliability(product)))}%"></i></div><div class="card-meta">Review reliability: ${reliability(product)}%</div><button class="card-button" data-product-link="${esc(link)}">Open report →</button></div></article>`; }
function filteredProducts() { const platformFilter = $("#filter-platform")?.value || "all"; const riskFilter = $("#filter-risk")?.value || "all"; const sort = $("#filter-sort")?.value || "all"; const visible = state.products.filter(product => (platformFilter === "all" || platform(product) === platformFilter) && (riskFilter === "all" || riskLevel(product) === riskFilter)); return visible.sort((a,b) => { if (sort === "risk") return score(a)-score(b); if (sort === "reviews") return reviewCount(b)-reviewCount(a); return reliability(b)-reliability(a); }); }
function renderResults() { const products = filteredProducts(); $("#results-summary").textContent = products.length ? `${products.length} analyzed product${products.length === 1 ? "" : "s"} shown.` : "No products match your selected filters."; $("#product-grid").innerHTML = products.map(productCard).join(""); document.querySelectorAll("[data-product-link]").forEach(button => button.addEventListener("click", () => openProduct(button.dataset.productLink))); }
async function loadProducts() { const data = await api("/api/products"); state.products = data.products || []; state.catalogProducts = [...state.products]; const heroScore = $("#hero-score"); if (heroScore) { const first = state.products[0]; heroScore.textContent = first ? `${score(first).toFixed(0)}%` : "—"; } if ($("#search-form")) updateSearchUI(); }
function highlightReview(review) { const analysis = review.sentiment_analysis || {}; const terms = (analysis.risk?.evidence || []).filter(item => !item.negated).map(item => item.term).filter(Boolean); let content = esc(review.content || "No written review was collected."); terms.forEach(term => { const escaped = esc(term).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); if (escaped) content = content.replace(new RegExp(`(${escaped})`, "ig"), "<mark>$1</mark>"); }); return content; }
async function openProduct(link) { if (!link) return toast("This product has no listing link."); try { const data = await api(`/api/products/detail?${new URLSearchParams({link})}`); renderAnalysis(data.product, data.recommendations || []); go("analysis"); } catch (error) { toast(error.message, "error"); } }
function sentimentData(product) { const counts = product.sentiment_summary?.sentiment_counts || {}; return [{label:"Positive", value:Number(counts.positive || 0), color:"#12b76a"},{label:"Neutral", value:Number(counts.neutral || 0), color:"#94a3b8"},{label:"Negative", value:Number(counts.negative || 0), color:"#f04438"}]; }
function analysisMainIssue(summary) {
  const issues = Object.entries(summary.risk_categories || {}).sort((a, b) => Number(b[1]) - Number(a[1]));
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

function reviewCard(review) {
  const analysis = review.sentiment_analysis || {};
  const label = String(analysis.label || "unanalyzed").toLowerCase();
  const activeRisk = Boolean(analysis.risk?.detected);
  const stars = Math.max(0, Math.min(5, Number(review.rating || 0)));
  const meta = [review.time || review.date, review.variation].filter(Boolean).map(esc).join(" · ");
  return `<article class="review evidence-review"><div class="review-top"><div><b>${esc(review.author || "Verified shopper")}</b><span class="review-rating" aria-label="${stars} out of 5 stars">★ ${stars || "—"}</span></div><div class="review-labels"><span class="sentiment-pill ${esc(label)}">${esc(label)}</span>${activeRisk ? '<span class="risk-badge high">Risk evidence</span>' : ""}</div></div>${meta ? `<small class="review-meta">${meta}</small>` : ""}<p>${highlightReview(review)}</p></article>`;
}

function renderAnalysis(product, recommendations) {
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
  const saferRecommendations = recommendations.filter(item => score(item) < productRisk);
  const recommendationMarkup = saferRecommendations.length
    ? `<div class="recommendation-list analysis-recommendations">${saferRecommendations.map(item => {
        const itemLevel = riskLevel(item);
        const itemImage = safeImage(item);
        return `<article class="recommendation comparison-card"><div class="recommendation-image">${itemImage ? `<img src="${esc(itemImage)}" alt="">` : "No image"}</div><div><span class="risk-badge ${itemLevel.toLowerCase()}">${itemLevel} · ${score(item).toFixed(0)}% risk</span><b>${esc(item.name || "Unnamed product")}</b><small>${esc(money(item.price))} · ${reliability(item).toFixed(0)}% reliability · ${reviewCount(item).toLocaleString()} reviews</small><button class="table-action recommendation-open" data-product-link="${esc(safeLink(item.link))}">View comparison</button></div></article>`;
      }).join("")}</div>`
    : '<p class="analysis-empty">No sufficiently similar product with a lower measured risk is available in the analyzed collection.</p>';
  const integrityTotal = uniqueCount + duplicateCount;
  const uniqueWidth = integrityTotal ? uniqueCount / integrityTotal * 100 : 0;
  const duplicateWidth = integrityTotal ? duplicateCount / integrityTotal * 100 : 0;

  $("#analysis-content").innerHTML = `<div class="analysis-header analysis-summary-header"><div class="product-image">${image ? `<img src="${esc(image)}" alt="">` : "Listing image unavailable"}</div><div class="analysis-product-copy"><span class="platform-badge">${esc(platform(product))}</span><h2>${esc(product.name || "Unnamed product")}</h2><p>${esc(money(product.price))} · ★ ${esc(product.rating || "—")} · ${uniqueCount.toLocaleString()} sampled reviews</p>${link ? `<a class="text-link" target="_blank" rel="noreferrer" href="${esc(link)}">View original listing ↗</a>` : ""}</div><div class="analysis-score-card"><div class="score-pair"><div><span>Risk score</span><strong class="${riskClass}">${productRisk.toFixed(0)}%</strong><small>Lower is better</small></div><div><span>Reliability</span><strong>${productReliability.toFixed(0)}%</strong><small>Higher is better</small></div></div><div class="risk-meter" role="img" aria-label="${productRisk.toFixed(0)} percent risk, ${productRiskLevel}"><div><span class="low"></span><span class="moderate"></span><span class="high"></span><i style="left:calc(${Math.min(100, Math.max(0, productRisk))}% - 6px)"></i></div><small><span>Low</span><span>Moderate</span><span>High</span></small></div></div></div><div class="analysis-kpis"><article><span>Reviews analyzed</span><b>${uniqueCount.toLocaleString()}</b><small>Unique sample</small></article><article><span>Reviews with risk signals</span><b>${riskReviewCount.toLocaleString()}</b><small>${(riskReviewRate * 100).toFixed(0)}% of sample</small></article><article><span>Main reported issue</span><b>${esc(mainIssue)}</b><small>Most frequent risk category</small></article><article><span>Duplicates excluded</span><b>${duplicateCount.toLocaleString()}</b><small>Not used in scoring</small></article></div><div class="analysis-grid insight-grid"><section class="panel"><h3>Why this score?</h3><p class="panel-subtitle">The weighted score uses unique reviews: 30% negative sentiment and 70% active product-risk signals.</p><div class="contribution-list"><div class="contribution-row"><div><span>Negative-review rate</span><b>${negativeCount}/${uniqueCount} · ${(negativeRate * 100).toFixed(1)}%</b></div><div class="contribution-track"><i class="negative" style="width:${negativeRate * 100}%"></i></div><small>Contributes ${negativePoints.toFixed(1)} points</small></div><div class="contribution-row"><div><span>Reviews containing risk signals</span><b>${riskReviewCount}/${uniqueCount} · ${(riskReviewRate * 100).toFixed(1)}%</b></div><div class="contribution-track"><i class="risk" style="width:${riskReviewRate * 100}%"></i></div><small>Contributes ${riskPoints.toFixed(1)} points</small></div></div><div class="score-formula"><span>Final weighted risk</span><b>${productRisk.toFixed(1)}%</b></div></section><section class="panel"><h3>Sentiment distribution</h3><p class="panel-subtitle">Review counts and percentages after duplicate exclusion.</p>${sentimentStack(product)}</section></div><div class="analysis-grid secondary-insights"><section class="panel"><h3>Top reported issues</h3><p class="panel-subtitle">Risk categories found in the sampled reviews.</p>${issueBars(summary, uniqueCount)}</section><section class="panel"><h3>Review integrity</h3><p class="panel-subtitle">Duplicate reviews are shown separately and excluded from analysis.</p><div class="integrity-stack" role="img" aria-label="${uniqueCount} unique and ${duplicateCount} duplicate reviews"><span class="unique" style="width:${uniqueWidth}%"></span><span class="duplicate" style="width:${duplicateWidth}%"></span></div><div class="integrity-legend"><div><i class="unique"></i><span>Unique reviews</span><b>${uniqueCount}</b></div><div><i class="duplicate"></i><span>Duplicates excluded</span><b>${duplicateCount}</b></div></div></section></div><section class="panel recommendations analysis-recommendation-panel"><div class="section-heading"><div><h3>Lower-risk alternatives</h3><p>Similar analyzed products with a lower risk score than this listing.</p></div></div>${recommendationMarkup}</section><section class="panel evidence-panel"><div class="section-heading"><div><h3>Review evidence</h3><p>Risk terms are highlighted only when the model recorded active evidence.</p></div><span id="review-result-count" class="result-count"></span></div><div class="review-tools"><div class="review-filter-chips" role="group" aria-label="Filter reviews"><button class="active" data-review-filter="all">All</button><button data-review-filter="risk">Risk evidence</button><button data-review-filter="negative">Negative</button><button data-review-filter="neutral">Neutral</button><button data-review-filter="positive">Positive</button></div><div class="review-search-row"><label>Search reviews<input id="review-search" type="search" placeholder="Keyword or reviewer"></label><label>Rating<select id="review-rating"><option value="all">All ratings</option><option value="5">5 stars</option><option value="4">4 stars</option><option value="3">3 stars</option><option value="2">2 stars</option><option value="1">1 star</option></select></label></div></div><div id="analysis-review-list" class="review-list"></div><button id="show-more-reviews" class="secondary-button show-more" type="button">Show more reviews</button></section><p class="analysis-disclaimer">This report analyzes a sample of public reviews. It supports shopping decisions but does not prove that a listing or product is authentic or counterfeit.</p>`;

  document.querySelectorAll(".recommendation-open").forEach(button => button.addEventListener("click", () => openProduct(button.dataset.productLink)));

  let activeFilter = "all";
  let visibleLimit = 10;
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
    $("#analysis-review-list").innerHTML = visible.length ? visible.map(reviewCard).join("") : '<p class="analysis-empty">No reviews match these filters.</p>';
    const showMore = $("#show-more-reviews");
    showMore.hidden = visible.length >= filtered.length;
    showMore.textContent = `Show more reviews (${filtered.length - visible.length} remaining)`;
  };
  document.querySelectorAll("[data-review-filter]").forEach(button => button.addEventListener("click", () => {
    activeFilter = button.dataset.reviewFilter;
    visibleLimit = 10;
    document.querySelectorAll("[data-review-filter]").forEach(item => item.classList.toggle("active", item === button));
    drawReviews();
  }));
  $("#review-search").addEventListener("input", () => { visibleLimit = 10; drawReviews(); });
  $("#review-rating").addEventListener("change", () => { visibleLimit = 10; drawReviews(); });
  $("#show-more-reviews").addEventListener("click", () => { visibleLimit += 10; drawReviews(); });
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
      pollJob(data.job.id, data.job);
    } else {
      state.products = data.products || [];
      if (!state.products.length) {
        showSearchFeedback("No saved analysis matched this search. Choose Collect fresh reviews to scan the marketplace, or try another name.", "No saved match", "warning");
        return;
      }
      go("results");
    }
  } catch (error) {
    showSearchFeedback(error.message);
  } finally {
    if (!jobStarted) setSearchBusy(false);
  }
}
async function pollJob(id, initialJob = {}) { const panel = $("#job-panel"); panel.hidden = false; $("#job-title").textContent = initialJob.platform ? `${MARKETPLACES[initialJob.platform] || initialJob.platform} live scan` : "Live analysis running"; $("#job-message").textContent = initialJob.message || "Opening the marketplace and preparing collection…"; setJobStage(Number(initialJob.stage || 1), initialJob.progress); startJobClock(initialJob.created_at); async function tick() { try { const job = await api(`/api/jobs/${id}`); $("#job-message").textContent = job.message; setJobStage(Number(job.stage || 1), job.progress); if (job.state === "complete") { panel.hidden = true; finishJob(); await loadProducts(); toast(job.message); go("results"); return; } if (job.state === "error") { panel.hidden = true; finishJob(); showSearchFeedback(job.message, "Live scan stopped"); toast("Live scan stopped. See the message near the search form.", "error"); return; } setTimeout(tick, 2500); } catch (error) { panel.hidden = true; finishJob(); showSearchFeedback(error.message, "Could not read scan progress"); toast("Scan progress could not be updated.", "error"); } } tick(); }
async function uploadDataset(event) { const file = event.target.files[0]; if (!file) return; const body = new FormData(); body.append("file", file); try { const data = await api("/api/upload", {method:"POST",body}); state.products = data.products || []; state.catalogProducts = [...state.products]; updateSearchUI(); toast(`${data.count} product${data.count === 1 ? "" : "s"} imported and analyzed.`); go("results"); } catch (error) { toast(error.message, "error"); } finally { event.target.value = ""; } }
function table(rows, columns) { if (!rows.length) return "<p class=\"muted\" style=\"padding:15px\">No data available.</p>"; return `<div class="table-wrap"><table class="data-table"><thead><tr>${columns.map(column => `<th>${esc(column)}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${columns.map(column => `<td>${esc(row[column] ?? "—")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`; }
function dashboardRiskDistribution(distribution) {
  const levels = [
    {label: "Low", className: "low"},
    {label: "Moderate", className: "moderate"},
    {label: "High", className: "high"},
  ];
  const total = levels.reduce((sum, level) => sum + Number(distribution[level.label] || 0), 0);
  if (!total) return '<p class="dashboard-empty">Analyze a product to see an overview of its risk level.</p>';
  const segments = levels.map(level => {
    const count = Number(distribution[level.label] || 0);
    const percentage = count / total * 100;
    return `<span class="risk-segment ${level.className}" style="width:${percentage}%" title="${level.label}: ${count} product${count === 1 ? "" : "s"}" aria-label="${level.label}: ${count} products"></span>`;
  }).join("");
  const legend = levels.map(level => {
    const count = Number(distribution[level.label] || 0);
    const percentage = Math.round(count / total * 100);
    return `<div><i class="${level.className}"></i><span>${level.label}</span><b>${count}</b><small>${percentage}%</small></div>`;
  }).join("");
  return `<div class="risk-stack" role="img" aria-label="Risk levels for ${total} analyzed products">${segments}</div><div class="risk-legend">${legend}</div>`;
}

function dashboardProductTable(rows, compact = false) {
  if (!rows.length) return '<p class="dashboard-empty">No saved reports match your search or filter.</p>';
  const head = compact
    ? "<th>Product</th><th>Main concern</th><th>Reviews analyzed</th><th>Risk level</th><th><span class=\"sr-only\">Action</span></th>"
    : "<th>Product</th><th>Marketplace</th><th>Reviews analyzed</th><th>Review reliability</th><th>Risk level</th><th><span class=\"sr-only\">Action</span></th>";
  const body = rows.map(row => {
    const name = String(row.Product || "Unnamed product");
    const image = safeLink(row.Image);
    const link = safeLink(row.Link);
    const level = String(row["Risk level"] || "Low");
    const levelClass = level.toLowerCase();
    const reliabilityValue = Number(row["Reliability score"] || 0);
    const reliabilityWidth = Math.min(100, Math.max(0, reliabilityValue));
    const product = `<div class="dashboard-product"><div class="dashboard-thumb">${image ? `<img src="${esc(image)}" alt="">` : `<span>${esc(name.charAt(0).toUpperCase())}</span>`}</div><div><b title="${esc(name)}">${esc(name)}</b><small>${esc(row.Price || "Price unavailable")}</small></div></div>`;
    const details = compact
      ? `<td><span class="issue-label">${esc(row["Main issue"] || "No common concern")}</span></td><td>${Number(row.Reviews || 0).toLocaleString()}</td>`
      : `<td>${esc(row.Platform || "Marketplace")}</td><td>${Number(row.Reviews || 0).toLocaleString()}</td><td><div class="table-reliability"><span><i style="width:${reliabilityWidth}%"></i></span><b>${reliabilityValue.toFixed(0)}%</b></div></td>`;
    const action = link ? `<button class="table-action" data-dashboard-product-link="${esc(link)}">Open report</button>` : '<span class="muted">No report</span>';
    return `<tr><td>${product}</td>${details}<td><span class="risk-badge ${esc(levelClass)}">${esc(level)} · ${Number(row["Risk score"] || 0).toFixed(0)}%</span></td><td>${action}</td></tr>`;
  }).join("");
  return `<div class="table-wrap dashboard-table-wrap"><table class="data-table dashboard-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function bindDashboardProductLinks(holder) {
  holder.querySelectorAll("[data-dashboard-product-link]").forEach(button => {
    button.addEventListener("click", () => openProduct(button.dataset.dashboardProductLink));
  });
}

function renderAnalyzedProducts(holder, rows) {
  state.adminProducts = rows;
  holder.innerHTML = `<section class="dashboard-section"><div class="section-heading"><div><h3>All analyzed products</h3><p>Find and reopen a saved product report.</p></div><span id="analyzed-count" class="result-count"></span></div><div class="dashboard-filters"><label>Find a product<input id="dashboard-search" type="search" placeholder="Enter a product or marketplace"></label><label>Risk level<select id="dashboard-risk"><option value="all">All risk levels</option><option>Low</option><option>Moderate</option><option>High</option></select></label></div><div id="dashboard-products"></div></section>`;
  const draw = () => {
    const query = $("#dashboard-search").value.trim().toLowerCase();
    const risk = $("#dashboard-risk").value;
    const visible = state.adminProducts.filter(row => {
      const matchesQuery = !query || `${row.Product || ""} ${row.Platform || ""}`.toLowerCase().includes(query);
      const matchesRisk = risk === "all" || row["Risk level"] === risk;
      return matchesQuery && matchesRisk;
    });
    $("#analyzed-count").textContent = `${visible.length} saved report${visible.length === 1 ? "" : "s"}`;
    $("#dashboard-products").innerHTML = dashboardProductTable(visible);
    bindDashboardProductLinks(holder);
  };
  $("#dashboard-search").addEventListener("input", draw);
  $("#dashboard-risk").addEventListener("change", draw);
  draw();
}

async function renderAdmin() {
  const holder = $("#admin-content");
  holder.innerHTML = '<div class="dashboard-loading"><div class="spinner"></div><span>Updating your insights…</span></div>';
  try {
    if (state.admin === "database") {
      const data = await api("/api/admin/database");
      renderAnalyzedProducts(holder, data.products || []);
      return;
    }

    state.admin = "overview";
    const data = await api("/api/admin/overview");
    const fallbackProducts = state.products || [];
    const summary = data.summary || {
      products: fallbackProducts.length || Number(data.catalog_count || 0),
      reviews: fallbackProducts.reduce((total, product) => total + reviewCount(product), 0),
      average_reliability: fallbackProducts.length
        ? fallbackProducts.reduce((total, product) => total + reliability(product), 0) / fallbackProducts.length
        : 0,
      high_risk: Number(data.risk_distribution?.High || 0),
      marketplaces: new Set(fallbackProducts.map(platform).filter(Boolean)).size,
    };
    holder.innerHTML = `<div class="metrics dashboard-metrics"><article class="metric"><small>Products checked</small><strong>${Number(summary.products || 0).toLocaleString()}</strong><span>Saved product reports</span></article><article class="metric"><small>Reviews analyzed</small><strong>${Number(summary.reviews || 0).toLocaleString()}</strong><span>Across all saved reports</span></article><article class="metric"><small>Average review reliability</small><strong>${Number(summary.average_reliability || 0).toFixed(0)}%</strong><span>Higher is better</span></article><article class="metric" data-tone="danger"><small>Products to review carefully</small><strong>${Number(summary.high_risk || 0).toLocaleString()}</strong><span>Rated high risk</span></article><article class="metric"><small>Marketplaces included</small><strong>${Number(summary.marketplaces || 0).toLocaleString()}</strong><span>Shopee, Lazada, or Temu</span></article></div><div class="dashboard-grid"><section class="panel dashboard-section"><div class="section-heading"><div><h3>Product risk levels</h3><p>See how analyzed products are grouped from low to high risk.</p></div></div>${dashboardRiskDistribution(data.risk_distribution || {})}</section><section class="panel dashboard-section high-risk-section"><div class="section-heading"><div><h3>Products to review carefully</h3><p>These products have the highest risk scores in your saved reports.</p></div></div>${data.flagged?.length ? dashboardProductTable(data.flagged, true) : '<div class="dashboard-empty dashboard-empty-success"><span>✓</span><div><b>No high-risk products right now</b><p>None of the analyzed products are currently rated high risk.</p></div></div>'}</section></div>`;
    bindDashboardProductLinks(holder);
  } catch (error) {
    holder.innerHTML = `<p class="notice">${esc(error.message)}</p>`;
  }
}
function setup() {
  document.querySelectorAll("[data-route]").forEach(link => link.addEventListener("click", event => {
    event.preventDefault();
    go(link.dataset.route);
    if (link.hasAttribute("data-focus-search")) focusProductSearch();
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
  const refreshAdmin = $("#refresh-admin");
  if (refreshAdmin) refreshAdmin.addEventListener("click", renderAdmin);
  document.querySelectorAll("[data-admin]").forEach(button => button.addEventListener("click", () => {
    state.admin = button.dataset.admin;
    document.querySelectorAll("[data-admin]").forEach(item => item.classList.toggle("active", item === button));
    renderAdmin();
  }));

  const navToggle = $(".nav-toggle");
  if (navToggle) navToggle.addEventListener("click", event => {
    const nav = $("#main-nav");
    if (!nav) return;
    const open = nav.classList.toggle("open");
    event.currentTarget.setAttribute("aria-expanded", String(open));
    event.currentTarget.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") closeMainNavigation();
  });
  document.addEventListener("click", event => {
    const header = $(".site-header");
    if (header && !header.contains(event.target)) closeMainNavigation();
  });
  window.addEventListener("scroll", updateHeaderState, {passive: true});

  document.querySelectorAll("[data-scroll]").forEach(link => link.addEventListener("click", event => {
    event.preventDefault();
    go("home");
    closeMainNavigation();
    requestAnimationFrame(() => {
      const section = $("#how");
      if (section) section.scrollIntoView({behavior: "smooth"});
    });
  }));
  updateSearchUI();
  updateHeaderState();
}
document.addEventListener("DOMContentLoaded", async () => { setup(); try { await loadProducts(); } catch (error) { toast("Saved results could not be loaded. You can still import a scraper JSON file.","error"); } const route = window.location.hash.slice(1); if (["results","analysis","admin"].includes(route)) go(route); });
