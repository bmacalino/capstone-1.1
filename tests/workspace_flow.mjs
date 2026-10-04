import assert from "node:assert/strict";
import vm from "node:vm";
import { readFile } from "node:fs/promises";

// Dependency-free controller tests. The DOM adapter checks route/state behavior,
// not browser layout or CSS rendering.
//
// Run with Node:  node tests/workspace_flow.mjs
// No Node installed? VS Code ships one:
//   ELECTRON_RUN_AS_NODE=1 "<path to>/Code.exe" tests/workspace_flow.mjs
const source = await readFile(new URL("../static/js/app.js", import.meta.url), "utf8");
const template = await readFile(new URL("../templates/index.html", import.meta.url), "utf8");
class Element {
  constructor(dataset = {}) {
    this.dataset = dataset; this.hidden = false; this.value = ""; this.innerHTML = "";
    this.textContent = ""; this.style = {}; this.attributes = {}; this.events = {};
    this.classes = new Set();
    this.classList = {
      add: name => this.classes.add(name),
      remove: name => this.classes.delete(name),
      contains: name => this.classes.has(name),
      toggle: (name, force) => { const active = force ?? !this.classes.has(name); active ? this.classes.add(name) : this.classes.delete(name); return active; },
    };
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  removeAttribute(name) { delete this.attributes[name]; }
  hasAttribute(name) { return Object.hasOwn(this.attributes, name); }
  addEventListener(name, callback) { this.events[name] = callback; }
  querySelectorAll() { return []; }
  querySelector() { return new Element(); }
  closest() { return null; }
  append(node) { node.parent = this; }
  focus() { this.focused = true; }
  scrollIntoView() {}
  getBoundingClientRect() { return {top: 1000, bottom: 1200}; }
  contains() { return false; }
}
function harness(storage = new Map(), hash = "") {
  const ids = new Map([...template.matchAll(/id="([^"]+)"/g)].map(match => [match[1], new Element()]));
  // These controls are rendered by renderAnalysis, rather than the HTML shell.
  for (const id of ["review-search", "review-rating", "review-result-count", "analysis-review-list", "show-more-reviews"]) ids.set(id, new Element());
  ids.get("review-rating").value = "all";
  ids.get("history-risk").value = "all";
  ids.get("filter-risk").value = "all";
  ids.get("filter-platform").value = "all";
  ids.get("product-count").value = "6";
  ids.get("review-count").value = "30";
  const views = [...template.matchAll(/data-view="([^"]+)"/g)].map(match => {
    const node = ids.get(match[1]); node.dataset.view = match[1]; node.hidden = match[1] !== "landing"; return node;
  });
  const routes = [...template.matchAll(/data-route="([^"]+)"/g)].map(match => new Element({route: match[1]}));
  const nav = ["home", "history", "wishlist", "settings"].map(route => routes.find(node => node.dataset.route === route));
  const classes = new Map([".nav-toggle", ".pub-header"].map(selector => [selector, new Element()]));
  // Home's three card rows: Recently analyzed, Caution, Safety reports.
  const rows = new Map(["recent", "caution", "safety"].map(key => [key, new Element()]));
  const markets = ["shopee", "lazada", "temu"].map(value => { const element = new Element(); element.value = value; return element; });
  const body = new Element(), html = new Element();
  const document = {
    body, documentElement: html,
    querySelector(selector) {
      if (selector === '[data-view]:not([hidden])') return views.find(node => !node.hidden);
      if (selector.startsWith("#") && !selector.includes(" ")) return ids.get(selector.slice(1)) || null;
      if (classes.has(selector)) return classes.get(selector);
      const row = selector.match(/^\[data-home-row="(\w+)"\]$/);
      if (row) return rows.get(row[1]) || null;
      if (selector.includes('input[name="marketplace"]')) return markets.find(node => selector.includes('value="' + node.value + '"')) || markets[0];
      return null;
    },
    querySelectorAll(selector) {
      if (selector === "[data-view]") return views;
      if (selector === "[data-route]") return routes;
      if (selector === ".app-rail [data-route]") return nav;
      if (selector === 'input[name="marketplace"]') return markets;
      return [];
    },
    addEventListener() {},
  };
  const pushes = [], requests = [], window = {
    location: {hash}, scrollY: 0, scrollTo() {}, addEventListener() {},
    history: {
      pushState(_, __, value) { pushes.push(value); window.location.hash = value; },
      replaceState(_, __, value) { window.location.hash = value; },
    },
  };
  const localStorage = {
    getItem: key => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value),
    removeItem: key => storage.delete(key),
  };
  const context = vm.createContext({
    document, window, localStorage, sessionStorage: localStorage,
    URL, URLSearchParams, Intl, console,
    requestAnimationFrame: callback => callback(),
    setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1, clearInterval() {},
    fetch: async (url, options) => {
      requests.push({url, options});
      if (context.respond) return context.respond(url, options);
      throw new Error("Unexpected API call: " + url);
    },
  });
  vm.runInContext(source, context);
  const evaluate = code => vm.runInContext(code, context);
  evaluate("state.catalogLoading = false");
  evaluate("setup()");
  return {ids, views, nav, rows, classes, window, storage, requests, context, evaluate, state: evaluate("state"), pushes};
}
// The server words every verdict; the browser only lays it out. These sample
// products carry the same badge_view the server attaches.
const VERDICTS = {
  safety: {state: "safety_warning", tone: "high", headline: "Reviewers report safety problems", counts: {safety: 3, counterfeit: 0, defect: 1}},
  caution: {state: "possible_counterfeit_or_defect", tone: "moderate", headline: "Reviewers report fakes or defects", counts: {safety: 0, counterfeit: 4, defect: 2}},
  clear: {state: "no_warning_signs", tone: "low", headline: "No warning signs found", counts: {safety: 0, counterfeit: 0, defect: 0}},
  few: {state: "insufficient_evidence", tone: "unknown", headline: "Not enough reviews to judge", counts: {safety: 0, counterfeit: 0, defect: 0}},
};
const product = (link, name, verdict = "clear") => ({
  link, name, platform: "shopee", comments: [],
  badge_view: {available: true, detail: "", reviews_analyzed: 12, badges: [], ...VERDICTS[verdict]},
});
const first = product("https://shopee.ph/charger-i.1.2", "Actual charger", "safety");
const second = product("https://shopee.ph/headphones-i.3.4", "Actual headphones");
const ok = body => ({ok: true, json: async () => body});
const settle = async () => { for (let step = 0; step < 40; step++) await Promise.resolve(); };
const results = [];
async function check(name, test) { await test(); results.push(name); }

await check("Every workspace route uses the shared shell and a single visible page", () => {
  const app = harness();
  for (const route of ["home", "analyze", "history", "wishlist", "settings", "results", "analysis"]) {
    app.evaluate('go("' + route + '")');
    assert.deepEqual(app.views.filter(node => !node.hidden).map(node => node.dataset.view), [route]);
    assert.equal(app.ids.get("workspace-shell").hidden, false);
    assert.equal(app.ids.get(route).parent, app.ids.get("workspace-pages"));
  }
  app.evaluate('go("landing")');
  assert.equal(app.ids.get("workspace-shell").hidden, true);
});
await check("The sign-in page sits outside the app shell, and signed-in users skip it", () => {
  const app = harness();
  app.evaluate('go("login")');
  assert.equal(app.views.find(node => !node.hidden).dataset.view, "login");
  assert.equal(app.ids.get("workspace-shell").hidden, true);
  assert.notEqual(app.ids.get("login").parent, app.ids.get("workspace-pages"));
  app.state.user = {name: "Juan Dela Cruz", email: "juan@example.com"};
  app.evaluate('go("login")');
  assert.equal(app.window.location.hash, "#home");
});
await check("Legacy dashboard/admin links resolve without restoring the removed dashboard", () => {
  const app = harness();
  app.evaluate('go("dashboard")'); assert.equal(app.window.location.hash, "#home");
  app.evaluate('go("admin")'); assert.equal(app.window.location.hash, "#history");
  app.evaluate('go("missing")'); assert.equal(app.window.location.hash, "#home");
});
await check("The rail marks the current page", () => {
  const app = harness();
  app.evaluate('go("wishlist")');
  const active = app.nav.filter(link => link.classes.has("active")).map(link => link.dataset.route);
  assert.deepEqual(active, ["wishlist"]);
  assert.equal(app.nav.find(link => link.dataset.route === "wishlist").attributes["aria-current"], "page");
});
await check("A report opens its exact product; its top link starts another analysis", async () => {
  const app = harness();
  app.state.catalogProducts = [first, second];
  app.evaluate('go("wishlist")');
  app.context.respond = async url => {
    assert.equal(new URL(url, "http://localhost").searchParams.get("link"), second.link);
    return ok({product: second, recommendations: []});
  };
  await app.evaluate('openProduct(' + JSON.stringify(second.link) + ')');
  assert.equal(app.state.currentProduct.name, second.name);
  assert.equal(app.state.reportFrom, "wishlist");
  assert.match(app.ids.get("analysis-content").innerHTML, /Actual headphones/);
  // The rail still marks the page the report was opened from.
  assert.deepEqual(app.nav.filter(link => link.classes.has("active")).map(link => link.dataset.route), ["wishlist"]);
  app.ids.get("report-back").events.click();
  assert.equal(app.window.location.hash, "#analyze");
});
await check("A guest's wishlist and recent activity survive a fresh controller session", () => {
  const app = harness();
  app.state.catalogProducts = [first, second];
  app.evaluate('toggleWishlist(' + JSON.stringify(first.link) + ')');
  const reloaded = harness(app.storage);
  assert.equal(reloaded.state.wishlist[0], first.link);
  reloaded.evaluate('go("wishlist")');
  assert.match(reloaded.ids.get("wishlist-list").innerHTML, /Actual charger/);
  assert.doesNotMatch(reloaded.ids.get("wishlist-list").innerHTML, /Actual headphones/);
  reloaded.evaluate('toggleWishlist(' + JSON.stringify(first.link) + ')');
  assert.match(reloaded.ids.get("wishlist-list").innerHTML, /Keep a product in mind/);
});
await check("History filters and Home numbers use real records and meaningful empty states", () => {
  const app = harness();
  app.evaluate('go("home")');
  assert.match(app.ids.get("home-recent").innerHTML, /Your first analysis starts here/);
  assert.equal(app.ids.get("home-alert").hidden, true);
  app.state.catalogProducts = [first, second];
  app.ids.get("history-risk").value = "safety";
  app.evaluate("renderWorkspace()");
  assert.match(app.ids.get("history-list").innerHTML, /Actual charger/);
  assert.doesNotMatch(app.ids.get("history-list").innerHTML, /Actual headphones/);
  assert.match(app.ids.get("home-stats").innerHTML, /<b>2<\/b><span>Products analyzed/);
  assert.match(app.ids.get("home-stats").innerHTML, /<b>24<\/b><span>Reviews analyzed/);
  assert.match(app.ids.get("home-stats").innerHTML, /<b>1<\/b><span>With warnings/);
  app.ids.get("history-search").value = "no such product";
  app.evaluate("renderWorkspace()");
  assert.match(app.ids.get("history-list").innerHTML, /No reports match/);
});
await check("A listed product without a verdict count shows how many reviews were collected", () => {
  // Lists arrive without the reviews themselves: only their number.
  const app = harness();
  const {comments, ...listed} = product("https://shopee.ph/unchecked-i.5.6", "Not yet analyzed");
  app.state.catalogProducts = [{...listed, badge_view: {...listed.badge_view, reviews_analyzed: 0}, reviews_collected: 7}];
  app.evaluate('go("home")');
  assert.match(app.ids.get("home-stats").innerHTML, /<b>7<\/b><span>Reviews analyzed/);
});
await check("Report URLs restore the exact product after refresh; bare report URLs are not blank", async () => {
  const hash = "#analysis?" + new URLSearchParams({product: first.link, from: "home"});
  const app = harness(new Map(), hash);
  app.context.respond = async () => ok({product: first});
  await app.evaluate("routeFromHash()");
  assert.equal(app.state.currentProduct.link, first.link);
  assert.equal(app.state.reportFrom, "home");
  const empty = harness(new Map(), "#analysis");
  await empty.evaluate("routeFromHash()");
  assert.match(empty.ids.get("analysis-content").innerHTML, /Choose a product report/);
});
await check("Back navigation restores the previous page without adding a new history entry", async () => {
  const app = harness();
  app.evaluate('go("home"); go("history")');
  const count = app.pushes.length;
  app.window.location.hash = "#home"; // The browser restores an earlier hash on Back.
  await app.evaluate("routeFromHash()");
  assert.equal(app.views.find(node => !node.hidden).dataset.view, "home");
  assert.equal(app.pushes.length, count);
});
await check("A slow report response does not override a newer navigation choice", async () => {
  const app = harness();
  app.evaluate('go("history")');
  let resolve;
  app.context.respond = () => new Promise(done => { resolve = done; });
  const pending = app.evaluate('openProduct(' + JSON.stringify(first.link) + ')');
  app.evaluate('go("settings")');
  resolve(ok({product: first}));
  await pending;
  assert.equal(app.window.location.hash, "#settings");
  assert.equal(app.state.currentProduct, null);
});
await check("The bell shows a running scan and opens it; with no scan it says so", () => {
  const app = harness();
  const bell = app.ids.get("active-scan-link");
  bell.events.click();
  assert.equal(app.ids.get("toast").textContent, "No scans running.");
  assert.equal(bell.classes.has("has-dot"), false);
  app.evaluate('go("analyze"); setSearchBusy(true); go("home")');
  assert.equal(bell.classes.has("has-dot"), true);
  assert.match(bell.attributes["aria-label"], /scan is running/);
  bell.events.click();
  assert.equal(app.window.location.hash, "#analyze");
});
await check("Unsupported URLs do not trigger a scan", async () => {
  const app = harness();
  app.ids.get("search-query").value = "https://example.com/item";
  await app.evaluate('search({preventDefault(){}, submitter: {dataset: {searchIntent: "live"}}})');
  assert.equal(app.requests.length, 0);
  assert.equal(app.ids.get("search-feedback-title").textContent, "Unsupported product link");
});
await check("A guest who is out of free scans is sent to sign in, then back to Analyze", async () => {
  const app = harness();
  app.evaluate('go("analyze")');
  app.ids.get("search-query").value = "https://shopee.ph/new-product-i.9.9";
  app.context.respond = async () => ({ok: false, json: async () => ({error: "You've used your 3 free scans today.", limit_reached: true, guest: {limit: 3, left: 0}})});
  await app.evaluate('search({preventDefault(){}, submitter: {dataset: {searchIntent: "live"}}})');
  assert.equal(app.window.location.hash, "#login");
  assert.equal(app.state.guest.left, 0);
  assert.equal(app.state.afterLogin, "analyze");
  assert.equal(app.state.searchBusy, false);
});
await check("Completed direct scans open the requested report even with other catalog products", async () => {
  const app = harness();
  app.evaluate('go("analyze"); setSearchBusy(true)');
  app.context.respond = async url => ({ok: true, json: async () => {
    if (url.startsWith("/api/jobs/")) return {state: "complete", query: second.link, message: "Done"};
    if (url === "/api/products") return {products: [first, second]};
    assert.equal(new URL(url, "http://localhost").searchParams.get("link"), second.link);
    return {product: second};
  }});
  app.evaluate('pollJob("test-job", ' + JSON.stringify({query: second.link}) + ')');
  await settle();
  assert.equal(app.state.currentProduct?.link, second.link);
  assert.equal(app.state.searchBusy, false);
  assert.match(app.ids.get("active-scan-link").attributes["aria-label"], /report is ready/);
});

await check("Home sorts products into rows by what reviewers reported, never 'safe' or 'avoid'", () => {
  const app = harness();
  const lone = product("https://shopee.ph/lone-i.1.7", "One fire report, few reviews", "few");
  lone.badge_view = {...lone.badge_view, counts: {safety: 1, counterfeit: 0, defect: 0}};
  app.state.catalogProducts = [first, second, product("https://shopee.ph/fake-i.1.5", "Possibly fake", "caution"), product("https://shopee.ph/few-i.1.6", "Barely reviewed", "few"), lone];
  app.evaluate('go("home")');
  const recent = app.ids.get("home-recent").innerHTML, caution = app.ids.get("home-caution").innerHTML, safety = app.ids.get("home-safety").innerHTML;
  assert.match(recent, /No warnings/); assert.match(recent, /Too few reviews/);
  assert.match(caution, /Possibly fake/); assert.doesNotMatch(caution, /Actual charger/);
  assert.match(safety, /Actual charger/); assert.match(safety, /app-chip safety">Safety reports/);
  // A single safety report is surfaced even when there are too few reviews for a verdict.
  assert.match(safety, /One fire report, few reviews/); assert.match(safety, /app-chip safety">Safety report</);
  assert.equal(app.rows.get("caution").hidden, false);
  assert.equal(app.rows.get("safety").hidden, false);
  for (const html of [recent, caution, safety, app.ids.get("home-overview").innerHTML]) assert.doesNotMatch(html, />(Safe|Avoid)</);
  assert.match(app.ids.get("home-alert").innerHTML, /2 products have safety reports/);
  assert.match(app.ids.get("home-overview").innerHTML, /Based on 5 products on this browser/);
  assert.match(app.ids.get("home-stores").innerHTML, /Shopee PH.*3 of 5/);
});
await check("Rows with nothing in them stay hidden, and Recently analyzed shows ten at most", () => {
  const app = harness();
  app.state.catalogProducts = Array.from({length: 14}, (_, index) => product("https://shopee.ph/item-i.1." + index, "Gadget " + index));
  app.evaluate('go("home")');
  assert.equal((app.ids.get("home-recent").innerHTML.match(/class="app-product"/g) || []).length, 10);
  assert.equal(app.rows.get("caution").hidden, true);
  assert.equal(app.rows.get("safety").hidden, true);
  assert.equal(app.ids.get("home-alert").hidden, true);
});
await check("Home's hearts and wishlist count follow the wishlist", () => {
  const app = harness();
  app.state.catalogProducts = Array.from({length: 5}, (_, index) => product("https://shopee.ph/saved-i.2." + index, "Saved gadget " + index));
  app.state.wishlist = app.state.catalogProducts.slice(0, 3).map(item => item.link);
  app.evaluate('go("home")');
  assert.equal((app.ids.get("home-recent").innerHTML.match(/aria-pressed="true"/g) || []).length, 3);
  assert.match(app.ids.get("home-stats").innerHTML, /<b>3<\/b><span>In your wishlist/);
});
await check("Home escapes product names", () => {
  const app = harness();
  app.state.catalogProducts = [{...first, name: '<img src=x onerror=alert(1)>'}];
  app.evaluate('go("home")');
  const safety = app.ids.get("home-safety").innerHTML;
  assert.match(safety, /&lt;img/);
  assert.doesNotMatch(safety, /<img src=x/);
  assert.match(app.ids.get("home-alert").innerHTML, /1 product has safety reports/);
});
await check("A scan completed on Home stays on Home and exposes its exact saved report", async () => {
  const app = harness();
  app.evaluate('go("home")');
  app.context.respond = async url => ok(url.startsWith("/api/jobs/") ? {state: "complete", query: second.link, message: "Done"} : {products: [first, second]});
  await app.evaluate('pollJob("home-job", ' + JSON.stringify({query: second.link, platform: "shopee"}) + ')');
  assert.equal(app.window.location.hash, "#home");
  assert.equal(app.state.homeScan.productLink, second.link);
  assert.equal(app.ids.get("home-scan-title").textContent, "Your report is ready");
  assert.equal(app.ids.get("home-scan-status").hidden, false);
  const reloaded = harness(app.storage);
  reloaded.evaluate('go("home")');
  assert.equal(reloaded.state.homeScan.productLink, second.link);
  assert.equal(reloaded.ids.get("home-scan-action").textContent, "Open report →");
});
await check("A progress connection failure keeps an active scan locked and recoverable", async () => {
  const app = harness();
  app.context.respond = async () => { throw new Error("Connection interrupted"); };
  await app.evaluate('pollJob("recoverable-job", ' + JSON.stringify({query: first.link}) + ')');
  assert.equal(app.state.homeScan.state, "attention");
  assert.equal(app.state.activeJobId, "recoverable-job");
  assert.equal(app.state.searchBusy, true);
  assert.equal(app.ids.get("home-scan-action").textContent, "Check progress →");
  app.context.respond = async () => ok({state: "running", stage: 2, progress: 65});
  await app.evaluate('pollJob("recoverable-job", state.homeScan)');
  assert.equal(app.state.homeScan.state, "running");
  assert.equal(app.ids.get("home-scan-title").textContent, "Analyzing review signals");
  assert.equal(app.ids.get("home-scan-progress").attributes["aria-valuenow"], "65");
});
await check("Empty Home has onboarding without an invented alert or scan", () => {
  const app = harness();
  app.evaluate('go("home")');
  assert.equal(app.ids.get("home-alert").hidden, true);
  assert.equal(app.ids.get("home-scan-status").hidden, true);
  assert.match(app.ids.get("home-overview").innerHTML, /Check a product to see your overview/);
  assert.equal(app.ids.get("home-recent").attributes["aria-busy"], "false");
});

await check("A signed-in user's wishlist is saved to the account, not the browser", async () => {
  const app = harness();
  app.state.user = {name: "Juan Dela Cruz", email: "juan@example.com"};
  app.state.catalogProducts = [first, second];
  app.evaluate("cacheWorkspaceProducts(state.catalogProducts)");
  app.context.respond = async () => ok({saved: true});
  app.evaluate('toggleWishlist(' + JSON.stringify(first.link) + ')');
  await settle();
  const calls = app.requests.map(request => request.options.method + " " + request.url);
  assert.deepEqual(calls, ["PUT /api/me/activity", "PUT /api/me/wishlist"]);
  assert.equal(JSON.parse(app.requests[1].options.body).link, first.link);
  assert.equal(app.storage.has("defaketive-wishlist"), false);
  assert.equal(app.storage.has("defaketive-activity"), false);
  assert.match(app.ids.get("toast").textContent, /in your account/);
});
await check("Opening a report as a signed-in user records it in the account", async () => {
  const app = harness();
  app.state.user = {name: "Juan Dela Cruz", email: "juan@example.com"};
  app.context.respond = async url => ok(url.startsWith("/api/products/detail") ? {product: {...second, comments: [{content: "ok"}]}, recommendations: []} : {saved: true});
  await app.evaluate('openProduct(' + JSON.stringify(second.link) + ')');
  await settle();
  const saved = app.requests.find(request => request.url === "/api/me/activity");
  const body = JSON.parse(saved.options.body);
  assert.equal(body.link, second.link);
  assert.ok(body.openedAt > 0);
  // The account keeps a summary; the full review list stays in the analysis store.
  assert.equal(Object.hasOwn(body.product, "comments"), false);
});
await check("Signing in moves this browser's guest history into the account, once", async () => {
  const storage = new Map([
    ["defaketive-wishlist", JSON.stringify([first.link])],
    ["defaketive-activity", JSON.stringify({
      [first.link]: {product: first, openedAt: 0},          // saved but never opened
      [second.link]: {product: second, openedAt: 1700},     // opened
      "https://shopee.ph/seen-i.1.9": {product: product("https://shopee.ph/seen-i.1.9", "Only listed"), openedAt: 0},
    })],
  ]);
  const app = harness(storage);
  app.state.user = {name: "Juan Dela Cruz", email: "juan@example.com"};
  app.context.respond = async (url, options) => {
    assert.equal(url, "/api/me/import");
    const sent = JSON.parse(options.body);
    assert.deepEqual(Object.keys(sent.activity).sort(), [first.link, second.link].sort());
    assert.deepEqual(sent.wishlist, [first.link]);
    return ok({activity: sent.activity, wishlist: sent.wishlist});
  };
  await app.evaluate("loadLibrary()");
  assert.deepEqual([...app.state.wishlist], [first.link]);
  // The next guest on a shared computer must not see this user's history.
  assert.equal(storage.has("defaketive-wishlist"), false);
  assert.equal(storage.has("defaketive-activity"), false);
});
await check("Signing out leaves an empty guest view", () => {
  const app = harness();
  app.state.user = {name: "Juan Dela Cruz", email: "juan@example.com"};
  app.state.catalogProducts = [first];
  app.state.wishlist = [first.link];
  app.evaluate('state.activity = {' + JSON.stringify(first.link) + ': {product: ' + JSON.stringify(first) + ', openedAt: 99}}');
  app.evaluate("applyAuth({user: null, guest: {limit: 3, left: 3}}); forgetLibrary()");
  assert.equal(app.state.wishlist.length, 0);
  assert.equal(app.state.activity[first.link].openedAt, 0);
});

console.log(results.length + " workspace flow checks passed");
export { results };
