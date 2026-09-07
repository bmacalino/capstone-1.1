import assert from "node:assert/strict";
import vm from "node:vm";
import { readFile } from "node:fs/promises";

// Dependency-free controller tests. The DOM adapter checks route/state behavior,
// not browser layout or CSS rendering.
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
  const nav = ["home", "analyze", "history", "wishlist", "settings"].map(route => routes.find(node => node.dataset.route === route));
  const classes = new Map([".nav-toggle", ".site-header", ".workspace-mobile-menu", ".workspace-backdrop"].map(selector => [selector, new Element()]));
  const markets = ["shopee", "lazada", "temu"].map(value => { const element = new Element(); element.value = value; return element; });
  const body = new Element(), html = new Element();
  const document = {
    body, documentElement: html,
    querySelector(selector) {
      if (selector === '[data-view]:not([hidden])') return views.find(node => !node.hidden);
      if (selector.startsWith("#") && !selector.includes(" ")) return ids.get(selector.slice(1)) || null;
      if (classes.has(selector)) return classes.get(selector);
      if (selector.includes('input[name="marketplace"]')) return markets.find(node => selector.includes('value="' + node.value + '"')) || markets[0];
      return null;
    },
    querySelectorAll(selector) {
      if (selector === "[data-view]") return views;
      if (selector === "[data-route]") return routes;
      if (selector === ".workspace-nav [data-route]") return nav;
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
  return {ids, views, nav, classes, window, storage, requests, context, evaluate, state: evaluate("state"), pushes};
}
const product = (link, name, risk = 10) => ({link, name, platform: "shopee", comments: [], sentiment_summary: {risk_score: risk, review_count: 12}});
const first = product("https://shopee.ph/charger-i.1.2", "Actual charger", 80);
const second = product("https://shopee.ph/headphones-i.3.4", "Actual headphones");
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
await check("Legacy dashboard/admin links resolve without restoring the removed dashboard", () => {
  const app = harness();
  app.evaluate('go("dashboard")'); assert.equal(app.window.location.hash, "#home");
  app.evaluate('go("admin")'); assert.equal(app.window.location.hash, "#history");
  app.evaluate('go("missing")'); assert.equal(app.window.location.hash, "#home");
});
await check("A report opens its exact product and returns to its source page", async () => {
  const app = harness();
  app.state.catalogProducts = [first, second];
  app.evaluate('go("wishlist")');
  app.context.respond = async url => {
    assert.equal(new URL(url, "http://localhost").searchParams.get("link"), second.link);
    return {ok: true, json: async () => ({product: second, recommendations: []})};
  };
  await app.evaluate('openProduct(' + JSON.stringify(second.link) + ')');
  assert.equal(app.state.currentProduct.name, second.name);
  assert.equal(app.state.reportFrom, "wishlist");
  assert.match(app.ids.get("analysis-content").innerHTML, /Actual headphones/);
  app.ids.get("report-back").events.click();
  assert.equal(app.window.location.hash, "#wishlist");
});
await check("Wishlist and recent activity survive a fresh controller session", () => {
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
await check("History filters and Home metrics use real records and meaningful empty states", () => {
  const app = harness();
  app.evaluate('go("home")');
  assert.match(app.ids.get("home-recent").innerHTML, /Your first analysis starts here/);
  assert.equal(app.ids.get("home-risk-alert").innerHTML, "");
  app.state.catalogProducts = [first, second];
  app.ids.get("history-risk").value = "High";
  app.evaluate("renderWorkspace()");
  assert.match(app.ids.get("history-list").innerHTML, /Actual charger/);
  assert.doesNotMatch(app.ids.get("history-list").innerHTML, /Actual headphones/);
  assert.match(app.ids.get("home-metrics").innerHTML, /<b>24<\/b>/);
  app.ids.get("history-search").value = "no such product";
  app.evaluate("renderWorkspace()");
  assert.match(app.ids.get("history-list").innerHTML, /No reports match/);
});
await check("Report URLs restore the exact product after refresh; bare report URLs are not blank", async () => {
  const hash = "#analysis?" + new URLSearchParams({product: first.link, from: "home"});
  const app = harness(new Map(), hash);
  app.context.respond = async () => ({ok: true, json: async () => ({product: first})});
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
  resolve({ok: true, json: async () => ({product: first})});
  await pending;
  assert.equal(app.window.location.hash, "#settings");
  assert.equal(app.state.currentProduct, null);
});
await check("Mobile navigation closes on navigation and scan progress stays discoverable", () => {
  const app = harness();
  app.classes.get(".workspace-mobile-menu").events.click();
  assert.equal(app.classes.get(".workspace-mobile-menu").attributes["aria-expanded"], "true");
  app.evaluate('go("analyze"); setSearchBusy(true); go("home")');
  assert.equal(app.classes.get(".workspace-mobile-menu").attributes["aria-expanded"], "false");
  assert.equal(app.ids.get("active-scan-link").hidden, false);
  app.ids.get("active-scan-link").events.click();
  assert.equal(app.window.location.hash, "#analyze");
});
await check("Unsupported URLs do not trigger a scan", async () => {
  const app = harness();
  app.ids.get("search-query").value = "https://example.com/item";
  await app.evaluate('search({preventDefault(){}, submitter: {dataset: {searchIntent: "live"}}})');
  assert.equal(app.requests.length, 0);
  assert.equal(app.ids.get("search-feedback-title").textContent, "Unsupported product link");
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
  for (let step = 0; step < 40; step++) await Promise.resolve();
  assert.equal(app.state.currentProduct?.link, second.link);
  assert.equal(app.state.searchBusy, false);
  assert.equal(app.ids.get("active-scan-link").hidden, true);
});

await check("Home search finds reports beyond the four recent items and clears cleanly", () => {
  const app = harness();
  app.state.catalogProducts = Array.from({length: 7}, (_, index) => product("https://shopee.ph/item-i.1." + index, "Gadget " + index));
  app.evaluate('go("home")');
  assert.doesNotMatch(app.ids.get("home-recent").innerHTML, /Gadget 6/);
  app.ids.get("home-recent-search").value = "Gadget 6";
  app.ids.get("home-recent-search").events.input();
  assert.match(app.ids.get("home-recent").innerHTML, /Gadget 6/);
  assert.match(app.ids.get("home-recent-count").textContent, /1 matching report/);
  app.ids.get("home-clear-search").events.click();
  assert.equal(app.ids.get("home-recent-search").value, "");
  assert.match(app.ids.get("home-recent-count").textContent, /Showing 4 recent/);
});
await check("The last-report shortcut tracks opened reports, not wishlist clicks", () => {
  const app = harness();
  app.state.catalogProducts = [first, second];
  app.evaluate('rememberProduct(' + JSON.stringify(first) + '); toggleWishlist(' + JSON.stringify(second.link) + '); go("home")');
  assert.equal(app.ids.get("home-continue").dataset.productLink, first.link);
  assert.equal(app.ids.get("home-continue-name").textContent, first.name);
  assert.equal(app.ids.get("home-continue").hidden, false);
});
await check("Wishlist preview shows at most three real saved products", () => {
  const app = harness();
  app.state.catalogProducts = Array.from({length: 5}, (_, index) => product("https://shopee.ph/saved-i.2." + index, "Saved gadget " + index));
  app.state.wishlist = app.state.catalogProducts.map(item => item.link);
  app.evaluate('go("home")');
  assert.equal((app.ids.get("home-wishlist-preview").innerHTML.match(/home-saved-product/g) || []).length, 3);
  assert.equal(app.ids.get("home-wishlist-count").textContent, "5");
  assert.match(app.ids.get("home-wishlist-preview").innerHTML, /Saved gadget 4/);
});
await check("Home alerts name the product, escape its concern, and link to evidence", () => {
  const app = harness();
  app.state.catalogProducts = [{...first, name: '<img src=x onerror=alert(1)>', sentiment_summary: {...first.sentiment_summary, risk_categories: {"broken <script>": 2}}}];
  app.evaluate('go("home")');
  const alert = app.ids.get("home-risk-alert").innerHTML;
  assert.match(alert, /data-home-evidence=/);
  assert.match(alert, /&lt;img/);
  assert.doesNotMatch(alert, /<script>/);
  assert.match(alert, /Most reported concern:/);
});
await check("A scan completed on Home stays on Home and exposes its exact saved report", async () => {
  const app = harness();
  app.evaluate('go("home")');
  app.context.respond = async url => ({ok: true, json: async () => url.startsWith("/api/jobs/") ? {state: "complete", query: second.link, message: "Done"} : {products: [first, second]}});
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
  app.context.respond = async () => ({ok: true, json: async () => ({state: "running", stage: 2, progress: 65})});
  await app.evaluate('pollJob("recoverable-job", state.homeScan)');
  assert.equal(app.state.homeScan.state, "running");
  assert.equal(app.ids.get("home-scan-title").textContent, "Analyzing review signals");
  assert.equal(app.ids.get("home-scan-progress").attributes["aria-valuenow"], "65");
});
await check("Empty Home has onboarding without an invented last report or risk alert", () => {
  const app = harness();
  app.evaluate('go("home")');
  assert.equal(app.ids.get("home-continue").hidden, true);
  assert.equal(app.ids.get("home-risk-alert").innerHTML, "");
  assert.equal(app.ids.get("home-scan-status").hidden, true);
  assert.match(app.ids.get("home-wishlist-preview").innerHTML, /Keep your options together/);
  assert.equal(app.ids.get("home-recent").attributes["aria-busy"], "false");
});

export { results };
