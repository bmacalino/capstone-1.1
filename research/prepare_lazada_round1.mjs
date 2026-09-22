import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = path.resolve(import.meta.dirname, "..");
const artifactTool = path.join(ROOT, ".artifact-work", "node_modules", "@oai", "artifact-tool", "dist", "artifact_tool.mjs");
const { FileBlob, SpreadsheetFile } = await import(pathToFileURL(artifactTool));
const DEFAULT_INPUT = path.join(ROOT, "Gemini_Lazada_Annotation_Round_1_Fully_Annotated.xlsx");
const DEFAULT_OUTPUT = path.join(ROOT, "research", "data", "processed", "lazada_round1");
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const position = args.indexOf(name);
  return position >= 0 ? path.resolve(args[position + 1]) : fallback;
};
const input = option("--input", DEFAULT_INPUT);
const output = option("--output", DEFAULT_OUTPUT);

const norm = value => String(value ?? "").normalize("NFKC").trim().toLowerCase().replace(/\s+/gu, " ");
const csvCell = value => {
  const text = String(value ?? "");
  return /[",\r\n]/u.test(text) ? `"${text.replace(/"/gu, '""')}"` : text;
};
const writeCsv = async (target, rows, headers) => {
  const lines = [headers.map(csvCell).join(","), ...rows.map(row => headers.map(h => csvCell(row[h])).join(","))];
  await fs.writeFile(target, `\uFEFF${lines.join("\r\n")}\r\n`, "utf8");
};
const sha256 = value => crypto.createHash("sha256").update(value).digest("hex");

const riskMap = new Map([
  ["defect", "defect"], ["defective product", "defect"],
  ["performance", "performance"], ["compatibility issue", "performance"],
  ["quality", "quality"], ["quality issue", "quality"],
  ["durability", "durability"],
  ["damage", "damage"], ["damaged item", "damage"],
  ["missing_item", "missing_item"], ["missing item", "missing_item"],
  ["wrong_item", "wrong_item"], ["wrong item", "wrong_item"],
  ["wrong_color_or_size", "wrong_color_or_size"], ["wrong color or size", "wrong_color_or_size"],
  ["delivery", "delivery"], ["delivery issue", "delivery"],
  ["seller_service", "seller_service"], ["seller problem", "seller_service"], ["false advertising", "seller_service"],
  ["other", "other"], ["safety hazard", "other"]
]);
const sourceCategoryMap = new Map([
  ["lazada-electronics-three-star-first-retry", "electronics"],
  ["computer-accessories-three-star-first", "computer_peripherals"],
  ["smart-home-three-star-first", "smart_home"]
]);

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(input));
const sheet = [...workbook.worksheets][0];
const values = sheet.getUsedRange().values;
const headers = values[0].map(value => String(value ?? "").trim());
const index = Object.fromEntries(headers.map((header, column) => [header, column]));
const required = ["review_id", "platform", "product_id", "review_text", "category", "rating", "sentiment_1", "risk_detected_1", "risk_category_1", "risk_severity_1", "negated_1"];
const missing = required.filter(header => !(header in index));
if (missing.length) throw new Error(`Missing columns: ${missing.join(", ")}`);
const get = (row, header) => String(row[index[header]] ?? "").trim();

const rawRows = values.slice(1).filter(row => row.some(value => String(value ?? "").trim()));
const excluded = [];
const duplicates = [];
const seenText = new Set();
const records = [];
for (const row of rawRows) {
  const sourceCategory = norm(get(row, "category"));
  if (sourceCategory === "out_of_scope") {
    excluded.push({ review_id: get(row, "review_id"), reason: "out_of_scope" });
    continue;
  }
  const text = get(row, "review_text");
  const textKey = norm(text);
  if (seenText.has(textKey)) {
    duplicates.push({ review_id: get(row, "review_id"), reason: "duplicate_normalized_review_text", review_text: text });
    continue;
  }
  seenText.add(textKey);
  const adjudicatedSentiment = norm(get(row, "adjudicated_sentiment"));
  const adjudicatedRisk = norm(get(row, "adjudicated_risk_detected"));
  const finalRisk = adjudicatedRisk || norm(get(row, "risk_detected_1"));
  const rawRiskCategory = adjudicatedRisk ? norm(get(row, "adjudicated_risk_category")) : norm(get(row, "risk_category_1"));
  const finalRiskCategory = finalRisk === "yes" ? riskMap.get(rawRiskCategory) : "";
  if (finalRisk === "yes" && !finalRiskCategory) throw new Error(`Unknown risk category '${rawRiskCategory}' for ${get(row, "review_id")}`);
  records.push({
    review_id: get(row, "review_id"), platform: norm(get(row, "platform")), product_id: get(row, "product_id"),
    product_name: get(row, "product_name"), product_url: get(row, "product_url"),
    category: sourceCategoryMap.get(sourceCategory) ?? sourceCategory, rating: get(row, "rating"), review_text: text,
    source_file: get(row, "source_file"), source_rating_filter: get(row, "source_rating_filter"),
    dual_annotation_required: norm(get(row, "dual_annotation_required")), annotator_id: get(row, "annotator_1_id"),
    adjudicated_label: adjudicatedSentiment || norm(get(row, "sentiment_1")), risk_detected: finalRisk,
    risk_category: finalRiskCategory, risk_severity: finalRisk === "yes" ? (get(row, "adjudicated_risk_severity") || get(row, "risk_severity_1")) : "",
    negated: norm(get(row, "adjudicated_negated")) || norm(get(row, "negated_1")),
    annotation_notes: get(row, "adjudication_notes") || get(row, "notes_1"), split: ""
  });
}

// Deterministic product-grouped split. The objective balances row count,
// sentiment, and risk prevalence while guaranteeing zero product leakage.
const splitNames = ["train", "validation", "test"];
const target = { train: 0.70, validation: 0.15, test: 0.15 };
const labels = ["positive", "negative", "neutral", "mixed"];
const groups = new Map();
for (const record of records) {
  if (!groups.has(record.product_id)) groups.set(record.product_id, []);
  groups.get(record.product_id).push(record);
}
const totals = Object.fromEntries(labels.map(label => [label, records.filter(r => r.adjudicated_label === label).length]));
const riskTotal = records.filter(r => r.risk_detected === "yes").length;
const state = Object.fromEntries(splitNames.map(name => [name, { rows: 0, risk: 0, labels: Object.fromEntries(labels.map(l => [l, 0])) }]));
const score = (name, group) => {
  const nextRows = state[name].rows + group.length;
  const rowTarget = records.length * target[name];
  const rowDelta = (Math.abs(nextRows - rowTarget) - Math.abs(state[name].rows - rowTarget)) / records.length;
  const groupRisk = group.filter(r => r.risk_detected === "yes").length;
  const riskTarget = riskTotal * target[name];
  const riskDelta = (Math.abs(state[name].risk + groupRisk - riskTarget) - Math.abs(state[name].risk - riskTarget)) / Math.max(riskTotal, 1);
  const labelDelta = labels.reduce((sum, label) => {
    const labelTarget = totals[label] * target[name];
    const groupCount = group.filter(r => r.adjudicated_label === label).length;
    return sum + (Math.abs(state[name].labels[label] + groupCount - labelTarget) - Math.abs(state[name].labels[label] - labelTarget)) / Math.max(totals[label], 1);
  }, 0);
  return 4 * rowDelta + riskDelta + labelDelta;
};
const orderedGroups = [...groups.entries()].sort((a, b) => b[1].length - a[1].length || sha256(a[0]).localeCompare(sha256(b[0])));
for (const [, group] of orderedGroups) {
  const chosen = splitNames.map(name => [name, score(name, group)]).sort((a,b) => a[1]-b[1] || splitNames.indexOf(a[0])-splitNames.indexOf(b[0]))[0][0];
  for (const record of group) record.split = chosen;
  state[chosen].rows += group.length;
  state[chosen].risk += group.filter(r => r.risk_detected === "yes").length;
  for (const label of labels) state[chosen].labels[label] += group.filter(r => r.adjudicated_label === label).length;
}

const outputHeaders = ["review_id","platform","product_id","product_name","product_url","category","rating","review_text","source_file","source_rating_filter","dual_annotation_required","annotator_id","adjudicated_label","risk_detected","risk_category","risk_severity","negated","annotation_notes","split"];
await fs.mkdir(output, { recursive: true });
await writeCsv(path.join(output, "lazada_round1_all.csv"), records, outputHeaders);
for (const split of splitNames) await writeCsv(path.join(output, `lazada_round1_${split}.csv`), records.filter(r => r.split === split), outputHeaders);
await writeCsv(path.join(output, "lazada_round1_excluded.csv"), excluded, ["review_id", "reason"]);
await writeCsv(path.join(output, "lazada_round1_duplicates.csv"), duplicates, ["review_id", "reason", "review_text"]);

const manifest = {
  source_file: path.basename(input), source_sha256: sha256(await fs.readFile(input)),
  policy: { final_label: "adjudicated label when present, otherwise verified primary label", mixed: "retained for risk; excluded by the existing trainer from three-class sentiment", split: "deterministic product-grouped 70/15/15 target", risk_taxonomy: [...new Set(riskMap.values())].sort() },
  counts: { source_rows: rawRows.length, retained_rows: records.length, excluded_out_of_scope: excluded.length, removed_duplicate_texts: duplicates.length, products: groups.size },
  splits: Object.fromEntries(splitNames.map(name => [name, { rows: records.filter(r => r.split === name).length, products: new Set(records.filter(r => r.split === name).map(r => r.product_id)).size, sentiment: Object.fromEntries(labels.map(label => [label, records.filter(r => r.split === name && r.adjudicated_label === label).length])), risk_yes: records.filter(r => r.split === name && r.risk_detected === "yes").length }])),
  leakage_check: splitNames.every((name, i) => splitNames.slice(i + 1).every(other => ![...new Set(records.filter(r => r.split === name).map(r => r.product_id))].some(id => records.some(r => r.split === other && r.product_id === id))))
};
await fs.writeFile(path.join(output, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
console.log(JSON.stringify(manifest, null, 2));
