import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = path.resolve(import.meta.dirname, "..");
const artifactTool = path.join(
  ROOT,
  ".artifact-work",
  "node_modules",
  "@oai",
  "artifact-tool",
  "dist",
  "artifact_tool.mjs",
);
const { Workbook } = await import(pathToFileURL(artifactTool));

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const position = args.indexOf(name);
  return position >= 0 ? path.resolve(args[position + 1]) : fallback;
};

const DEFAULT_SOURCE_DIR = path.join(ROOT, "research", "data", "processed", "lazada_round1");
const DEFAULT_COMPLETED = path.join(
  ROOT,
  "outputs",
  "annotation_quality_round_1",
  "Completed_annotate.csv",
);
const DEFAULT_OUTPUT = path.join(ROOT, "outputs", "lazada_gold_test_v1");
const sourceDir = option("--source-dir", DEFAULT_SOURCE_DIR);
const completedPath = option("--completed", DEFAULT_COMPLETED);
const outputDir = option("--output", DEFAULT_OUTPUT);

const sourceAllPath = path.join(sourceDir, "lazada_round1_all.csv");
const sourceTestPath = path.join(sourceDir, "lazada_round1_test.csv");
const outputAllPath = path.join(outputDir, "lazada_round1_all_adjudicated.csv");
const outputTestPath = path.join(outputDir, "lazada_round1_test_adjudicated.csv");
const outputBlindPath = path.join(outputDir, "lazada_round1_remaining_blind_review.csv");
const manifestPath = path.join(outputDir, "manifest.json");

const sha256 = async (target) =>
  crypto.createHash("sha256").update(await fs.readFile(target)).digest("hex");

const csvCell = (value) => {
  const text = String(value ?? "");
  return /[",\r\n]/u.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};

const writeCsv = async (target, values) => {
  const body = values.map((row) => row.map(csvCell).join(",")).join("\r\n");
  await fs.writeFile(target, `\uFEFF${body}\r\n`, "utf8");
};

const loadCsv = async (target, sheetName) => {
  const workbook = await Workbook.fromCSV(await fs.readFile(target, "utf8"), { sheetName });
  const sheet = workbook.worksheets.getItem(sheetName);
  const values = sheet.getUsedRange(true).values;
  const headers = values[0].map((value) =>
    String(value ?? "").replace(/^\uFEFF+/u, "").trim(),
  );
  values[0] = headers;
  return {
    workbook,
    sheet,
    values,
    headers,
    column: Object.fromEntries(headers.map((header, index) => [header, index])),
  };
};

const sourceAll = await loadCsv(sourceAllPath, "All source reviews");
const sourceTest = await loadCsv(sourceTestPath, "Test source reviews");
const completed = await loadCsv(completedPath, "Completed adjudication");

for (const name of [
  "review_id",
  "annotator_id",
  "adjudicated_label",
  "risk_detected",
  "risk_category",
  "risk_severity",
  "annotation_notes",
  "split",
]) {
  if (!(name in sourceAll.column) || !(name in sourceTest.column)) {
    throw new Error(`Source dataset is missing required column: ${name}`);
  }
}
for (const name of [
  "Review ID",
  "Final sentiment",
  "Final risk",
  "Final category",
  "Final severity",
  "Decision reason",
  "Status",
]) {
  if (!(name in completed.column)) throw new Error(`Completed annotations are missing: ${name}`);
}

const completedById = new Map();
for (const row of completed.values.slice(1)) {
  const reviewId = String(row[completed.column["Review ID"]] ?? "").trim();
  if (!reviewId) continue;
  if (completedById.has(reviewId)) throw new Error(`Duplicate completed review ID: ${reviewId}`);
  const finalRisk = String(row[completed.column["Final risk"]] ?? "").trim().toLowerCase();
  const finalCategory = String(row[completed.column["Final category"]] ?? "").trim().toLowerCase();
  const finalSeverity = String(row[completed.column["Final severity"]] ?? "").trim();
  const decisionReason = String(row[completed.column["Decision reason"]] ?? "").trim();
  const status = String(row[completed.column.Status] ?? "").trim();
  if (!new Set(["yes", "no"]).has(finalRisk)) throw new Error(`${reviewId}: invalid final risk`);
  if (!decisionReason || status !== "Ready") throw new Error(`${reviewId}: adjudication is not ready`);
  if (finalRisk === "yes" && (!finalCategory || !finalSeverity)) {
    throw new Error(`${reviewId}: risk details are incomplete`);
  }
  if (finalRisk === "no" && (finalCategory || finalSeverity)) {
    throw new Error(`${reviewId}: no-risk row contains risk details`);
  }
  completedById.set(reviewId, {
    sentiment: String(row[completed.column["Final sentiment"]] ?? "").trim().toLowerCase(),
    risk: finalRisk,
    category: finalCategory,
    severity: finalSeverity,
    reason: decisionReason,
  });
}

const testIds = new Set(
  sourceTest.values.slice(1).map((row) => String(row[sourceTest.column.review_id] ?? "").trim()),
);
const missingFromTest = [...completedById.keys()].filter((id) => !testIds.has(id));
if (missingFromTest.length) {
  throw new Error(`Completed IDs are not in the locked test split: ${missingFromTest.join(", ")}`);
}

const updateRows = (dataset) => {
  const values = dataset.values.map((row) => [...row]);
  let updated = 0;
  for (const row of values.slice(1)) {
    const reviewId = String(row[dataset.column.review_id] ?? "").trim();
    const final = completedById.get(reviewId);
    if (!final) continue;
    if (String(row[dataset.column.split] ?? "").trim().toLowerCase() !== "test") {
      throw new Error(`${reviewId}: completed annotation is not assigned to test`);
    }
    row[dataset.column.annotator_id] = "human_adjudicated_round1";
    row[dataset.column.adjudicated_label] = final.sentiment;
    row[dataset.column.risk_detected] = final.risk;
    row[dataset.column.risk_category] = final.category;
    row[dataset.column.risk_severity] = final.severity;
    row[dataset.column.annotation_notes] = final.reason;
    updated += 1;
  }
  return { values, updated };
};

const updatedAll = updateRows(sourceAll);
const updatedTest = updateRows(sourceTest);
if (updatedAll.updated !== completedById.size || updatedTest.updated !== completedById.size) {
  throw new Error(
    `Merge count mismatch: completed=${completedById.size}, all=${updatedAll.updated}, test=${updatedTest.updated}`,
  );
}

const blindHeaders = [
  "review_id",
  "product_id",
  "product_name",
  "product_url",
  "category",
  "rating",
  "review_text",
  "reviewer",
  "human_sentiment",
  "human_risk_detected",
  "human_risk_category",
  "human_risk_severity",
  "decision_reason",
  "status",
];
const blindRows = sourceTest.values.slice(1)
  .filter((row) => !completedById.has(String(row[sourceTest.column.review_id] ?? "").trim()))
  .map((row) => [
    row[sourceTest.column.review_id],
    row[sourceTest.column.product_id],
    row[sourceTest.column.product_name],
    row[sourceTest.column.product_url],
    row[sourceTest.column.category],
    row[sourceTest.column.rating],
    row[sourceTest.column.review_text],
    "",
    "",
    "",
    "",
    "",
    "",
    "Pending",
  ]);
const blindValues = [blindHeaders, ...blindRows];

await fs.mkdir(outputDir, { recursive: true });
await writeCsv(outputAllPath, updatedAll.values);
await writeCsv(outputTestPath, updatedTest.values);
await writeCsv(outputBlindPath, blindValues);

const verifyAll = await loadCsv(outputAllPath, "Adjudicated all reviews");
const verifyTest = await loadCsv(outputTestPath, "Adjudicated test reviews");
const verifyBlind = await loadCsv(outputBlindPath, "Blind review queue");
if (verifyAll.values.length !== sourceAll.values.length) throw new Error("All-row count changed");
if (verifyTest.values.length !== sourceTest.values.length) throw new Error("Test-row count changed");
if (verifyBlind.values.length - 1 !== testIds.size - completedById.size) {
  throw new Error("Blind queue count is incorrect");
}

verifyAll.workbook.recalculate();
verifyTest.workbook.recalculate();
verifyBlind.workbook.recalculate();

const previewDir = path.join(ROOT, ".artifact-work", "lazada_gold_test_v1");
await fs.mkdir(previewDir, { recursive: true });
const preview = await verifyBlind.workbook.render({
  sheetName: "Blind review queue",
  range: "A1:N12",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  path.join(previewDir, "remaining_blind_review.png"),
  new Uint8Array(await preview.arrayBuffer()),
);

const riskCount = updatedTest.values.slice(1).filter(
  (row) => String(row[sourceTest.column.risk_detected] ?? "").trim().toLowerCase() === "yes",
).length;
const sentimentCounts = {};
for (const row of updatedTest.values.slice(1)) {
  const label = String(row[sourceTest.column.adjudicated_label] ?? "").trim().toLowerCase();
  sentimentCounts[label] = (sentimentCounts[label] ?? 0) + 1;
}
const manifest = {
  created_at: new Date().toISOString(),
  policy: {
    originals_preserved: true,
    merge_key: "review_id",
    completed_rows: completedById.size,
    remaining_blind_review_rows: blindRows.length,
    split: "All adjudicated rows remain in the locked test split.",
    blind_review: "Existing labels and model predictions are omitted from the remaining queue.",
  },
  counts: {
    all_rows: updatedAll.values.length - 1,
    test_rows: updatedTest.values.length - 1,
    completed_rows_merged: completedById.size,
    remaining_blind_review_rows: blindRows.length,
    test_risk_yes: riskCount,
    test_risk_no: updatedTest.values.length - 1 - riskCount,
    test_sentiment: Object.fromEntries(Object.entries(sentimentCounts).sort()),
  },
  files: {
    source_all: { path: path.relative(ROOT, sourceAllPath), sha256: await sha256(sourceAllPath) },
    source_test: { path: path.relative(ROOT, sourceTestPath), sha256: await sha256(sourceTestPath) },
    completed_annotations: { path: path.relative(ROOT, completedPath), sha256: await sha256(completedPath) },
    adjudicated_all: { path: path.relative(ROOT, outputAllPath), sha256: await sha256(outputAllPath) },
    adjudicated_test: { path: path.relative(ROOT, outputTestPath), sha256: await sha256(outputTestPath) },
    remaining_blind_review: { path: path.relative(ROOT, outputBlindPath), sha256: await sha256(outputBlindPath) },
  },
};
await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
console.log(JSON.stringify(manifest, null, 2));
