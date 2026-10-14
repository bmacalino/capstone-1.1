# DeFaketive Philippine E-Commerce Review Analyzer

DeFaketive is a capstone research prototype that collects public product reviews
from Shopee Philippines, Lazada Philippines, and Temu Philippines. It analyzes
English, Filipino, and Taglish review sentiment, identifies possible product-risk
signals, and presents the evidence in a local web dashboard.

> DeFaketive is a decision-support tool. Its risk score is not proof that a
> product or seller is fraudulent or counterfeit.

## Requirements

- Windows 10 or 11 (recommended)
- Python 3.11
- Google Chrome
- Git, if cloning the repository
- An internet connection for installation and live marketplace scans

Chrome may ask you to log in or solve a CAPTCHA during a live scan. Keep the
Chrome window open until collection finishes.
Shopee scans in the web app allow up to 15 minutes for manual verification.
If a challenge fails, the window stays open so you can recover manually; the
scraper does not automatically retry the challenge. Collection resumes when
Shopee redirects out of login/verification. Use Shopee login or QR login rather
than Google sign-in, which may reject automated browsers.

## Install on a new computer

Open PowerShell and run:

```powershell
git clone https://github.com/bmacalino/capstone-1.1.git
cd capstone-1.1
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m nltk.downloader vader_lexicon
```

If `py -3.11` is unavailable but `python --version` reports Python 3.11, use:

```powershell
python -m venv .venv
```

The `.venv` folder is created separately on every computer and is intentionally
not stored in GitHub. Do not copy an old `.venv` from another computer.

### About VADER

NLTK normally downloads `vader_lexicon` to the current Windows user's NLTK data
directory, commonly `%APPDATA%\nltk_data`. This is why it can appear outside the
project folder. It is a third-party language resource, not project source code.
Run the VADER download command once on each new computer or Windows account.

### Install the sentiment model

The app reads review sentiment with a fine-tuned XLM-RoBERTa model, slimmed to
run on an ordinary CPU. The model files are not stored in GitHub. Copy the
folder `sentiment-analysis\models\sentiment_onnx\xlmr_firecs_v1\` (it holds
`model.onnx`, `tokenizer.json` and `manifest.json`) from the computer that
built it, or rebuild it. Rebuilding needs an NVIDIA GPU, PyTorch with CUDA, and
about an hour. GPU training is not bit-for-bit repeatable, so a rebuilt model
gets a new version and must be re-measured:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\.venv\Scripts\python.exe .\research\finetune_transformer.py --model cardiffnlp/twitter-xlm-roberta-base-sentiment --text normalized --device cuda
.\.venv\Scripts\python.exe .\research\slim_sentiment_model.py --checkpoint sentiment-analysis\models\transformers\cardiffnlp__twitter-xlm-roberta-base-sentiment --name xlmr_firecs_v1 --text raw
```

Without the model, the app does not substitute another one: it shows **Review
analysis is unavailable** and analyzes nothing. The app also checks every file
against the SHA-256 in `manifest.json`, so an edited or partial copy is refused.
To run the older TF-IDF model instead (the research baseline), set
`$env:DEFAKETIVE_SENTIMENT_MODEL = "tfidf"` before starting the app.

The safety and defect checks also use small learned models, stored in git under
`sentiment-analysis\models\`. The app serves `hazard_classifier_v6`: a defect
model trained on every labelled round (rebuild with
`research\train_candidate_v4.py --ordinary`) and a safety filter that can only
remove a keyword safety flag, never one about fire, smoke or shock (rebuild with
`research\train_with_synthetic.py --freeze-safety`). Most of the labels were
proposed by the AI assistant, and the safety filter also learned from sentences
the assistant wrote, which are used for training only
(`research\RESULTS_TABLES.md`, Table 11j). A folder refuses to load if its files, the keyword lists or the
sentiment model it was built with have changed. Fake-product reports come from
keyword rules, and so do two labels that only inform: "poor quality" and "not
as advertised" are shown in the report and never change a verdict. To choose
another checker, set `DEFAKETIVE_HAZARD_MODEL` before starting the app:

- `$env:DEFAKETIVE_HAZARD_MODEL = "team_labels"`: the models trained on the
  team's labels only (`hazard_classifier_v2b`);
- `$env:DEFAKETIVE_HAZARD_MODEL = "keywords"`: the keyword checker alone.

Every choice reports all five labels. The report words them in one place,
`dashboard_utils.BADGE_COPY`: Safety problem, Faulty or broken, Poor quality,
Possible fake, Not as advertised.

A listing gets the strongest safety message, "Reviewers report safety
problems", when two separate reviews report fire, explosion, swelling or an
electric shock. One such report gives the milder "Some reviewers mention
safety concerns", with the reviewer's words (decided 12 Oct 2026; the
setting is `alert.single_report_tier` in
`sentiment-analysis/defaketive_sentiment/lexicons/hazard_parameters_v1.json`,
and `research/RESULTS_TABLES.md`, Table 12e, shows what one, two and three
reports would each do). A flag is real about half the time, which is why one
is not enough.

## Start the web app

### Easiest method on Windows

Open the project in File Explorer and double-click:

```text
run_capstone.cmd
```

Keep the terminal window open. The launcher starts the server and opens:

```text
http://127.0.0.1:5000
```

### Start from PowerShell

You do not need VS Code. From the project directory, run:

```powershell
.\.venv\Scripts\python.exe webapp.py
```

Then open `http://127.0.0.1:5000` in Chrome. Press `Ctrl+C` in PowerShell to stop
the app.

## Use the web app

### Search using a product name

1. Enter a name such as `wireless earbuds`.
2. Select Shopee PH, Lazada PH, or Temu PH.
3. Select **Search analyzed products** to look for an existing report, or
   **Collect fresh reviews** to run a marketplace scan.
4. If needed, expand **Advanced options** to change the product and review
   limits.
5. Complete any login or CAPTCHA in the Chrome window opened by the scraper.
6. Wait for scraping, sentiment analysis, and risk computation to finish.

Searching analyzed products does not contact the marketplace, so a new name may
return no matches. A fresh Shopee or Temu collection opens Chrome on the right
side of the screen so the app's progress panel remains visible. A Lazada
collection runs with no window (see below).

### Search using a product link

1. Paste a supported Shopee PH, Lazada PH, or Temu PH product URL.
2. If a saved report exists, select **View saved report** or **Refresh with new
   reviews**.
3. If no report exists, select **Analyze this product** and complete browser
   verification if requested.

For supported URLs, the app selects the marketplace automatically and scans one
product. Links from other countries or unsupported websites are rejected.

**Lazada scans read up to 60 written reviews from each star level** (up to
300 per product, one to two minutes) instead of the first 100 in Lazada's
order, so complaints in 1- and 2-star reviews are always reached. The scraper
also records how many ratings each star level has, and the report weights
every review back to that mix: a 5-star review on a listing with 13,704
five-star ratings counts for far more than a 1-star review on one with 35.
Review counts on the report stay counts of reviews read; percentages and
rates are weighted. The saved Lazada login is reused, and the scan no longer
waits for someone to scroll the page.

**Lazada scans run with no Chrome window.** Chrome still does the work, in
the background (headless). The window appears by itself only when Lazada asks
for a login or a verification, or when the reviews will not load. Those stay
yours to complete: the progress panel says "This scan needs you" and what to
do, and the scan carries on by itself afterwards (it waits up to ten
minutes). The first scan on a PC with no saved Lazada login is visible for
that reason. Measured on 12 Oct 2026 (`research/check_hidden_lazada.py`,
results in `research/data/reference_scans/hidden_mode_check.json`): Lazada
served a headless Chrome and a window parked outside the screen the same as a
visible one, five listings each, two runs each. Set
`DEFAKETIVE_LAZADA_WINDOW` to `offscreen` or `visible` to change this. Shopee
and Temu scans still open a window.

**Lazada reviews are read from Lazada's own review feed**, not off the page.
That is what makes a scan short (about 20 seconds to collect, then the
reading), and the progress panel follows it step by step. It also changes
what is read. The feed carries whole reviews, while the page reader kept only
the first part of a review written in several parts, and with the whole text
the checker flags more. The "what is typical" line was therefore measured
again on feed scans of the same 59 listings: 6.1% of reviews flagged as
faulty or broken on a typical listing, where page scans gave 4.7%
(`research/RESULTS_TABLES.md`, Table 12d). Two things follow. A report saved
before 12 Oct 2026 was read off the page and runs low against the new line,
so check that product again for a comparable report. And
`DEFAKETIVE_LAZADA_REVIEWS=page` brings the old reader back but not a line
that fits it; a test fails if the baseline and the review source are set
apart.

**Temu scans read on through Temu's own review list** (14 Oct 2026). Temu's
page loads its reviews ten entries at a time from its review list while the
list is scrolled. Every request it makes carries Temu's anti-bot tokens, and
one plain request without them was refused ("System busy", HTTP 429), so the
scraper never asks for a page itself: it scrolls and reads what the page
loaded (`research/check_temu_review_feed.py`,
`research/temu_feed_check.json`). Before this, every Temu scan stopped after
the first page, because the page reader no longer matches Temu's layout: the
saved Temu reports hold 6 to 29 reviews. Measured on one listing whose last
report read 15: a scan at the app's setting read 100 written reviews; reading
the whole list (535 entries, 194 of them written reviews) took about two
minutes and twenty seconds after Temu's check. Temu asks for a check (a
puzzle) often, usually as the product page opens: the progress panel says
"This scan needs you", the window comes to the front, and the scan waits up
to ten minutes. Temu's page no longer shows a review count, so the scraper
records the size of Temu's list instead; a scan that stops before the end of
the list is partial and its report says "Not enough reviews to judge". Temu
reviews are read lowest rating first and Temu offers no star filter, so they
cannot be weighted the way Lazada's are.

The **Analyzed products** page lets you filter saved reports and inspect review
evidence. The **Insights** page summarizes risk levels, review reliability, and
products that may need closer review.

## FiReCS sentiment evaluation

The included supervised sentiment classifier was trained from the human-labeled
FiReCS training split. To reproduce candidate selection, training, and the
recorded held-out evaluation:

```powershell
.\.venv\Scripts\python.exe .\train_firecs.py
```

To evaluate the currently deployed hybrid model:

```powershell
.\.venv\Scripts\python.exe .\evaluate_firecs.py
```

The command updates:

- `sentiment-analysis/evaluation/firecs_current_model.json`
- `sentiment-analysis/evaluation/firecs_current_model_errors.csv`

The trained classifier reached `0.8137` test macro F1. The deployed hybrid,
which retains explicit VADER/Taglish safeguards for short phrases, negation,
emoji, and product-risk evidence, reached `0.7370`; the original VADER baseline
was `0.4818`. Keep the official test split and its errors out of future tuning.

Optional SentiTaglish and TagaSenti source files are stored under
`sentiment-analysis/data`. Train them as a separate research candidate without
overwriting the deployed artifact:

```powershell
.\.venv\Scripts\python.exe .\train_firecs.py `
  --include-sentitaglish --include-tagasenti `
  --model-output .\sentiment-analysis\models\multisource_sentiment_candidate.joblib `
  --result-output .\sentiment-analysis\evaluation\multisource_sentiment_candidate.json `
  --errors-output .\sentiment-analysis\evaluation\multisource_sentiment_candidate_errors.csv
```

The leakage-filtered multisource experiment produced 32,342 training rows but
reached only `0.7951` FiReCS test macro F1, below the deployed classifier's
`0.8137`. It is retained for research and is not automatically promoted.

## Marketplace v2 research candidate

The experimental rebuild trains separate sentiment, risk detection, risk
category, and risk severity models with product-disjoint partitions. It remains
isolated from production because it has not passed every evaluation gate. See
[`research/MARKETPLACE_V2_MODEL_CARD.md`](research/MARKETPLACE_V2_MODEL_CARD.md)
for held-out results, limitations, and reproduction commands.

## Production risk scoring

The production path uses trained marketplace sentiment and binary-risk
classifiers at review level, with VADER and the editable lexicons retained as
supporting evidence. Product aggregation follows the manuscript formula: 30%
of the negative-review ratio plus 70% of the detected-risk ratio. Severity is
not part of the score. It analyzes all usable reviews or a representative
maximum of 100. Set `DEFAKETIVE_MODEL_MODE=manuscript` to run the strict
VADER/lexicon research baseline. See
[`research/RISK_SCORING_MODEL_PLAN.md`](research/RISK_SCORING_MODEL_PLAN.md) and
[`research/sample_size_protocol.md`](research/sample_size_protocol.md).


## Optional scraper command line

Run a Shopee keyword search directly:

```powershell
.\.venv\Scripts\python.exe .\src\retriv.py -k "laptop stand" -n 5 -r 30
```

Run a direct Shopee product URL:

```powershell
.\.venv\Scripts\python.exe .\src\retriv.py --product-url "https://shopee.ph/..." -r 30
```

Useful Shopee options include:

- `-k`, `--keyword`: product search term
- `-n`, `--num`: number of products
- `-r`, `--review-limit`: reviews collected per product
- `--product-url`: direct Shopee PH product URL; repeatable
- `--all-star-types`: collect reviews separately for every star rating
- `--star-limit-per-type`: review limit for each star rating
- `--output`: custom output JSON file

Results are saved incrementally as JSON. The `lazada-scraper` and `temu-scraper`
folders contain their corresponding command-line connectors.

## Troubleshooting

### `ERR_NETWORK_ACCESS_DENIED`

Stop the server and start `run_capstone.cmd` from File Explorer or a normal
PowerShell window. Do not start a live scan from a restricted coding-agent
terminal. Also check Windows Firewall or antivirus settings for Chrome and
Python.

### Port 5000 is already in use

Find the process listening on port 5000:

```powershell
$listener = Get-NetTCPConnection -LocalPort 5000 -State Listen
Get-Process -Id $listener.OwningProcess
```

If it is an old Capstone Python process, stop it and restart the app:

```powershell
Stop-Process -Id $listener.OwningProcess
```

Do not stop the process if it belongs to another application you still need.

### VADER resource is missing

Run:

```powershell
.\.venv\Scripts\python.exe -m nltk.downloader vader_lexicon
```

### PowerShell cannot find Python 3.11

Install Python 3.11 from python.org, enable the Python launcher during setup,
close PowerShell, and repeat the installation steps.

### Marketplace scan returns an error

- Confirm that Chrome and the internet connection work normally.
- Close leftover scraper-controlled Chrome windows and try again.
- Complete login or CAPTCHA prompts before the verification timeout.
- Keep product and review limits low while testing.
- Open **Admin > Scraper** to inspect the latest local log.

## Local files and privacy

Runtime files such as `.venv`, browser profiles, cookies, logs, generated
marketplace JSON, and the SQLite database are excluded from Git. They stay on
the computer running the app. Do not commit account cookies, credentials, or
private datasets.

The app keeps each finished review analysis in `instance/analysis_cache/`, so
it is not repeated every time the app starts. A copy is used only for the same
source and the same model version; copies from other model versions are removed
automatically. Delete the folder to force a fresh analysis. Set
`DEFAKETIVE_ANALYSIS_CACHE` to keep it somewhere else.

Use the scrapers responsibly, keep request volumes low, and follow each
marketplace's applicable terms and local law.

## Tests

Run the main test suite:

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

Run the standalone sentiment-analysis tests:

```powershell
cd sentiment-analysis
..\.venv\Scripts\python.exe -m unittest -v
```

## License and attribution

This project is licensed under the MIT License. See [LICENSE](LICENSE) and
[NOTICE.md](NOTICE.md).

The scraper foundation is derived from the MIT-licensed
[`bmacalino/shopee_scraper`](https://github.com/bmacalino/shopee_scraper), which
itself builds on work by `dtungpka`.

## Capstone team

- **Capstone Adviser:** Kathleen De Guzman
- **Project Leader:** Mark Rhean Caballero
- **Programmer:** Brian Macalino
- **Sub-Programmer and Designer:** Ryan Rovic Retuya
- **Designer:** Princethom Guyo
