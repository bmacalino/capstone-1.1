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
3. Enable **Live marketplace scan**.
4. Choose the product and review limits.
5. Select **Check now**.
6. Complete any login or CAPTCHA in the Chrome window opened by the scraper.
7. Wait for scraping, sentiment analysis, and risk computation to finish.

If **Live marketplace scan** is disabled, a product-name search only checks
results already saved in the app. It does not contact the marketplace, so a new
name may return no matches.

### Search using a product link

1. Paste a supported Shopee PH, Lazada PH, or Temu PH product URL.
2. Enable **Live marketplace scan** to collect fresh reviews.
3. Select **Check now** and complete verification if requested.

For supported URLs, the app detects the marketplace from the link. Links from
other countries or unsupported websites are rejected.

### Use an existing result without scraping

Select **Import JSON** in the navigation bar and choose a JSON result generated
by one of the included scrapers. The app analyzes the imported products locally.

The results page lets you inspect sentiment and risk evidence, filter products,
open detailed reports, view suggested lower-risk alternatives, and export CSV
files. The administration area shows locally stored results, lexicons, scraper
logs, and model-evaluation tools.

## FiReCS sentiment evaluation

The repository includes the human-labeled FiReCS train and test sets. To run the
current model against the held-out test set:

```powershell
.\.venv\Scripts\python.exe .\evaluate_firecs.py
```

The command updates:

- `sentiment-analysis/evaluation/firecs_test_baseline.json`
- `sentiment-analysis/evaluation/firecs_test_errors.csv`

The CSV contains the reviews that the model mislabeled and can be used to guide
future lexicon and model improvements.

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

Use the scrapers responsibly, keep request volumes low, and follow each
marketplace's applicable terms and local law.

## Legacy Streamlit app

The original Streamlit prototype remains available during the HTML/CSS/JavaScript
migration:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

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
