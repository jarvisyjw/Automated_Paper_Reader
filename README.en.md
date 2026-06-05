# Codex Automated Paper Reader (CAPR)

> [中文](README.md) | English

CAPR is a lightweight workflow for building a customizable daily academic-paper reader with Codex.

It separates the job into two parts:

1. `paper-daily` fetches, normalizes, deduplicates, and coarsely ranks candidate papers from arXiv, OpenReview, and optional OpenAlex.
2. Codex reads the candidate pool, scores papers semantically, selects the most useful papers, and writes a Markdown daily report.

The core principle is intentional: the Python scripts do not write the final literature review. They only prepare a high-quality candidate pool so Codex can act as the research assistant.


## 🖼️ Daily Report预览

![Daily Report Preview](paper-daily/tutorial/daily%20report.png)



## 📰 News

- **2026-05-18** ✨ Added OpenAlex as an optional fetch source: it can run alongside arXiv / OpenReview when enabled, but remains off by default so the daily primary flow is unchanged.
- **2026-05-18** ⚙️ Fixed the arXiv daily-fetch logic: CAPR now treats HTML recent-list announcement dates as the daily candidate-batch source, avoids misses from lagging abs/API dates, and writes a no-new-batch note instead of reusing older batches when the target date has no fresh or non-duplicate candidates.

## ✨ Features

- Fetch recent papers from arXiv and OpenReview, with OpenAlex available as a disabled-by-default optional source.
- Normalize metadata into a consistent JSON schema.
- Deduplicate by source ID and normalized title.
- Rank candidates with transparent rule-based retrieval hints.
- Run network preflight checks before fetching, so network/proxy failures do not silently produce empty reports.
- Bypass broken environment proxy variables by default, with an opt-in proxy switch.
- Fall back to arXiv HTML recent-list pages when the arXiv export API is rate-limited.
- Keep final scoring, reading, and report writing in Codex instead of a brittle keyword template.

## 🗂️ Repository Layout

```text
Codex_Automated_Paper_Reader/
├── README.md
├── README.en.md
├── Paper_Reader.template.txt       # Chinese automation prompt template
├── Paper_Reader.template.en.txt    # English automation prompt template
├── LICENSE
└── paper-daily/
    ├── config.yaml
    ├── requirements.txt
    ├── tutorial/
    │   ├── daily report.png
    │   └── codex automation.png
    ├── scripts/
    │   ├── daily_papers.py
    │   ├── fetch_arxiv.py
    │   ├── fetch_openalex.py
    │   ├── fetch_openreview.py
    │   ├── rank_papers.py
    │   └── utils.py
    └── tests/
```

Runtime outputs are ignored by Git:

```text
paper-daily/data/raw/
paper-daily/data/processed/
paper-daily/logs/
paper-daily/reports/
```

## 🚀 Quick Start

### 📥 1. Clone the project

```bash
git clone https://github.com/<your-user>/Codex_Automated_Paper_Reader.git
cd Codex_Automated_Paper_Reader/paper-daily
```

### 🐍 2. Create a Python environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install -r requirements.txt
```

### 🔎 3. Fetch candidate papers

```bash
python scripts/daily_papers.py --config config.yaml --date today --stage fetch --force
```

Outputs:

```text
data/raw/YYYY-MM-DD.json
data/processed/YYYY-MM-DD_candidates.json
logs/YYYY-MM-DD.log
```

This command only creates the candidate pool. It does not write the final daily report.
`data/raw/YYYY-MM-DD.json` includes `duplicate_check`. If its status is
`duplicate_of_previous`, today's candidate pool is identical to the most recent
previous pool and `recommended_action` is `write_no_new_batch_note`, so Codex
should write a short "no new candidate batch today" note instead of repeating
the Top 10 report.
For arXiv, CAPR prefers the HTML recent-list announcement date. If the target
date has no new announcement batch, the workflow does not reuse an older batch.

## 🤖 Codex Automation

CAPR is designed for Codex standalone automation.

1. Copy one of the public templates. The Chinese template is the default, and an English template is available when needed:

```bash
# Chinese template
cp ../Paper_Reader.template.txt ../Paper_Reader.txt

# English template
cp ../Paper_Reader.template.en.txt ../Paper_Reader.txt
```

2. Edit `Paper_Reader.txt` locally:

- Set the correct `paper-daily` working directory.
- Set the Python executable used by your machine.
- Customize your research background and scoring criteria.

3. Schedule the prompt as a Codex automation.

The local `Paper_Reader.txt` file is ignored by Git because it usually contains machine-specific paths and private screening preferences. The public repository only includes the Chinese and English template files.

![Codex Automation](paper-daily/tutorial/codex%20automation.png)

## 🧾 Candidate Schema

Each candidate in `data/processed/YYYY-MM-DD_candidates.json` roughly follows this shape:

```json
{
  "id": "2605.12345",
  "source": "arxiv",
  "title": "Paper title",
  "authors": ["Author One", "Author Two"],
  "abstract": "Paper abstract...",
  "url": "https://arxiv.org/abs/2605.12345",
  "pdf_url": "https://arxiv.org/pdf/2605.12345",
  "published_at": "2026-05-15T00:00:00+00:00",
  "updated_at": "2026-05-15T00:00:00+00:00",
  "venue": "cs.LG",
  "categories": ["cs.LG", "cs.AI"],
  "keyword_score": 7.0,
  "retrieval_reason": "matched method-transfer terms...",
  "matched_keywords": ["uncertainty estimation"],
  "negative_matches": [],
  "coarse_retrieval_score": 8.0
}
```

The retrieval scores are only hints. Codex should still read and score the papers semantically.

## 📚 Recommended Codex Workflow

After the fetch stage, Codex should:

1. Read `data/processed/YYYY-MM-DD_candidates.json`.
2. Score each paper for methodological relevance, inspiration value, transferability, paper quality, novelty, and actionability.
3. Save semantic scores to `data/processed/YYYY-MM-DD_scored.json`.
4. Open or inspect the top candidates when possible.
5. Write the final report to `reports/YYYY-MM-DD.md`.
6. Clearly state whether each summary is based on abstract-only reading, paper-page reading, or PDF reading.

## ⚙️ Configuration

Edit `paper-daily/config.yaml` to customize:

- research profile
- positive and negative keywords
- arXiv categories
- OpenReview venues
- candidate limits
- arXiv retry and HTML fallback behavior
- output directories

### 🌐 Proxy Behavior

Some automation environments expose unusable loopback proxy variables.

CAPR ignores environment proxies by default for paper-source HTTP requests. If you need to use your system proxy, set:

```bash
PAPER_DAILY_USE_ENV_PROXY=1
```

## ✅ Tests

Run the test suite from `paper-daily`:

```bash
python -m pytest tests
```


## 📌 Project Status

CAPR is an early research automation tool. The current focus is reliability, transparent candidate retrieval, and keeping the final literature review human/Codex-readable rather than template-generated.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE).


```bash
export OPENAI_API_KEY="sk-8e027a39d9f04cccbf0dc7ba3de51f4d" 
export OPENAI_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_MODEL_NAME="qwen3.7-max"
```