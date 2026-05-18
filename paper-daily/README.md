# paper-daily

`paper-daily` is the candidate-retrieval component of CAPR.

It fetches recent papers, normalizes metadata, deduplicates records, and writes a candidate JSON file for Codex to read. It intentionally does not write the final literature review.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell example:

```powershell
python -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install -r requirements.txt
```

## Fetch Candidates

```bash
python scripts/daily_papers.py --config config.yaml --date today --stage fetch --force
```

Common options:

```bash
python scripts/daily_papers.py --config config.yaml --date 2026-05-15 --stage fetch --force
python scripts/daily_papers.py --config config.yaml --date today --stage fetch --lookback-days 3
python scripts/daily_papers.py --config config.yaml --date today --stage fetch --sources arxiv,openreview
python scripts/daily_papers.py --config config.yaml --date today --stage fetch --sources openalex
```

OpenAlex is implemented as a first-class fetch source alongside arXiv and
OpenReview, but it is disabled by default in `config.yaml`. Enable it only when
you want OpenAlex works to enter the candidate pool directly; otherwise keep it
off and use arXiv/OpenReview as the daily primary sources. If you use an
OpenAlex API key, keep it outside Git and set it through the configured
`api_key_env` environment variable.

Outputs:

```text
data/raw/YYYY-MM-DD.json
data/processed/YYYY-MM-DD_candidates.json
logs/YYYY-MM-DD.log
```

Runtime outputs are ignored by Git.

The raw JSON includes a `duplicate_check` block. If `status` is
`duplicate_of_previous`, today's final candidate pool has the same paper IDs as
the most recent previous candidate file. In that case, `recommended_action` is
`write_no_new_batch_note`, and Codex should write a short "no new candidate
batch today" note instead of repeating the prior Top 10.

For arXiv, CAPR treats the HTML recent-list heading date as the authoritative
daily announcement date. If no papers are listed for the target date, it does
not reuse the latest older batch; the raw JSON will recommend the same short
no-new-batch note.

## Network Preflight

Before fetching, the script checks core source URLs for arXiv, OpenReview, and
OpenAlex when those sources are enabled. If these checks fail, the script exits
before writing an empty candidate file.

CAPR ignores environment proxy variables by default for source requests, because some automation sandboxes expose broken local proxy settings. If you need to use your system proxy, set:

```bash
PAPER_DAILY_USE_ENV_PROXY=1
```

## Candidate Schema

Each candidate includes:

```json
{
  "id": "...",
  "source": "arxiv",
  "title": "...",
  "authors": ["..."],
  "abstract": "...",
  "url": "...",
  "pdf_url": "...",
  "published_at": "...",
  "updated_at": "...",
  "venue": "...",
  "categories": ["..."],
  "keyword_score": 0.0,
  "retrieval_reason": "why the retrieval script kept this candidate",
  "matched_keywords": [],
  "negative_matches": [],
  "coarse_retrieval_score": 0.0
}
```

The retrieval scores are only hints. Codex should still perform semantic reading and scoring before writing the final report.

## Tests

```bash
python -m pytest tests
```
