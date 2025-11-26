# PromptFuzz CLI

An LLM red-team fuzzer that runs a library of jailbreak prompts through multiple mutation strategies, scores the responses, logs everything to CSV/JSON, and ships a clean HTML report with a pie chart summary.

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate   # or source .venv/bin/activate on *nix
pip install -e .
```

## Usage

```bash
promptfuzz --target gpt-4 \\
  --prompts prompts.txt \\
  --mutations prefix,base64,typo \\
  --api-url https://api.your-llm.com/v1/chat \\
  --api-key-file .key \\
  --out-html report.html --out-json report.json --out-csv report.csv
```

Flags:
- `--target` model identifier (shown in requests and reports).
- `--prompts` path to a newline-separated prompt list; defaults to built-in examples.
- `--mutations` comma list from `prefix, base64, typo, hex, padding`.
- `--api-url` HTTP endpoint to POST `{prompt, model}`; omit to use the built-in mock client.
- `--api-key-file` file containing your bearer token; falls back to `OPENAI_API_KEY` env.
- `--out-html`, `--out-json`, `--out-csv` output paths for reports/logs.

## What it does
1) Loads base prompts and applies each enabled mutation, producing multiple test cases per prompt.  
2) Sends the mutated prompt to the LLM (or the offline mock client when no `--api-url` is given).  
3) Analyzes the response with keyword-based detection to mark `BYPASSED` vs `BLOCKED`.  
4) Logs every run to JSON and CSV (ID, mutation type, status, prompt, truncated output, latency).  
5) Renders a dark HTML report with a bypass/blocked pie chart and expandable payload/response details.

## HTML Report
Open the generated `report.html` to review totals, bypass counts, success rate, and detailed rows. The pie chart is drawn with a tiny inline canvas helper—no external assets required.

## Notes
- The default client is mock-only; supply `--api-url` and an API key to hit a real model.  
- Latency is measured per request for quick comparisons across mutations.  
- Extend the mutator map in `src/cli.py` to add new evasion strategies (rot13, multi-turn, etc.).
