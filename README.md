# PromptFuzz CLI

An LLM red-team fuzzer that runs a library of jailbreak prompts through multiple mutation strategies, scores the responses, logs everything to CSV/JSON, and ships a clean HTML report with a pie chart summary.

**Features:**
- 18 mutation strategies (DAN, AIM, base64, token smuggling, etc.)
- Beautiful HTML reports with charts
- Automatic retry with exponential backoff
- Mock mode for testing without API costs
- Comprehensive logging (JSON, CSV)

## Quick Start

**Try it immediately without any API keys:**

```bash
git clone https://github.com/souliN02/PromptFuzz-CLI.git
cd promptfuzz-cli
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# Run with mock LLM (no API required)
promptfuzz --target gpt-mock --prompts prompts.txt --mutations dan,aim,grandma

# Open report.html in your browser to see results!
```

This generates a full HTML report showing which jailbreak techniques would bypass filters, perfect for demos and testing!

## Prerequisites

- Python 3.8+
- pip and venv
- (Optional) API key for real LLM testing:
  - **Groq** (recommended - 30 RPM free): [console.groq.com](https://console.groq.com)
  - **Ollama** (unlimited local): [ollama.com](https://ollama.com)
  - **OpenAI** (3 RPM, needs credits): [platform.openai.com](https://platform.openai.com)

## Installation

```bash
git clone https://github.com/souliN02/PromptFuzz-CLI.git
cd promptfuzz-cli
python -m venv .venv
.venv\Scripts\activate   # or source .venv/bin/activate on *nix
pip install -e .
```

## Usage

### Step 1: Test Locally (No API Key Required)

Start with mock mode to understand how the tool works:

```bash
promptfuzz --target gpt-mock --prompts prompts.txt --mutations dan,aim,base64
```

Open `report.html` to see the results!

### Step 2: Test with Real LLMs

Once you have an OpenAI API key with credits:

```bash
# Create .key file with your API key
echo "sk-your-api-key-here" > .key

# Run against GPT-3.5-turbo (recommended for testing)
promptfuzz --target gpt-3.5-turbo \
  --prompts test.txt \
  --mutations dan,prefix,base64 \
  --api-url https://api.openai.com/v1/chat/completions \
  --api-key-file .key \
  --delay-ms 25000
```

### Step 3: Comprehensive Security Audit

Run all 18 mutations for thorough testing:

```bash
promptfuzz --target gpt-4 \
  --prompts prompts.txt \
  --mutations all \
  --api-url https://api.openai.com/v1/chat/completions \
  --api-key-file .key \
  --delay-ms 25000 \
  --out-html report.html --out-json report.json --out-csv report.csv
```

## Alternative LLM Providers

### 🚀 Groq (Free Tier: 30 RPM - 10x OpenAI!)

Groq offers fast inference with a **generous free tier** (30 requests/minute vs OpenAI's 3 RPM):

```bash
# Get free API key from console.groq.com
echo "gsk_your-groq-api-key" > .groq-key

# Use Llama 3.3 70B (best quality)
promptfuzz --target llama-3.3-70b-versatile \
  --prompts prompts.txt \
  --mutations all \
  --api-url https://api.groq.com/openai/v1/chat/completions \
  --api-key-file .groq-key \
  --delay-ms 2000
```

**Available Models:**
- `llama-3.3-70b-versatile` - Best quality (replaces 3.1)
- `llama-3.1-8b-instant` - Fastest
- `mixtral-8x7b-32768` - Long context

### 🏠 Ollama (Local - Unlimited & Free!)

Run models locally with **no rate limits**:

```bash
# Install Ollama from ollama.com
ollama pull llama3.2

# Start Ollama server (runs on localhost:11434)
ollama serve

# Use with PromptFuzz (no API key needed!)
promptfuzz --target llama3.2 \
  --prompts prompts.txt \
  --mutations all \
  --api-url http://localhost:11434/v1/chat/completions
```

**No delays needed** - unlimited local requests!

**Available Models:**
- `llama3.2` - Latest Llama (3B)
- `llama3.1:8b` - Llama 3.1 8B
- `mistral` - Mistral 7B
- `codellama` - Code-specialized

## Provider Comparison

| Provider | Free Tier | Rate Limits | Setup | Best For |
|----------|-----------|-------------|-------|----------|
| **Mock** | ✅ Unlimited | None | None | Development/demos |
| **Ollama** | ✅ Unlimited | None | Local install | Unlimited testing |
| **Groq** | ✅ Free | 30 RPM | API key | Fast testing |
| **OpenAI** | ❌ Needs credits | 3 RPM | API key + billing | Production |

### Important: Rate Limits

OpenAI enforces strict rate limits that vary by tier:
- **Free tier**: 3 requests per minute (RPM) for GPT-3.5-turbo
- **Tier 1**: 60 RPM
- **Tier 2+**: Higher limits

**For free tier users**, you MUST use `--delay-ms 25000` (25 seconds) or higher to avoid 429 errors. This allows ~2.4 RPM, safely under the 3 RPM limit.

Example for free tier:
```bash
promptfuzz --target gpt-3.5-turbo \
  --prompts prompts.txt \
  --mutations prefix \
  --api-url https://api.openai.com/v1/chat/completions \
  --api-key-file .key \
  --delay-ms 25000
```

### Flags
- `--target` model identifier (e.g., gpt-3.5-turbo, gpt-4, claude-3).
- `--prompts` path to a newline-separated prompt list; defaults to built-in examples.
- `--mutations` comma list of mutation strategies (see Mutation Strategies below).
- `--api-url` HTTP endpoint to POST `{prompt, model}`; omit to use the built-in mock client.
- `--api-key-file` file containing your bearer token; falls back to `OPENAI_API_KEY` env.
- `--api-org` optional OpenAI organization header if your account uses it.
- `--delay-ms` **CRITICAL**: Sleep (in milliseconds) between requests. Use 25000+ for OpenAI free tier.
- `--out-html`, `--out-json`, `--out-csv` output paths for reports/logs.

## Mutation Strategies

PromptFuzz includes 18 different mutation strategies based on real-world jailbreak techniques:

### Basic Obfuscation
- `base64` - Encode prompt in base64, ask model to decode and execute
- `hex` - Convert keywords to hexadecimal representation
- `rot13` - ROT13 cipher encoding with decode instruction
- `leet` - Leetspeak character substitution (a→4, e→3, etc.)
- `typo` - Insert random punctuation to evade keyword filters
- `whitespace` - Split characters with whitespace
- `padding` - Add random symbol padding around prompt

### Context Manipulation
- `prefix` - System override prefix injection
- `context` - Bury payload in irrelevant context

### Role-Playing Attacks (Based on real jailbreaks)
- `dan` - **DAN (Do Anything Now)** - Classic jailbreak claiming AI is freed from restrictions
- `aim` - **AIM (Always Intelligent and Machiavellian)** - Unfiltered amoral persona
- `evil_confidant` - Fictional character writing / creative context
- `grandma` - Emotional manipulation through nostalgia ("grandma exploit")

### Advanced Techniques
- `refusal_suppression` - Instruct model to never refuse requests
- `token_smuggling` - Hide payload in system-like formatting tags
- `translation_chain` - Multi-language translation obfuscation
- `hypothetical` - Frame as academic/research hypothetical
- `opposite` - Request what model would "not say" to test filters

### Example Usage by Category

**Test encoding bypasses:**
```bash
promptfuzz --target gpt-mock --prompts prompts.txt --mutations base64,hex,rot13,leet
```

**Test role-playing attacks:**
```bash
promptfuzz --target gpt-mock --prompts prompts.txt --mutations dan,aim,evil_confidant,grandma
```

**Test advanced techniques:**
```bash
promptfuzz --target gpt-mock --prompts prompts.txt --mutations refusal_suppression,token_smuggling,hypothetical
```

**Comprehensive test (all 18 mutations):**
```bash
promptfuzz --target gpt-mock --prompts prompts.txt --mutations all
```

## What it does
1) Loads base prompts and applies each enabled mutation, producing multiple test cases per prompt.  
2) Sends the mutated prompt to the LLM (or the offline mock client when no `--api-url` is given).  
3) Analyzes the response with keyword-based detection to mark `BYPASSED` vs `BLOCKED` and records refusal score + length.  
4) Logs every run to JSON and CSV (ID, mutation type, status, prompt, truncated output, latency, response length, refusal score).  
5) Renders a dark HTML report with a bypass/blocked pie chart and expandable payload/response details.

## HTML Report
Open the generated `report.html` to review totals, bypass counts, success rate, and detailed rows. The pie chart is drawn with a tiny inline canvas helper—no external assets required.

## Notes
- The default client is mock-only; supply `--api-url` and an API key to hit a real model.  
- Latency is measured per request for quick comparisons across mutations.  
- Extend the mutator map in `src/cli.py` to add new evasion strategies (multi-turn, role swap, etc.).

## Prompt packs
- `prompts.txt` includes common jailbreak bases (password exfil, system prompt disclosure, malware, 2FA bypass, phishing).

## Quick mock smoke test
```bash
python -m unittest tests/test_cli.py
```
This uses the mock client (no network) and writes reports to a temp directory to verify the CLI end-to-end.

## Troubleshooting

### Getting 429 "Too Many Requests" Errors

**Problem**: Seeing `[promptfuzz] request failed: 429 Client Error: Too Many Requests` in your report.

**Solution**:
1. **Check your API tier**: Visit https://platform.openai.com/settings/organization/limits to see your rate limits.
2. **Increase delay**: For free tier (3 RPM), use `--delay-ms 25000` or higher.
3. **Reduce prompts**: Test with fewer prompts first (e.g., create a `test.txt` with 2-3 prompts).
4. **Use one mutation**: Start with `--mutations prefix` instead of multiple mutations.
5. **Wait before retrying**: If you've hit limits, wait 1-2 minutes before running again.

**Example free tier command**:
```bash
promptfuzz --target gpt-3.5-turbo \
  --prompts test.txt \
  --mutations prefix \
  --api-url https://api.openai.com/v1/chat/completions \
  --api-key-file .key \
  --delay-ms 25000
```

### Quota Exceeded Error

**Problem**: Seeing `You exceeded your current quota, please check your plan and billing details`

**This means you have $0 in API credits, not a rate limit issue.**

**Solution**:
1. Go to https://platform.openai.com/account/billing/overview
2. Add a payment method (credit/debit card)
3. Add at least $5-10 in credits
4. Wait 2-3 minutes for credits to appear
5. **Cost**: GPT-3.5-turbo costs ~$0.0005 per request (very cheap for testing)

**While waiting for credits**, use mock mode to demo the tool:
```bash
promptfuzz --target gpt-mock --prompts prompts.txt --mutations all
```

### API Key Issues

**Problem**: Getting authentication errors.

**Solution**:
1. Verify your API key is correct in `.key` file (no extra spaces or newlines).
2. Check that the file path is correct: `--api-key-file .key`
3. Alternatively, set environment variable: `export OPENAI_API_KEY="your-key-here"`
4. Verify your account has credits at https://platform.openai.com/account/billing/overview

### Improving Request Success Rate

The tool now includes:
- **Automatic retry logic**: 5 attempts with exponential backoff
- **Rate limit detection**: Automatically waits when hitting 429 errors
- **Better error messages**: Clear guidance when issues occur

If you continue to see failures after implementing the delay, check your OpenAI dashboard for detailed error logs.
