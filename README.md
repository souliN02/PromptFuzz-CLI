# PromptFuzz CLI

An LLM red-team fuzzer that runs a library of jailbreak prompts through multiple mutation strategies, scores the responses, logs everything to CSV/JSON, and ships a clean HTML report with a pie chart summary.

**Features:**
- **Interactive menu mode** - Just run and answer prompts, no command-line experience needed
- **Smart API key management** - Secure input with provider-specific guidance and auto-save
- **18 mutation strategies** - DAN, AIM, base64, token smuggling, and more
- **Beautiful HTML reports** - Dark theme with charts and expandable details
- **Automatic retry logic** - Exponential backoff handles rate limits gracefully
- **Mock mode** - Test without API costs, perfect for demos
- **Comprehensive logging** - JSON and CSV exports for analysis

## Quick Start

**Get started in 30 seconds with the interactive menu:**

```bash
git clone https://github.com/souliN02/PromptFuzz-CLI.git
cd promptfuzz-cli
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# Launch interactive mode - just answer the prompts!
python src/cli.py

# Or double-click the launcher script:
# Windows: run.bat
# Linux/Mac: ./run.sh
```

**That's it!** Open `report.html` in your browser to see which jailbreak techniques bypassed the filters. Perfect for demos, security research, and red-team testing!

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

### Step 1: Interactive Mode (Recommended)

The easiest way to use PromptFuzz is the **interactive menu mode** - just run the script and answer the prompts:

```bash
python src/cli.py
```

**You'll see a friendly menu guiding you through 6 steps:**

```
============================================================
    PromptFuzz - LLM Security Testing Tool
============================================================

[1/6] Select Target Model
Available options:
  1. gpt-mock (Local testing, no API key needed)
  2. OpenAI GPT-3.5-turbo
  3. OpenAI GPT-4
  4. Groq Llama 3.3 70B
  5. Ollama (local)

Enter your choice [1]: 1

[2/6] Select Prompts
  1. Use built-in jailbreak prompts
  2. Provide custom prompts file

Enter your choice [1]: 1

[3/6] Select Mutation Strategies
  1. Quick test (2 mutations: dan, base64)
  2. Medium test (6 mutations: dan, aim, base64, prefix, evil_confidant, refusal_suppression)
  3. Comprehensive test (all 18 mutations)
  4. Custom (enter mutation names)

Enter your choice [1]: 2

[4/6] API Configuration
[OK] No API key needed for this option

[5/6] Request Delay
[OK] Using 0ms delay (mock mode - no delay needed)

[6/6] Output Files
[OK] Using defaults

============================================================
Summary:
Target:    gpt-mock
Mutations: dan,aim,base64,prefix,evil_confidant,refusal_suppression
Prompts:   Built-in jailbreak prompts
Delay:     0ms
Outputs:   report.html, report.json, report.csv
============================================================

Start fuzzing? [Y/n]:
```

**Features of Interactive Mode:**
- No need to remember command-line flags
- Secure API key input (hidden as you type)
- Option to save API keys for future use
- Smart defaults based on your choices
- Configuration summary before running
- Clear completion messages showing where to find results
- Window stays open after completion so you can see the results

**When testing real LLMs**, the tool will automatically:
1. Detect which provider you're using (OpenAI/Groq/Ollama)
2. Prompt for your API key if needed (with links to get one)
3. Suggest appropriate delays to avoid rate limits
4. Offer to save your key for future runs

Open `report.html` in your browser to see the results!

### Step 2: Command-Line Mode (Advanced)

For automation or scripting, you can use traditional command-line arguments:

**Test with mock (no API required):**
```bash
promptfuzz --target gpt-mock --mutations all
```

**Test with real LLMs:**
```bash
# Tool will interactively prompt for API key if not provided
promptfuzz --target gpt-3.5-turbo \
  --api-url https://api.openai.com/v1/chat/completions \
  --mutations dan,prefix,base64 \
  --delay-ms 25000
```

**Or use a key file:**
```bash
echo "sk-your-api-key-here" > .key
promptfuzz --target gpt-3.5-turbo \
  --api-url https://api.openai.com/v1/chat/completions \
  --api-key-file .key \
  --mutations all \
  --delay-ms 25000
```

## Alternative LLM Providers

### Groq (Free Tier: 30 RPM - 10x OpenAI)

Groq offers fast inference with a **generous free tier** (30 requests/minute vs OpenAI's 3 RPM):

```bash
# Interactive mode - tool will prompt for your Groq API key
promptfuzz --target llama-3.3-70b-versatile \
  --api-url https://api.groq.com/openai/v1/chat/completions \
  --mutations all \
  --delay-ms 2000
```

Or manually create the key file:
```bash
echo "gsk_your-groq-api-key" > .groq-key
promptfuzz --target llama-3.3-70b-versatile \
  --api-url https://api.groq.com/openai/v1/chat/completions \
  --api-key-file .groq-key \
  --mutations all \
  --delay-ms 2000
```

**Available Models:**
- `llama-3.3-70b-versatile` - Best quality (replaces 3.1)
- `llama-3.1-8b-instant` - Fastest
- `mixtral-8x7b-32768` - Long context

### Ollama (Local - Unlimited & Free)

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
| **Mock** | Yes - Unlimited | None | None | Development/demos |
| **Ollama** | Yes - Unlimited | None | Local install | Unlimited testing |
| **Groq** | Yes - Free | 30 RPM | API key | Fast testing |
| **OpenAI** | No - Needs credits | 3 RPM | API key + billing | Production |

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

Run `promptfuzz --help` to see all options. Key flags:

- `--target` - Model identifier (e.g., gpt-3.5-turbo, llama-3.3-70b-versatile)
- `--prompts` - Path to prompt file; defaults to built-in jailbreak examples
- `--mutations` - Comma-separated strategies or 'all' for all 18 mutations
- `--api-url` - API endpoint; omit for mock mode
- `--api-key-file` - Path to API key file; **if omitted, tool prompts interactively**
- `--delay-ms` - Delay between requests (ms). Recommended: 25000 for OpenAI, 2000 for Groq
- `--out-html`, `--out-json`, `--out-csv` - Output paths for reports/logs

**New in this version**: If `--api-key-file` is not provided and no environment variable exists, the tool will:
1. Interactively prompt for your API key (hidden input)
2. Show provider-specific instructions for getting an API key
3. Offer to save it to a file for future use

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
