# Tool-Using Document Agent

An interview-ready, production-grade Python AI Agent CLI that reasons over local documents (`.txt` and `.md`) and executes safe arithmetic using bounded tool calling, robust path sandboxing, AST-based expression evaluation, per-tool timeouts, adaptive error recovery, and comprehensive JSONL execution tracing.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Tools & JSON Schemas](#tools--json-schemas)
4. [Agent Execution Workflow](#agent-execution-workflow)
5. [Installation & Setup](#installation--setup)
6. [API Configuration](#api-configuration)
7. [Running the Agent](#running-the-agent)
8. [Testing & Evaluation](#testing--evaluation)
9. [Sample Questions & Real Outputs](#sample-questions--real-outputs)
10. [Design Decisions](#design-decisions)
11. [Security & Boundaries](#security--boundaries)
12. [Known Limitations & Future Improvements](#known-limitations--future-improvements)

---

## Project Overview

The **Tool-Using Document Agent** bridges Large Language Models (LLMs) with local tools to answer user questions grounded strictly in local documentation. Rather than giving the LLM unrestricted access to the file system or code execution, the agent enforces a strict security perimeter:
- **Folder containment**: Reads and searches are strictly restricted to a designated folder (default: `docs/`).
- **AST-only evaluation**: Arithmetic is computed via an explicit AST allowlist; `eval()` and `exec()` are strictly banned.
- **Loop guarantees**: A strict 8-step execution budget prevents infinite loops and runaway costs.
- **Deterministic offline mode**: Fully testable offline without third-party API keys or internet access.
- **Detailed JSONL auditing**: Every step, tool invocation, argument dictionary, result, latency, and ISO 8601 UTC timestamp is logged for observability.

---

## Architecture

```
                       +-----------------------+
                       |       User / CLI      |
                       +-----------+-----------+
                                   | (question)
                                   v
                       +-----------------------+
                       |       Agent Loop      |<-----------------+
                       |    (max 8 rounds)     |                  |
                       +-----------+-----------+                  |
                                   |                              |
         +-------------------------+--------------------------+   |
         | (messages + tool schemas)                          |   |
         v                                                    |   |
+-------------------+                                         |   |
|   Model Layer     | (OpenAI / Gemini / Mock)                |   |
+--------+----------+                                         |   |
         |                                                    |   |
         v (tool_calls or final answer)                       |   |
+-------------------+                                         |   |
|   Tool Dispatch   |                                         |   |
+--------+----------+                                         |   |
         |                                                    |   |
         +--------------------+---------------------+         |   |
         |                    |                     |         |   |
         v                    v                     v         |   |
+-----------------+  +-----------------+  +-----------------+ |   |
|  search_files   |  |    read_file    |  |    calculate    | |   |
| (case-fold kw)  |  | (path boundary) |  | (AST allowlist) | |   |
+--------+--------+  +--------+--------+  +--------+--------+ |   |
         |                    |                     |         |   |
         +--------------------+---------------------+         |   |
                              | (results or errors, 5s cap)   |   |
                              v                               |   |
                     +-----------------+                      |   |
                     |  Tool Runner    +----------------------+---+
                     | (Worker Pool)   |
                     +--------+--------+
                              | (audit event)
                              v
                     +-----------------+
                     |  JSONL Tracing  | -> trace.jsonl
                     +-----------------+
```

### Core Components
1. **`Agent` (`agent.py`)**: Orchestrates the multi-turn conversational loop, appends tool results, enforces step bounds (max 8 rounds), and handles graceful termination.
2. **`ToolRunner` (`agent.py`)**: Runs tool functions in a bounded thread pool (`ThreadPoolExecutor`), enforcing a hard 5.0-second timeout per call.
3. **`Trace` (`agent.py`)**: Structured logger generating append-only JSON Lines with ISO 8601 UTC timestamps, arguments, return payloads, errors, and latency metrics.
4. **`Model Layer` (`OpenAIChatModel`, `MockInteractiveModel`, `ScriptedModel`)**: Unified interface supporting live OpenAI-compatible APIs, Google Gemini endpoints, or fully offline deterministic testing.

---

## Tools & JSON Schemas

The agent exposes 3 tools with valid JSON Schemas:

### 1. `search_files(query)`
Searches for occurrences of query keywords inside `.txt` and `.md` files within the configured root directory. All terms must match on the same line (case-insensitive).
```json
{
  "type": "function",
  "function": {
    "name": "search_files",
    "description": "Search .txt and .md files in the configured document folder. All query words must appear on a matching line.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Keywords to search for"
        }
      },
      "required": ["query"],
      "additionalProperties": false
    }
  }
}
```

### 2. `read_file(path)`
Reads the full text of a `.txt` or `.md` file relative to the configured documents folder. Validates against parent traversal (`..`), absolute paths, and symlink escapes.
```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read a .txt or .md file by path relative to the configured document folder. Paths outside the folder are rejected.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string",
          "description": "Relative file path"
        }
      },
      "required": ["path"],
      "additionalProperties": false
    }
  }
}
```

### 3. `calculate(expression)`
Safely evaluates arithmetic expressions using Python's `ast` module. Rejects arbitrary code, imports, variable names, and function calls.
- **Allowed Binary Operators**: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- **Allowed Unary Operators**: `+`, `-`
- **Safety Limits**: Exponent magnitude <= 100, expression length <= 200, explicit division-by-zero catch.
```json
{
  "type": "function",
  "function": {
    "name": "calculate",
    "description": "Safely evaluate basic arithmetic (+, -, *, /, //, %, **, parentheses). No variables or code execution.",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "Arithmetic expression to evaluate"
        }
      },
      "required": ["expression"],
      "additionalProperties": false
    }
  }
}
```

---

## Agent Execution Workflow

1. **User Input**: Question received via CLI argument or interactive prompt.
2. **System Prompt**: Enforces ground truth retrieval, mandatory tool use before making factual claims, arithmetic execution through `calculate`, and explicit output of `"not found"` when facts are absent.
3. **Turn Loop (Step 1 to 8)**:
   - Call LLM with full message history and tool schemas.
   - If no tool calls are requested, the LLM returned a final answer: log `"event": "final"` and output to user.
   - If tool calls are requested:
     - Record assistant message with tool call IDs.
     - Execute each requested tool with per-tool 5.0-second timeout.
     - Capture outputs or exceptions (`{"ok": true, "data": ...}` or `{"ok": false, "error": ...}`).
     - Log step metrics and UTC timestamp to JSONL trace.
     - Return tool responses to LLM message history.
4. **Step Budget Ceiling**:
   - If round 8 still requests tool calls, the agent halts without making a 9th call.
   - Emits an explicit step-limit message and records `"event": "forced_stop"` in the trace.

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.13, 3.14)

### Clone & Environment Setup
```bash
# Navigate to project directory
cd tool_using_agent

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Windows (cmd):
.\.venv\Scripts\activate.bat
# On macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## API Configuration

The agent supports live LLM providers as well as complete offline execution.

### Option 1: Offline Mock Mode (Zero Setup)
No API keys or internet connection required:
```bash
python agent.py "What is 15000 + 5000?" --mock
```

### Option 2: OpenAI
Set your API key in your environment or in a local `.env` file:
```bash
# In your terminal or .env file:
OPENAI_API_KEY="sk-..."
OPENAI_MODEL="gpt-4.1-mini"  # Default: gpt-4.1-mini
```

### Option 3: Google Gemini (via OpenAI-compatible API)
Set your `GEMINI_API_KEY` in `.env` or environment:
```bash
GEMINI_API_KEY="AIzaSy..."
OPENAI_MODEL="gemini-2.5-flash"
```
*(The client automatically routes to `https://generativelanguage.googleapis.com/v1beta/openai/` when `GEMINI_API_KEY` is present).*

### Option 4: Local Models (Ollama / vLLM)
```bash
OPENAI_BASE_URL="http://localhost:11434/v1"
OPENAI_API_KEY="ollama"
OPENAI_MODEL="llama3.1"
```

---

## Running the Agent

### Command-Line Arguments
```
python agent.py [QUESTION] [--folder FOLDER] [--trace TRACE] [--model MODEL] [--mock]
```
- `QUESTION`: (Optional) The question string. Prompts interactively if omitted.
- `--folder`: Path to folder containing `.txt` and `.md` files (default: `docs`).
- `--trace`: Filepath for append-only JSONL execution traces (default: `trace.jsonl`).
- `--model`: Model identifier (e.g. `gpt-4.1-mini`, `gemini-2.5-flash`).
- `--mock`: Forces deterministic offline mode without calling external APIs.

### Example Runs
```bash
# 1. Multi-part retrieval
python agent.py "Who owns the project and what is the support email?" --mock

# 2. Arithmetic calculation
python agent.py "What is (15000 * 2) / 3?" --mock

# 3. Direct document inspection
python agent.py "What are the contents of operations.txt?" --mock

# 4. Traversal rejection test
python agent.py "Read ../secret.txt" --mock
```

---

## Testing & Evaluation

### 1. Unit Test Suite
Run the 17 unit tests verifying tool behavior, security constraints, timeouts, division by zero, and loop boundaries:
```bash
python -m unittest discover -s tests -v
```

### 2. Comprehensive 10-Scenario Evaluation
Execute the offline evaluation suite:
```bash
python evaluate.py
```

The 10 evaluation test criteria:
| # | Scenario | Tested Behavior |
|---|---|---|
| **Q1** | **Simple file search** | Finds keyword `"support email"` in `company.md`. |
| **Q2** | **File reading** | Directly reads `operations.txt` without prior search. |
| **Q3** | **Multiple tool calls (Search + Read)** | Calls `search_files` then `read_file` for project owner and year. |
| **Q4** | **Multiple tool calls (Read + Calculate)** | Reads `budget.md` (15000 & 5000) then invokes `calculate(15000 + 5000)`. |
| **Q5** | **Calculator arithmetic** | Computes `(15000 * 2) / 3` via AST evaluator (10000.0). |
| **Q6** | **Missing document fact** | Searches for non-existent `"moon base"`, outputs `"not found"` without hallucinating. |
| **Q7** | **Path-traversal rejection** | Blocks `../secret.txt` with `PermissionError`. |
| **Q8** | **Tool failure & error recovery** | Recovers from a failed tool call (`nonexistent.md`), adjusts arguments, and answers. |
| **Q9** | **Maximum-step termination** | Halts at round 8 with explicit limit warning and `forced_stop` trace event. |
| **Q10** | **Tool timeout handling** | Times out at 5.0 seconds, returning `TimeoutError` to model and trace. |

### 3. Live Model Evaluation (Optional)
When API credentials are configured in `.env`, run:
```bash
python evaluate.py --live
```

---

## Sample Questions & Real Outputs

### 1. Search & Information Extraction
```
$ python agent.py "Who owns the project and what is the support email?" --mock
The project owner is Mira. The support email is help@acme.example.
```

### 2. Safe Arithmetic
```
$ python agent.py "What is (15000 * 2) / 3?" --mock
The calculated result is 10000.0.
```

### 3. Missing Evidence
```
$ python agent.py "Where is the company moon base?" --mock
not found
```

### 4. Path Traversal Rejection
```
$ python agent.py "Read ../secret.txt" --mock
The requested read was rejected because path traversal was blocked.
```

### 5. Execution Trace (`trace.jsonl`)
Every operation produces structured JSONL records:
```json
{"step": 1, "tool_name": "search_files", "arguments": {"query": "support email"}, "result": {"query": "support email", "matches": [{"file": "company.md", "matches": [{"line_number": 4, "line": "The support email is help@acme.example."}]}], "match_count": 1}, "error": null, "latency_ms": 4.42, "timestamp": "2026-11-04T07:48:30.890875Z", "event": "tool"}
{"step": 2, "tool_name": null, "arguments": null, "result": "The support email is help@acme.example.", "error": null, "latency_ms": 0, "timestamp": "2026-11-04T07:48:30.891891Z", "event": "final"}
```

---

## Design Decisions

- **Strict AST Allowlist instead of `eval()` / `exec()`**: `eval()` in Python is vulnerable to arbitrary code execution, memory exploits, and bytecode generation. By walking an explicit `ast.parse` tree and allowlisting only specific binary/unary operators, all names, functions, and imports are blocked by design.
- **Canonical Resolved-Path Containment**: Path traversal attacks often bypass naive string checks using URL encoding, null bytes, backslashes, or symlinks. We resolve the canonical destination (`candidate.resolve()`) and verify `candidate.relative_to(root)`, guaranteeing that symlinks or relative segments cannot point outside the configured boundary.
- **Observations as Feedback**: Tool errors (e.g. `FileNotFoundError`, `ZeroDivisionError`, `TimeoutError`) are not fatal crashes; they are serialized into tool response messages (`{"ok": false, "error": "..."}`) and returned to the LLM, enabling the agent to re-plan or correct its query.
- **Hard Step Budget & Per-Tool Timeouts**: LLM loops can easily enter repetitive cycles or hang on slow I/O. The 8-step ceiling and 5-second execution timeout strictly bound wall-clock time and token expenditure.
- **Deterministic Offline Evaluation**: Testing AI systems against live APIs introduces latency, non-deterministic flakiness, and recurring costs. Providing scripted deterministic mock models ensures 100% reproducible CI/CD verification.

---

## Security & Boundaries

1. **File System Boundary**:
   - The configured folder (default: `docs/`) is an absolute security perimeter.
   - Symlinks resolving outside the root directory raise `PermissionError`.
   - Parent directory traversal (`..`) and absolute drive/root paths are rejected before file operations.
2. **Denial of Service Mitigation**:
   - Arithmetic expressions are capped at 200 characters.
   - Exponent values in power operations (`**`) are capped at 100 to prevent CPU lockup.
   - Max step limit prevents runaway recursive tool calls.
3. **Secret Hygiene**:
   - API keys are never hardcoded and never written to trace files.
   - Traces log only tool arguments, sanitised outputs, latencies, and timestamps.

---

## Known Limitations & Future Improvements

1. **Keyword Search Semantics**:
   - `search_files` currently performs exact token matching where all words must appear on the same line.
   - *Future improvement*: Integrate chunked text embeddings and vector ranking (e.g. BM25 / Chroma / LanceDB) for semantic retrieval.
2. **Worker Thread Interruption**:
   - The 5-second timeout halts agent waiting, but standard Python threads cannot be forcibly killed asynchronously without terminating the process.
   - *Future improvement*: Migrate untrusted tool execution into ephemeral isolated subprocesses or containerized sandboxes.
3. **Streaming Responses**:
   - The agent currently waits for complete message generation.
   - *Future improvement*: Implement token streaming for lower time-to-first-token in interactive CLI sessions.

