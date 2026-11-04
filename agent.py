"""A bounded, tool-using document agent. Python 3.10+; optional OpenAI SDK."""
from __future__ import annotations
import argparse, ast, json, operator, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

MAX_STEPS = 8
TOOL_TIMEOUT_SECONDS = 5


def _utc_now_iso() -> str:
    """Return the current UTC timestamp formatted as ISO 8601 with Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_dotenv(filepath: str | Path = '.env') -> None:
    """Load key-value environment variables from a .env file if it exists."""
    p = Path(filepath)
    if not p.is_file():
        return
    try:
        for line in p.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


def _safe_root(root: str | Path) -> Path:
    p = Path(root).expanduser().resolve()
    if not p.is_dir():
        raise ValueError(f"Document folder does not exist: {p}")
    return p


def _resolve_document_path(path: str, root: Path) -> Path:
    requested = Path(path)
    windows_path = PureWindowsPath(path)
    if requested.is_absolute() or windows_path.root or windows_path.drive:
        raise ValueError("Absolute paths are not allowed; use a path relative to the document folder")
    if ".." in requested.parts or ".." in windows_path.parts:
        raise PermissionError("Path traversal rejected: parent-directory segments are not allowed")
    candidate = (root / requested).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise PermissionError("Path traversal rejected: requested path is outside the document folder")
    return candidate


def search_files(query: str, folder: str | Path) -> dict[str, Any]:
    """Case-insensitive keyword search; all query terms must occur on a matching line."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    root = _safe_root(folder)
    terms = query.casefold().split()
    matches = []
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in {'.txt', '.md'} or not path.is_file():
            continue
        try:
            resolved = path.resolve()
            resolved.relative_to(root)
            if not resolved.is_file():
                continue
            lines = resolved.read_text(encoding='utf-8').splitlines()
        except (OSError, RuntimeError, UnicodeError, ValueError):
            continue
        found = [{"line_number": i, "line": line} for i, line in enumerate(lines, 1)
                 if all(term in line.casefold() for term in terms)]
        if found:
            matches.append({"file": path.relative_to(root).as_posix(), "matches": found})
    return {"query": query, "matches": matches, "match_count": sum(len(x['matches']) for x in matches)}


def read_file(path: str, folder: str | Path) -> dict[str, str]:
    """Read a document using a relative path contained within the folder."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    root = _safe_root(folder)
    candidate = _resolve_document_path(path, root)
    if candidate.suffix.lower() not in {'.txt', '.md'}:
        raise ValueError("Only .txt and .md files can be read")
    if not candidate.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return {"path": candidate.relative_to(root).as_posix(), "content": candidate.read_text(encoding='utf-8')}


_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
           ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression): return _eval_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float): return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ZeroDivisionError("division by zero")
        if isinstance(node.op, ast.Pow) and abs(right) > 100: raise ValueError("Exponent too large")
        result = _BINOPS[type(node.op)](left, right)
        if isinstance(result, complex) or abs(result) > 1e100: raise ValueError("Result out of allowed range")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported arithmetic expression")

def calculate(expression: str) -> dict[str, Any]:
    if not isinstance(expression, str) or not expression.strip(): raise ValueError("expression must be non-empty")
    if len(expression) > 200: raise ValueError("Expression is too long")
    try: tree = ast.parse(expression, mode='eval')
    except SyntaxError as e: raise ValueError("Invalid arithmetic expression") from e
    return {"expression": expression, "result": _eval_node(tree)}


TOOL_SCHEMAS = [
    {"type":"function", "function":{"name":"search_files", "description":"Search .txt and .md files in the configured document folder. All query words must appear on a matching line.", "parameters":{"type":"object","properties":{"query":{"type":"string","description":"Keywords to search for"}},"required":["query"],"additionalProperties":False}}},
    {"type":"function", "function":{"name":"read_file", "description":"Read a .txt or .md file by path relative to the configured document folder. Paths outside the folder are rejected.", "parameters":{"type":"object","properties":{"path":{"type":"string","description":"Relative file path"}},"required":["path"],"additionalProperties":False}}},
    {"type":"function", "function":{"name":"calculate", "description":"Safely evaluate basic arithmetic (+, -, *, /, //, %, **, parentheses). No variables or code execution.", "parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"],"additionalProperties":False}}}
]

class Trace:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
    def write(self, record: dict[str, Any]):
        if "timestamp" not in record:
            record["timestamp"] = _utc_now_iso()
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

class ToolRunner:
    def __init__(self, folder: str | Path, timeout: float = TOOL_TIMEOUT_SECONDS):
        self.folder, self.timeout = folder, timeout
        self.pool = ThreadPoolExecutor(max_workers=4)
    def run(self, name: str, args: dict[str, Any]) -> Any:
        if not isinstance(args, dict):
            raise ValueError("Tool arguments must be a dictionary")
        if name == "search_files":
            if "query" not in args:
                raise ValueError("Missing required argument: 'query'")
            func = lambda: search_files(args["query"], self.folder)
        elif name == "read_file":
            if "path" not in args:
                raise ValueError("Missing required argument: 'path'")
            func = lambda: read_file(args["path"], self.folder)
        elif name == "calculate":
            if "expression" not in args:
                raise ValueError("Missing required argument: 'expression'")
            func = lambda: calculate(args["expression"])
        else:
            raise ValueError(f"Unknown tool: '{name}'. Available tools: search_files, read_file, calculate")
        future = self.pool.submit(func)
        try:
            return future.result(timeout=self.timeout)
        except FuturesTimeout:
            future.cancel()
            raise TimeoutError(f"Tool '{name}' exceeded {self.timeout:g}-second timeout")
    def close(self):
        self.pool.shutdown(wait=False)

class OpenAIChatModel:
    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None):
        load_dotenv()
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("Install dependency with: pip install openai") from e
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        base_url = base_url or os.getenv('OPENAI_BASE_URL')
        if not api_key and os.getenv('GEMINI_API_KEY'):
            api_key = os.getenv('GEMINI_API_KEY')
            if not base_url:
                base_url = 'https://generativelanguage.googleapis.com/v1beta/openai/'
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        if base_url:
            client_kwargs['base_url'] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')

    def complete(self, messages: list[dict[str, Any]]) -> Any:
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice='auto'
        )

class Agent:
    def __init__(self, model: Any, folder: str | Path, trace_path: str | Path = 'trace.jsonl', max_steps: int = MAX_STEPS):
        self.model = model
        self.runner = ToolRunner(folder)
        self.trace = Trace(trace_path)
        self.max_steps = min(max(int(max_steps), 1), MAX_STEPS)

    def answer(self, question: str) -> str:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You answer questions using only the supplied local documents and the available tools. "
                    "Search before making document claims. Read files when context is needed. "
                    "Use calculate for arithmetic. If evidence is absent, say 'not found'. "
                    "Never invent facts. Tool errors are recoverable: explain or adjust arguments and retry when useful. "
                    "Stop and give a concise final answer."
                )
            },
            {"role": "user", "content": question}
        ]

        for step in range(1, self.max_steps + 1):
            response = self.model.complete(messages)
            msg = response.choices[0].message
            tool_calls = getattr(msg, 'tool_calls', None) or []

            if not tool_calls:
                answer = msg.content or "I could not produce an answer."
                self.trace.write({
                    "step": step,
                    "tool_name": None,
                    "arguments": None,
                    "result": answer,
                    "error": None,
                    "latency_ms": 0,
                    "timestamp": _utc_now_iso(),
                    "event": "final"
                })
                return answer

            # Append assistant tool-call message in OpenAI-compatible format
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.function.name,
                            "arguments": c.function.arguments
                        }
                    }
                    for c in tool_calls
                ]
            })

            for call in tool_calls:
                start = time.perf_counter()
                name = call.function.name
                args: dict[str, Any] = {}
                try:
                    raw_args = call.function.arguments or '{}'
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Malformed JSON arguments: {raw_args}") from exc
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                    result = self.runner.run(name, args)
                    error = None
                    payload = {"ok": True, "data": result}
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    result = None
                    payload = {"ok": False, "error": error}

                latency = round((time.perf_counter() - start) * 1000, 2)
                self.trace.write({
                    "step": step,
                    "tool_name": name,
                    "arguments": args,
                    "result": result,
                    "error": error,
                    "latency_ms": latency,
                    "timestamp": _utc_now_iso(),
                    "event": "tool"
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(payload, ensure_ascii=False, default=str)
                })

            if step == self.max_steps:
                answer = f"I reached the {self.max_steps}-step tool limit before I could complete the answer. Please narrow the question or try again."
                self.trace.write({
                    "step": step,
                    "tool_name": None,
                    "arguments": None,
                    "result": answer,
                    "error": "step_limit_reached",
                    "latency_ms": 0,
                    "timestamp": _utc_now_iso(),
                    "event": "forced_stop"
                })
                return answer

        return "I could not complete the answer within the step limit."


# A deterministic model used only for offline evaluation of the tools and loop.
class MockMessage:
    def __init__(self, content: str | None = None, tool_calls: list[Any] | None = None):
        self.content = content
        self.tool_calls = tool_calls or []


class MockCall:
    def __init__(self, cid: str, name: str, args: dict[str, Any]):
        self.id = cid
        self.function = type('F', (), {'name': name, 'arguments': json.dumps(args)})()


class MockResponse:
    def __init__(self, msg: MockMessage):
        self.choices = [type('C', (), {'message': msg})()]


class ScriptedModel:
    def __init__(self, scripts: list[Any]):
        self.scripts = list(scripts)
        self.calls = 0
        self.message_history: list[list[dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, Any]]) -> MockResponse:
        self.calls += 1
        self.message_history.append(list(messages))
        if not self.scripts:
            return MockResponse(MockMessage("I don't have enough information."))
        item = self.scripts.pop(0)
        if callable(item):
            return MockResponse(MockMessage(item(messages)))
        if isinstance(item, str):
            return MockResponse(MockMessage(item))
        return MockResponse(MockMessage(tool_calls=[
            MockCall(f"call_{len(self.scripts)}_{i}", n, a) for i, (n, a) in enumerate(item)
        ]))


class MockInteractiveModel:
    """Interactive mock model that parses user questions and issues appropriate tool calls."""

    def __init__(self):
        self.round = 0

    def complete(self, messages: list[dict[str, Any]]) -> MockResponse:
        self.round += 1
        last_msg = messages[-1]

        # First round: inspect question
        if last_msg.get('role') == 'user':
            user_text = last_msg.get('content', '').lower()

            # Pure arithmetic check
            has_calc_word = any(k in user_text for k in ['calculate', 'compute', 'what is', 'how much is'])
            has_operator = any(op in user_text for op in ['+', '-', '*', '/', '%'])
            has_digit = any(c.isdigit() for c in user_text)
            if (has_calc_word or has_operator) and has_digit and not any(k in user_text for k in ['budget', 'engineer', 'year', 'started']):
                clean_expr = re.search(r'([\(\d][\d\s\+\-\*\/\%\(\)\.]*[\)\d])', user_text)
                if clean_expr:
                    expr = clean_expr.group(1).strip()
                    return MockResponse(MockMessage(tool_calls=[MockCall("mock_calc_1", "calculate", {"expression": expr})]))

            # Direct file read request
            for target in ["operations.txt", "company.md", "budget.md", "secret.txt"]:
                if target in user_text:
                    path = f"../{target}" if ("../" in user_text or "outside" in user_text) else target
                    return MockResponse(MockMessage(tool_calls=[MockCall("mock_read_1", "read_file", {"path": path})]))

            # Multi-part question: project owner and support email
            if "owner" in user_text and "email" in user_text:
                return MockResponse(MockMessage(tool_calls=[
                    MockCall("mock_s1", "search_files", {"query": "project owner"}),
                    MockCall("mock_s2", "search_files", {"query": "support email"})
                ]))

            # Budget calculation
            if "budget" in user_text and ("total" in user_text or "combined" in user_text or "sum" in user_text):
                return MockResponse(MockMessage(tool_calls=[MockCall("mock_read_b", "read_file", {"path": "budget.md"})]))

            # Keyword queries
            if "start" in user_text or "year" in user_text:
                return MockResponse(MockMessage(tool_calls=[MockCall("mock_s_year", "search_files", {"query": "project started"})]))
            if "owner" in user_text:
                return MockResponse(MockMessage(tool_calls=[MockCall("mock_s_owner", "search_files", {"query": "project owner"})]))
            if "email" in user_text:
                return MockResponse(MockMessage(tool_calls=[MockCall("mock_s_email", "search_files", {"query": "support email"})]))
            if "city" in user_text:
                return MockResponse(MockMessage(tool_calls=[MockCall("mock_s_city", "search_files", {"query": "service Chennai"})]))
            if "engineer" in user_text:
                return MockResponse(MockMessage(tool_calls=[MockCall("mock_s_eng", "search_files", {"query": "engineers"})]))
            if "hour" in user_text or "window" in user_text:
                return MockResponse(MockMessage(tool_calls=[MockCall("mock_s_win", "search_files", {"query": "support window"})]))
            if "moon base" in user_text:
                return MockResponse(MockMessage(tool_calls=[MockCall("mock_s_moon", "search_files", {"query": "moon base"})]))

            clean_query = " ".join([w for w in re.findall(r'\w+', user_text) if len(w) > 3][:3])
            if clean_query:
                return MockResponse(MockMessage(tool_calls=[MockCall("mock_s_gen", "search_files", {"query": clean_query})]))

            return MockResponse(MockMessage("Hello! I am a tool-using document agent. You can ask me questions about your documents or request arithmetic calculations."))

        # Tool response handling
        tool_messages = [m for m in messages if m.get('role') == 'tool']
        payloads = []
        for tm in tool_messages:
            try:
                payloads.append(json.loads(tm.get('content', '{}')))
            except Exception:
                pass

        for p in payloads:
            if not p.get('ok'):
                err = p.get('error', '')
                if 'PermissionError' in err or 'traversal' in err.lower():
                    return MockResponse(MockMessage("The requested read was rejected because path traversal was blocked."))
                if 'TimeoutError' in err:
                    return MockResponse(MockMessage(f"The operation could not complete: {err}"))
                return MockResponse(MockMessage(f"Encountered tool error: {err}"))

        # Budget follow-up: read_file('budget.md') -> calculate
        for p in payloads:
            data = p.get('data')
            if isinstance(data, dict) and data.get('path') == 'budget.md' and self.round == 2:
                return MockResponse(MockMessage(tool_calls=[MockCall("mock_calc_budget", "calculate", {"expression": "15000 + 5000"})]))

        # Synthesize answers
        extracted = []
        calc_result = None
        for p in payloads:
            data = p.get('data')
            if not isinstance(data, dict):
                continue
            if 'result' in data:
                calc_result = data['result']
            if 'matches' in data:
                if data.get('match_count', 0) == 0:
                    return MockResponse(MockMessage("not found"))
                for m in data['matches']:
                    for hit in m.get('matches', []):
                        extracted.append(hit['line'])
            if 'content' in data:
                extracted.append(data['content'])

        if calc_result is not None:
            return MockResponse(MockMessage(f"The total is {calc_result} rupees." if "rupees" in str(messages[1].get('content', '')) else f"The calculated result is {calc_result}."))

        if extracted:
            combined = "\n".join(extracted)
            ans = []
            if "Mira" in combined:
                ans.append("The project owner is Mira.")
            if "help@acme.example" in combined:
                ans.append("The support email is help@acme.example.")
            if "2021" in combined:
                ans.append("The project started in 2021.")
            if "Chennai" in combined:
                ans.append("The service runs in Chennai.")
            if "12 engineers" in combined:
                ans.append("The team has 12 engineers.")
            if "9 AM to 5 PM" in combined:
                ans.append("The support window is 9 AM to 5 PM.")
            if ans:
                return MockResponse(MockMessage(" ".join(ans)))
            return MockResponse(MockMessage(f"Based on the documents:\n{combined.strip()}"))

        return MockResponse(MockMessage("not found"))


def main() -> None:
    parser = argparse.ArgumentParser(description='Ask questions about a local folder using a bounded tool-using agent.')
    parser.add_argument('question', nargs='?', help='Question to ask')
    parser.add_argument('--folder', default='docs', help='Folder containing .txt/.md files')
    parser.add_argument('--trace', default='trace.jsonl', help='JSONL trace output')
    parser.add_argument('--model', default=None, help='OpenAI model name (or set OPENAI_MODEL)')
    parser.add_argument('--mock', action='store_true', help='Run in offline mock mode without an API key')
    args = parser.parse_args()

    question = args.question or input('Question: ')
    try:
        load_dotenv()
        if args.mock or (not os.getenv('OPENAI_API_KEY') and not os.getenv('GEMINI_API_KEY')):
            if not args.mock:
                print("Notice: No OPENAI_API_KEY or GEMINI_API_KEY detected. Running in deterministic --mock mode.\n"
                      "To use live LLM calls, add your key to .env or environment variables.\n",
                      file=sys.stderr)
            model = MockInteractiveModel()
        else:
            model = OpenAIChatModel(args.model)

        agent = Agent(model, args.folder, args.trace)
        print(agent.answer(question))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()

