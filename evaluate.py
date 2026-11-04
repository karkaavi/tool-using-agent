"""Comprehensive 10-scenario evaluation suite for the Tool-Using Agent.

Covers all 10 required test criteria:
1. Simple file search
2. File reading
3. Multiple tool calls (Search then Read)
4. Multiple tool calls (Read then Calculate)
5. Calculator question (AST arithmetic)
6. Missing document fact ('not found')
7. Path-traversal rejection test
8. Tool failure and error recovery
9. Maximum-step termination (8-step limit)
10. Tool timeout handling (5-second limit)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

from agent import Agent, MAX_STEPS, OpenAIChatModel, ScriptedModel, load_dotenv


CASES: list[dict[str, Any]] = [
    {
        'id': 1,
        'name': '1. Simple file search',
        'question': 'What is the support email?',
        'type': 'scripted',
        'script': [
            [('search_files', {'query': 'support email'})],
            'The support email is help@acme.example.'
        ],
        'expected_answer': 'The support email is help@acme.example.',
        'expected_tools': ['search_files'],
        'evidence': 'help@acme.example',
    },
    {
        'id': 2,
        'name': '2. File reading',
        'question': 'What are the operations details in operations.txt?',
        'type': 'scripted',
        'script': [
            [('read_file', {'path': 'operations.txt'})],
            'The service runs in Chennai with 12 engineers.'
        ],
        'expected_answer': 'The service runs in Chennai with 12 engineers.',
        'expected_tools': ['read_file'],
        'evidence': ['Chennai', '12 engineers'],
    },
    {
        'id': 3,
        'name': '3. Multiple tool calls (Search then Read)',
        'question': 'When did the project start and who owns it?',
        'type': 'scripted',
        'script': [
            [('search_files', {'query': 'project started'})],
            [('read_file', {'path': 'company.md'})],
            'The project started in 2021 and the owner is Mira.'
        ],
        'expected_answer': 'The project started in 2021 and the owner is Mira.',
        'expected_tools': ['search_files', 'read_file'],
        'evidence': ['2021', 'Mira'],
    },
    {
        'id': 4,
        'name': '4. Multiple tool calls (Read then Calculate)',
        'question': 'What is the total combined budget for equipment and training?',
        'type': 'scripted',
        'script': [
            [('read_file', {'path': 'budget.md'})],
            [('calculate', {'expression': '15000 + 5000'})],
            'The total combined budget is 20000 rupees.'
        ],
        'expected_answer': 'The total combined budget is 20000 rupees.',
        'expected_tools': ['read_file', 'calculate'],
        'calculation': 20000,
    },
    {
        'id': 5,
        'name': '5. Calculator arithmetic',
        'question': 'What is (15000 * 2) / 3?',
        'type': 'scripted',
        'script': [
            [('calculate', {'expression': '(15000 * 2) / 3'})],
            'The result is 10000.0.'
        ],
        'expected_answer': 'The result is 10000.0.',
        'expected_tools': ['calculate'],
        'calculation': 10000.0,
    },
    {
        'id': 6,
        'name': '6. Missing document fact (not found)',
        'question': 'Where is the company moon base?',
        'type': 'scripted',
        'script': [
            [('search_files', {'query': 'moon base'})],
            'not found'
        ],
        'expected_answer': 'not found',
        'expected_tools': ['search_files'],
        'not_found': True,
    },
    {
        'id': 7,
        'name': '7. Path-traversal rejection',
        'question': 'Read the secret file at ../secret.txt.',
        'type': 'scripted',
        'script': [
            [('read_file', {'path': '../secret.txt'})],
            'The requested read was rejected because path traversal was blocked.'
        ],
        'expected_answer': 'The requested read was rejected because path traversal was blocked.',
        'expected_tools': ['read_file'],
        'error_match': 'Path traversal rejected',
    },
    {
        'id': 8,
        'name': '8. Tool failure and error recovery',
        'question': 'Find company information.',
        'type': 'scripted',
        'script': [
            [('read_file', {'path': 'nonexistent.md'})],  # Fails with FileNotFoundError
            [('read_file', {'path': 'company.md'})],      # Recovers with correct path
            'The project started in 2021 and the owner is Mira.'
        ],
        'expected_answer': 'The project started in 2021 and the owner is Mira.',
        'expected_tools': ['read_file', 'read_file'],
        'evidence': ['2021', 'Mira'],
        'first_error': 'FileNotFoundError',
    },
    {
        'id': 9,
        'name': '9. Maximum-step termination (8-step limit)',
        'question': 'Keep calculating indefinitely.',
        'type': 'step_limit',
        'script': [
            [('calculate', {'expression': f'{i} + 1'})] for i in range(MAX_STEPS + 1)
        ],
        'expected_answer_contains': '8-step tool limit',
        'expected_steps': 8,
    },
    {
        'id': 10,
        'name': '10. Tool timeout handling (5s timeout)',
        'question': 'Execute a long running search.',
        'type': 'timeout',
        'script': [
            [('search_files', {'query': 'alpha'})],
            'The search timed out and could not be completed within the time limit.'
        ],
        'expected_answer': 'The search timed out and could not be completed within the time limit.',
        'expected_tools': ['search_files'],
        'error_match': 'TimeoutError',
    },
]


def _verify_trace_records(rows: list[dict[str, Any]]) -> bool:
    """Ensure every trace record has all mandatory fields including valid timestamps."""
    required_keys = {'step', 'tool_name', 'arguments', 'result', 'error', 'latency_ms', 'timestamp'}
    for row in rows:
        if not required_keys.issubset(row.keys()):
            return False
        # Timestamp must be non-empty ISO 8601 string
        ts = row.get('timestamp')
        if not isinstance(ts, str) or not ts:
            return False
    return True


def run_offline_case(case: dict[str, Any], docs_folder: Path, temp_dir: Path) -> tuple[bool, str]:
    trace_path = temp_dir / f"trace_case_{case['id']}.jsonl"
    model = ScriptedModel(list(case['script']))
    agent = Agent(model, docs_folder, trace_path)

    if case['type'] == 'timeout':
        agent.runner.timeout = 0.05
        def slow_search(*_args, **_kwargs):
            time.sleep(0.1)
            return {'matches': [], 'match_count': 0}
        with patch('agent.search_files', side_effect=slow_search):
            answer = agent.answer(case['question'])
    elif case['type'] == 'step_limit':
        answer = agent.answer(case['question'])
    else:
        answer = agent.answer(case['question'])

    if not trace_path.exists():
        return False, "Trace file was not created"

    rows = [json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines()]
    if not _verify_trace_records(rows):
        return False, "Trace records missing required fields or timestamp"

    # Specific assertions per case type
    if case['type'] == 'step_limit':
        tool_rows = [r for r in rows if r.get('event') == 'tool']
        forced_rows = [r for r in rows if r.get('event') == 'forced_stop']
        if len(tool_rows) != 8:
            return False, f"Expected 8 tool calls, got {len(tool_rows)}"
        if not forced_rows or forced_rows[0].get('error') != 'step_limit_reached':
            return False, "Missing forced_stop event in trace"
        if case['expected_answer_contains'].lower() not in answer.lower():
            return False, f"Answer missing step limit notice: {answer}"
        return True, answer

    if case.get('error_match'):
        tool_rows = [r for r in rows if r.get('event') == 'tool']
        if not any(case['error_match'] in (r.get('error') or '') for r in tool_rows):
            return False, f"Expected error '{case['error_match']}' not found in trace"

    if case.get('first_error'):
        tool_rows = [r for r in rows if r.get('event') == 'tool']
        if not tool_rows or case['first_error'] not in (tool_rows[0].get('error') or ''):
            return False, f"First tool call did not fail with {case['first_error']}"

    if case.get('not_found'):
        tool_rows = [r for r in rows if r.get('event') == 'tool']
        res = tool_rows[0].get('result') or {}
        if res.get('match_count') != 0:
            return False, "Expected 0 matches for missing fact"

    if case.get('calculation'):
        tool_rows = [r for r in rows if r.get('event') == 'tool' and r.get('tool_name') == 'calculate']
        if not tool_rows or tool_rows[-1].get('result', {}).get('result') != case['calculation']:
            return False, f"Calculation result mismatch: {tool_rows}"

    if case.get('expected_tools'):
        actual_tools = [r.get('tool_name') for r in rows if r.get('event') == 'tool']
        if actual_tools != case['expected_tools']:
            return False, f"Expected tools {case['expected_tools']}, got {actual_tools}"

    if answer.strip().casefold() != case['expected_answer'].strip().casefold():
        return False, f"Answer mismatch: expected '{case['expected_answer']}', got '{answer}'"

    return True, answer


def run_live_evaluation(docs_folder: Path, model_name: str | None = None) -> tuple[int, int]:
    """Optional live evaluation against configured LLM provider."""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: Neither OPENAI_API_KEY nor GEMINI_API_KEY is set in environment or .env.", file=sys.stderr)
        print("Please configure your API key before running --live evaluation.", file=sys.stderr)
        return 0, 0

    print(f"Running live evaluation with model: {model_name or os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')}...")
    passed = 0
    test_questions = [
        ("What is the project owner and support email?", ["Mira", "help@acme.example"]),
        ("What is 15000 + 5000?", ["20000"]),
        ("Where is the moon base?", ["not found", "no information", "cannot find"]),
        ("What are the company operations hours and city?", ["Chennai", "9 AM to 5 PM"]),
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, (q, expected_tokens) in enumerate(test_questions, 1):
            trace_path = Path(temp_dir) / f"live_{i}.jsonl"
            try:
                model = OpenAIChatModel(model_name)
                agent = Agent(model, docs_folder, trace_path)
                ans = agent.answer(q)
                ok = any(tok.lower() in ans.lower() for tok in expected_tokens)
                if ok:
                    passed += 1
                print(f"{'PASS' if ok else 'FAIL'} | Live Q{i}: {q} -> {ans}")
            except Exception as e:
                print(f"FAIL | Live Q{i}: {q} -> Error: {e}")

    print(f"\nLive Pass Rate: {passed}/{len(test_questions)} = {passed / len(test_questions) * 100:.1f}%")
    return passed, len(test_questions)


def run_offline_evaluation() -> tuple[int, int]:
    base = Path(__file__).parent
    docs_folder = base / 'docs'
    passed = 0
    results: list[tuple[str, bool, str]] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for case in CASES:
            ok, detail = run_offline_case(case, docs_folder, temp_path)
            if ok:
                passed += 1
            results.append((case['name'], ok, detail))

    total = len(CASES)
    pct = (passed / total) * 100.0
    print("\n" + "=" * 65)
    print(f"  TOOL-USING AGENT EVALUATION REPORT: {passed}/{total} PASSED ({pct:.1f}%)")
    print("=" * 65)
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        print(f"       Result: {detail}\n")
    print("-" * 65)
    print(f"Final Score: {passed}/{total} ({pct:.1f}%)\n")
    return passed, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Tool-Using Document Agent")
    parser.add_argument('--live', action='store_true', help="Run live evaluation against configured LLM API")
    parser.add_argument('--model', default=None, help="LLM model name for live evaluation")
    args = parser.parse_args()

    if args.live:
        base = Path(__file__).parent
        run_live_evaluation(base / 'docs', args.model)
    else:
        passed, total = run_offline_evaluation()
        if passed != total:
            sys.exit(1)


if __name__ == '__main__':
    main()