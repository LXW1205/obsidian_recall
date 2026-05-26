"""Tests for agent module — plan parsing."""

import json
from agent import parse_plan


def test_parse_valid_json():
    plan = json.dumps([
        {"tool": "read_note", "args": {"filepath": "test.md"}, "description": "Read a note"},
    ])
    result = parse_plan(plan)
    assert len(result) == 1
    assert result[0]["tool"] == "read_note"
    assert result[0]["args"]["filepath"] == "test.md"


def test_parse_json_in_code_block():
    plan = '''```json
[
  {"tool": "create_note", "args": {"filepath": "new.md", "content": "# Hello"}, "description": "Create note"}
]
```'''
    result = parse_plan(plan)
    assert len(result) == 1
    assert result[0]["tool"] == "create_note"


def test_parse_json_in_code_block_no_lang():
    plan = '''```
[
  {"tool": "list_notes", "args": {}, "description": "List all notes"}
]
```'''
    result = parse_plan(plan)
    assert len(result) == 1
    assert result[0]["tool"] == "list_notes"


def test_parse_json_with_extra_text():
    plan = (
        'Here is my plan:\n\n'
        '[{"tool": "read_note", "args": {"filepath": "test.md"}, "description": "Test"}]\n\n'
        'End of plan.'
    )
    result = parse_plan(plan)
    assert len(result) == 1
    assert result[0]["tool"] == "read_note"


def test_parse_invalid_json():
    result = parse_plan("This is not JSON at all.")
    assert result[0]["tool"] == "error"
    assert "parse" in result[0]["description"].lower()


def test_parse_empty():
    result = parse_plan("")
    assert result[0]["tool"] == "error"


def test_parse_not_a_list():
    result = parse_plan('{"tool": "read_note", "args": {}}')
    assert result[0]["tool"] == "error"


def test_parse_multiple_actions():
    plan = json.dumps([
        {"tool": "search_notes", "args": {"query": "python"}, "description": "Search"},
        {"tool": "read_note", "args": {"filepath": "python.md"}, "description": "Read"},
        {"tool": "edit_note", "args": {"filepath": "python.md", "new_content": "# Updated"}, "description": "Update"},
    ])
    result = parse_plan(plan)
    assert len(result) == 3
    assert result[0]["tool"] == "search_notes"
    assert result[1]["tool"] == "read_note"
    assert result[2]["tool"] == "edit_note"
