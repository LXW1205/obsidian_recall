"""
agent.py — Agent Mode Planning + Execution Loop

Receives natural language instructions, plans a sequence of tool calls via Gemini,
presents the plan for user confirmation, and executes after approval.
"""

import json
import os
import re
import shutil
from datetime import datetime
from typing import Optional

from google import genai
from dotenv import load_dotenv

from ingest import VAULT_PATH, reindex_file, get_google_client
from opencode_client import get_llm_client
from tools import list_notes as tools_list_notes
import tools

load_dotenv()

BACKUP_DIR = os.path.join(os.path.dirname(VAULT_PATH), "backups")
if not os.path.isabs(BACKUP_DIR):
    BACKUP_DIR = os.path.join("/app", "backups")

AGENT_SYSTEM_PROMPT = """\
You are an autonomous note management agent for an Obsidian vault.
The user will give you a natural language instruction.
You must plan a sequence of tool calls to complete the task.

Available tools:
- read_note(filepath): Read content of a note
- list_notes(folder=None): List all notes, optionally filtered by folder
- create_note(filepath, content): Create a new note with given content
- edit_note(filepath, new_content): Overwrite a note's content
- rename_note(old_path, new_name): Rename a note file (new_name should include .md)
- move_note(filepath, target_folder): Move note to a different folder
- add_tag(filepath, tag): Add a #tag to a note's frontmatter
- delete_note(filepath): Delete a note (high risk)
- merge_notes(source_paths, target_path): Merge multiple notes into one
- find_duplicates(threshold=0.92): Find semantically similar notes
- find_broken_links(): Scan for broken [[wikilinks]]
- search_notes(query): Semantic search via ChromaDB

Rules:
- Always use the minimum number of operations needed
- Never delete notes unless explicitly instructed to
- Always prefer moving over deleting
- When renaming, base the new name on the actual content of the note (read it first)
- When adding tags, use lowercase with hyphens (e.g. #machine-learning)
- For bulk operations, read notes first to understand content before acting
- Return your plan as a JSON array of tool call objects

Response format — return ONLY valid JSON, no markdown, no explanation:
[
  {
    "tool": "tool_name",
    "args": {"arg1": "value1", "arg2": "value2"},
    "description": "Human-readable description of what this does"
  }
]

If the instruction is unclear or cannot be completed with available tools, return:
[{"tool": "error", "args": {}, "description": "Explanation of why the task cannot be completed"}]"""

MODEL_NAME = "gemini-flash-lite-latest"


def backup_vault(vault_path: str = None, backup_dir: str = None) -> dict:
    """Create a full timestamped snapshot of the vault."""
    vault_path = vault_path or VAULT_PATH
    backup_dir = backup_dir or BACKUP_DIR

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"vault_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copytree(vault_path, backup_path)
        return {
            "success": True,
            "backup_path": backup_path,
            "timestamp": timestamp,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def create_daily_note() -> str:
    """Create or return today's daily note path."""
    daily_dir = os.path.join(VAULT_PATH, "Daily")
    os.makedirs(daily_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    daily_path = os.path.join(daily_dir, f"{today}.md")

    if not os.path.exists(daily_path):
        with open(daily_path, "w", encoding="utf-8") as f:
            f.write(f"# {today}\n\n")

    return daily_path


def log_agent_action(action: str, detail: str = ""):
    """Log an agent action to today's daily note."""
    daily_path = create_daily_note()
    timestamp = datetime.now().strftime("%H:%M")

    with open(daily_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if Agent Actions section exists
    if "## Agent Actions" not in content:
        content += "\n## Agent Actions\n"

    detail_str = f" — {detail}" if detail else ""
    content += f"- [{timestamp}] {action}{detail_str}\n"

    with open(daily_path, "w", encoding="utf-8") as f:
        f.write(content)


def parse_plan(response_text: str) -> list[dict]:
    """Parse the Gemini response into a plan (list of tool calls)."""
    # Try to extract JSON from the response
    # Handle markdown code blocks
    json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", response_text)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = response_text.strip()

    try:
        plan = json.loads(json_str)
        if isinstance(plan, list):
            return plan
    except json.JSONDecodeError:
        pass

    # Fallback: try to find JSON array in text
    array_match = re.search(r"\[[\s\S]*\]", response_text)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass

    return [
        {
            "tool": "error",
            "args": {},
            "description": f"Could not parse agent plan. Raw response: {response_text[:200]}",
        }
    ]


def plan_actions(user_instruction: str) -> dict:
    """
    Send user instruction to Gemini and get back a plan.

    Returns:
        {
            "success": bool,
            "plan": [...],  # list of tool call dicts
            "backup": {...},  # backup info
            "error": str or None
        }
    """
    # Step 1: Backup vault
    backup = backup_vault()
    if not backup["success"]:
        return {
            "success": False,
            "plan": [],
            "backup": backup,
            "error": f"Backup failed: {backup['error']}",
        }

    # Step 2: Get current vault state for context
    try:
        notes_result = tools.list_notes()
        note_count = len(notes_result.get("notes", []))
        note_list = "\n".join(notes_result.get("notes", [])[:50])  # First 50 for context
        if len(notes_result.get("notes", [])) > 50:
            note_list += f"\n... and {len(notes_result['notes']) - 50} more notes"
    except Exception:
        note_count = 0
        note_list = "(could not list notes)"

    # Step 3: Send to LLM for planning
    prompt_text = f"""Current vault state: {note_count} notes total.

Notes:
{note_list}

User instruction: {user_instruction}

Return your plan as a JSON array."""

    try:
        llm = get_llm_client()
        if llm is not None:
            # Opencode provider
            response_text = llm.generate_content(
                prompt=prompt_text,
                system_instruction=AGENT_SYSTEM_PROMPT,
            )
        else:
            # Gemini provider
            response = get_google_client().models.generate_content(
                model=MODEL_NAME,
                contents=prompt_text,
                config={"system_instruction": AGENT_SYSTEM_PROMPT},
            )
            response_text = response.text

        plan = parse_plan(response_text)

        return {
            "success": True,
            "plan": plan,
            "backup": backup,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "plan": [],
            "backup": backup,
            "error": f"LLM planning failed: {str(e)}",
        }


def execute_plan(plan: list[dict]) -> list[dict]:
    """
    Execute a plan (list of tool calls) sequentially.

    Returns list of result dicts for each action.
    """
    results = []

    tool_map = {
        "read_note": tools.read_note,
        "list_notes": tools.list_notes,
        "create_note": tools.create_note,
        "edit_note": tools.edit_note,
        "rename_note": tools.rename_note,
        "move_note": tools.move_note,
        "add_tag": tools.add_tag,
        "delete_note": tools.delete_note,
        "merge_notes": tools.merge_notes,
        "find_duplicates": tools.find_duplicates,
        "find_broken_links": tools.find_broken_links,
        "search_notes": tools.search_notes,
    }

    for action in plan:
        tool_name = action.get("tool", "")
        args = action.get("args", {})
        description = action.get("description", tool_name)

        if tool_name == "error":
            results.append({
                "success": False,
                "action": description,
                "error": description,
            })
            continue

        tool_fn = tool_map.get(tool_name)
        if not tool_fn:
            results.append({
                "success": False,
                "action": description,
                "error": f"Unknown tool: {tool_name}",
            })
            continue

        try:
            result = tool_fn(**args)
            result["description"] = description
            results.append(result)

            # Log to daily note
            if result.get("success"):
                log_agent_action(description, result.get("filepath", ""))

            # Re-index affected files after write operations
            write_tools = {"create_note", "edit_note", "rename_note", "move_note", "add_tag", "delete_note", "merge_notes"}
            if tool_name in write_tools and result.get("filepath"):
                try:
                    reindex_file(os.path.join(VAULT_PATH, result["filepath"]))
                except Exception:
                    pass  # Non-critical

        except Exception as e:
            results.append({
                "success": False,
                "action": description,
                "error": str(e),
            })

    return results
