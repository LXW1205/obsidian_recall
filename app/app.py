"""
app.py — Streamlit Chat UI

Browser-based chat interface for querying Obsidian notes with natural language.
Supports two modes:
- Recall Mode: Read-only RAG querying with source attribution
- Agent Mode: Autonomous note management with confirmation gates

Three-tier memory:
- Tier 1: Hot memory (session context, preferences)
- Tier 2: ChromaDB (full vault, on-demand retrieval)
- Tier 3: Daily notes (auto-generated session logs)
"""

import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from ingest import VAULT_PATH, full_reindex, get_stats
from query import ask
from agent import execute_plan, plan_actions
from watcher import start_watcher

load_dotenv()

# ─── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Obsidian Recall",
    page_icon="🧠",
    layout="wide",
)

# ─── Start File Watcher ────────────────────────────────────────────────────────

if "watcher_started" not in st.session_state:
    start_watcher()
    st.session_state.watcher_started = True

# ─── Session State Init ────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_indexed" not in st.session_state:
    st.session_state.last_indexed = None

if "mode" not in st.session_state:
    st.session_state.mode = "recall"

if "agent_plan" not in st.session_state:
    st.session_state.agent_plan = None

if "agent_backup" not in st.session_state:
    st.session_state.agent_backup = None

if "agent_results" not in st.session_state:
    st.session_state.agent_results = None

if "recently_accessed" not in st.session_state:
    st.session_state.recently_accessed = []

# ─── Tier 1: Hot Memory Helpers ────────────────────────────────────────────────

def load_preferences() -> str:
    """Load user preferences from System/Assistant/preferences.md."""
    prefs_path = os.path.join(VAULT_PATH, "System", "Assistant", "preferences.md")
    if os.path.exists(prefs_path):
        try:
            with open(prefs_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    return ""


def get_hot_memory() -> dict:
    """Build hot memory block for context injection."""
    return {
        "conversation": st.session_state.messages[-10:],
        "active_notes": st.session_state.recently_accessed[-5:],
        "preferences": load_preferences(),
    }


def track_accessed_note(filepath: str):
    """Track a note as recently accessed (for hot memory)."""
    if filepath not in st.session_state.recently_accessed:
        st.session_state.recently_accessed.append(filepath)
        # Keep only last 10
        if len(st.session_state.recently_accessed) > 10:
            st.session_state.recently_accessed = st.session_state.recently_accessed[-10:]


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Obsidian Recall")

    # Mode switcher
    mode = st.radio(
        "Mode",
        ["Recall Mode", "Agent Mode", "Evaluate"],
        index=0 if st.session_state.mode == "recall" else (1 if st.session_state.mode == "agent" else 2),
        key="mode_selector",
    )
    st.session_state.mode = "recall" if mode == "Recall Mode" else ("agent" if mode == "Agent Mode" else "evaluate")

    st.divider()

    # Stats
    stats = get_stats()
    st.metric("Notes Indexed", stats["total_files_indexed"])
    st.metric("Chunks in DB", stats["total_chunks"])

    if st.session_state.last_indexed:
        st.caption(f"Last indexed: {st.session_state.last_indexed}")
    else:
        st.caption("Not yet indexed")

    st.divider()

    # Metadata filters
    st.subheader("Filters")
    
    # Folder filter
    folder_filter = st.text_input(
        "Folder path (optional)",
        value="",
        placeholder="e.g., Projects/",
        help="Only search notes in this folder path",
    )
    
    # Tag filter
    tags_input = st.text_input(
        "Tags (optional, comma-separated)",
        value="",
        placeholder="e.g., active, important",
        help="Only search notes containing these tags",
    )
    
    # Parse tags
    tags_filter = [t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else None

    st.divider()

    # Re-index button
    if st.button("Re-index All Notes", type="primary", use_container_width=True):
        with st.spinner("Re-indexing all notes..."):
            result = full_reindex(force=True)
            st.session_state.last_indexed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if result.get("total_files") is not None:
                st.success(
                    f"Indexed {result['indexed']} files, "
                    f"{result['total_chunks']} chunks, "
                    f"{result['skipped']} unchanged, "
                    f"{result['errors']} errors"
                )
            else:
                st.error(f"Re-index failed: {result.get('error', 'Unknown error')}")

    # Clear chat button
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_plan = None
        st.session_state.agent_results = None
        st.rerun()

    st.divider()
    st.caption("Self-hosted RAG for your Obsidian vault")

# ─── Main Chat Area ────────────────────────────────────────────────────────────

if st.session_state.mode == "recall":
    st.title("Recall Mode")
    st.caption("Ask questions about your notes — answers are grounded in your vault with source citations.")
else:
    st.title("Agent Mode")
    st.caption("Tell me what to do with your notes — I'll plan actions for your approval.")

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Show answer label badge for assistant messages
        if msg["role"] == "assistant" and "answer_label" in msg:
            label = msg["answer_label"]
            if label == "grounded_answer":
                st.success("✓ Grounded Answer")
            elif label == "insufficient_context":
                st.warning("⚠ Insufficient Context")
            elif label == "conflicting_context":
                st.error("✗ Conflicting Context")
            
            # Show grounding score if available
            if "grounding" in msg:
                grounding = msg["grounding"]
                score = grounding.get("grounding_score", 0)
                if score > 0:
                    st.caption(f"Grounding score: {score:.0%}")
        
        # Show sources
        if msg["role"] == "assistant" and "sources" in msg:
            if msg["sources"]:
                st.markdown("**Sources:**")
                for source in msg["sources"]:
                    st.markdown(f"- `{source}`")

# Agent Mode: Show confirmation panel if plan is pending
if st.session_state.mode == "agent" and st.session_state.agent_plan is not None:
    st.divider()
    with st.container(border=True):
        st.subheader("Agent Plan")

        backup = st.session_state.agent_backup
        if backup and backup.get("success"):
            st.success(f"Backup created: {backup.get('backup_path', 'N/A')}")

        plan = st.session_state.agent_plan

        # Count write actions
        write_actions = [a for a in plan if a.get("tool") not in ("read_note", "list_notes", "search_notes", "find_duplicates", "find_broken_links", "error")]
        affected_files = set()
        for a in write_actions:
            fp = a.get("args", {}).get("filepath") or a.get("args", {}).get("old_path") or a.get("args", {}).get("target_path")
            if fp:
                affected_files.add(fp)

        st.write(f"The agent will perform **{len(plan)}** actions:")

        for i, action in enumerate(plan, 1):
            tool = action.get("tool", "unknown")
            desc = action.get("description", "")
            st.write(f"{i}. **{tool}** — {desc}")

        if affected_files:
            st.warning(f"This will modify **{len(affected_files)}** files.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm & Execute", type="primary", use_container_width=True):
                with st.spinner("Executing plan..."):
                    results = execute_plan(plan)
                    st.session_state.agent_results = results
                    st.session_state.agent_plan = None
                    st.session_state.agent_backup = None
                    st.session_state.last_indexed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Build results summary
                    summary_lines = ["## Execution Results\n"]
                    succeeded = sum(1 for r in results if r.get("success"))
                    failed = sum(1 for r in results if not r.get("success"))

                    for r in results:
                        status = "✅" if r.get("success") else "❌"
                        action_desc = r.get("description", r.get("action", "Unknown"))
                        error = r.get("error", "")
                        summary_lines.append(f"{status} {action_desc}")
                        if error:
                            summary_lines.append(f"   Error: {error}")

                    summary_text = "\n".join(summary_lines)
                    summary_text += f"\n\n**Summary:** {succeeded} succeeded, {failed} failed."

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": summary_text,
                        "sources": [],
                    })
                    st.rerun()

        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.agent_plan = None
                st.session_state.agent_backup = None
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Plan cancelled. No actions were executed.",
                    "sources": [],
                })
                st.rerun()

# Chat input
placeholder = (
    "Ask a question about your notes..."
    if st.session_state.mode == "recall"
    else "Tell me what to do with your notes..."
)

if prompt := st.chat_input(placeholder):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.mode == "recall":
                # Build conversation history for multi-turn
                conversation_history = []
                for m in st.session_state.messages[:-1]:
                    conversation_history.append(m)

                result = ask(
                    prompt,
                    conversation_history=conversation_history,
                    hot_memory=get_hot_memory(),
                    folder=folder_filter if folder_filter else None,
                    tags=tags_filter,
                )

                response_text = result["answer"]
                sources = result["sources"]
                answer_label = result.get("answer_label", "grounded_answer")
                grounding = result.get("grounding", {})

                st.markdown(response_text)
                
                # Show answer label badge
                if answer_label == "grounded_answer":
                    st.success("✓ Grounded Answer")
                elif answer_label == "insufficient_context":
                    st.warning("⚠ Insufficient Context")
                elif answer_label == "conflicting_context":
                    st.error("✗ Conflicting Context")
                
                # Show grounding score
                grounding_score = grounding.get("grounding_score", 0)
                if grounding_score > 0:
                    st.caption(f"Grounding score: {grounding_score:.0%}")
                
                if sources:
                    st.markdown("**Sources:**")
                    for source in sources:
                        st.markdown(f"- `{source}`")
                        track_accessed_note(source)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources,
                    "answer_label": answer_label,
                    "grounding": grounding,
                })

            else:
                # Agent Mode: Plan actions
                plan_result = plan_actions(prompt)

                if not plan_result["success"]:
                    error_msg = f"Planning failed: {plan_result.get('error', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                    })
                else:
                    plan = plan_result["plan"]
                    backup = plan_result["backup"]

                    # Check for error plan
                    if plan and plan[0].get("tool") == "error":
                        error_msg = plan[0].get("description", "Could not plan actions.")
                        st.markdown(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "sources": [],
                        })
                    else:
                        # Store plan for confirmation
                        st.session_state.agent_plan = plan
                        st.session_state.agent_backup = backup

                        # Show plan preview in chat
                        preview_lines = ["I've created a plan to complete your request:\n"]
                        preview_lines.append(f"**Backup:** `{backup.get('backup_path', 'N/A')}`\n")
                        preview_lines.append(f"**{len(plan)} actions planned:**\n")
                        for i, action in enumerate(plan, 1):
                            preview_lines.append(f"{i}. {action.get('description', action.get('tool', ''))}")
                        preview_lines.append("\nReview the plan above and click **Confirm & Execute** or **Cancel**.")

                        preview_text = "\n".join(preview_lines)
                        st.markdown(preview_text)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": preview_text,
                            "sources": [],
                        })

# ─── Evaluate Mode ─────────────────────────────────────────────────────────────

if st.session_state.mode == "evaluate":
    st.title("Evaluate Mode")
    st.caption("Test retrieval quality with sample queries and measure Recall@K, Precision@K, and Faithfulness.")
    
    from evaluate import run_evaluation
    
    # Evaluation input
    st.subheader("Test Queries")
    default_queries = """[
  {"question": "What is RAG?", "expected_sources": []},
  {"question": "How does chunking work?", "expected_sources": []}
]"""
    queries_json = st.text_area(
        "Enter test queries as JSON array",
        value=default_queries,
        height=200,
    )
    
    if st.button("Run Evaluation", type="primary", use_container_width=True):
        import json
        try:
            queries = json.loads(queries_json)
            with st.spinner("Running evaluation..."):
                results = run_evaluation(queries)
            
            st.success("Evaluation complete!")
            
            # Display summary
            st.subheader("Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Queries", results["summary"]["total_queries"])
            col2.metric("Recall@5", f"{results['summary']['recall_at_k']:.0%}")
            col3.metric("Precision@5", f"{results['summary']['precision_at_k']:.0%}")
            col4.metric("Faithfulness", f"{results['summary']['faithfulness']:.0%}")
            
            # Display per-query results
            st.subheader("Per-Query Results")
            for r in results["results"]:
                with st.expander(f"Q: {r['question']}"):
                    st.write(f"**Status:** {r['status']}")
                    st.write(f"**Retrieved sources:** {r['retrieved_sources']}")
                    if r.get("expected_sources"):
                        st.write(f"**Expected sources:** {r['expected_sources']}")
                    st.write(f"**Answer label:** {r['answer_label']}")
                    if r.get("answer"):
                        st.write(f"**Answer:** {r['answer'][:200]}...")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

