import asyncio
import os
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# --------------------------------------------------
# Config
# --------------------------------------------------
SOURCE_FOLDER = "./source_folder"
Path(SOURCE_FOLDER).mkdir(exist_ok=True)

llm = ChatOpenAI(model="gpt-4o-mini")

# --------------------------------------------------
# State Schema
# --------------------------------------------------
class WorkflowState(TypedDict, total=False):
    files: List[str]
    checked_at: str
    processed_files: List[str]
    processed_at: str
    iteration: int
    max_iterations: int

# --------------------------------------------------
# Async Nodes
# --------------------------------------------------
async def check_new_files(state: dict) -> dict:
    iteration = state.get("iteration", 0) + 1
    print(f"\n[check_new_files] scanning folder... (iteration {iteration})")
    files = [
        str(Path(SOURCE_FOLDER) / f)
        for f in os.listdir(SOURCE_FOLDER)
        if (Path(SOURCE_FOLDER) / f).is_file()
    ]
    return {
        "files": files,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration
    }

async def watchdog_wait(state: dict) -> dict:
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 0)

    # Check if we should exit
    if max_iterations > 0 and iteration >= max_iterations:
        print(f"[watchdog_wait] reached max iterations ({max_iterations}), exiting...")
        return {"should_exit": True, **state}

    print("[watchdog_wait] sleeping...")
    await asyncio.sleep(2)
    return state

async def process_new_files(state: dict) -> dict:
    print("[process_new_files] processing files")
    processed = []
    for f in state.get("files", []):
        print(f"  → {f}")
        processed.append(f)
        os.remove(f)
    return {
        "processed_files": processed,
        "processed_at": datetime.now(timezone.utc).isoformat()
    }

async def log_results(state: dict) -> dict:
    print("\n[log_results]")
    print("Processed:", state.get("processed_files"))
    return state

# --------------------------------------------------
# LLM Router (Conditional Edge)
# --------------------------------------------------
def agent_router(state: dict) -> str:
    """
    LLM decides which edge to take.
    MUST return one of: 'process', 'wait', or 'exit'
    """
    # Check exit condition first
    if state.get("should_exit"):
        print("[router] should_exit flag detected, routing to exit")
        return "exit"

    prompt = f"""
You are a workflow router.

Files detected: {state.get("files")}

Rules:
- If files exist → return "process"
- If no files → return "wait"

Return ONLY ONE WORD.
"""
    decision = llm.invoke(prompt).content.strip().lower()
    print(f"[router] decision = {decision}")
    return decision

# --------------------------------------------------
# Build LangGraph
# --------------------------------------------------
graph = StateGraph(WorkflowState)

graph.add_node("check_new_files", check_new_files)
graph.add_node("watchdog_wait", watchdog_wait)
graph.add_node("process_new_files", process_new_files)
graph.add_node("log_results", log_results)

graph.add_conditional_edges(
    "check_new_files",
    agent_router,
    {
        "process": "process_new_files",
        "wait": "watchdog_wait",
        "exit": END
    }
)

graph.add_edge("watchdog_wait", "check_new_files")
graph.add_edge("process_new_files", "log_results")
graph.add_edge("log_results", END)

graph.set_entry_point("check_new_files")

app = graph.compile()

# --------------------------------------------------
# Visualize Graph
# --------------------------------------------------
try:
    print("\n--- Mermaid Diagram ---")
    print(app.get_graph().draw_mermaid())
except Exception:
    pass

# --------------------------------------------------
# Run
# --------------------------------------------------
async def main():
    print("=" * 60)
    print("File Watchdog Started")
    print(f"Monitoring folder: {SOURCE_FOLDER}")
    print("=" * 60)
    print("\nOptions:")
    print("  - Press Ctrl+C to stop manually")
    print("  - Or set max_iterations in initial state")
    print(f"  - Add files to '{SOURCE_FOLDER}' to trigger processing\n")

    # Set max_iterations to limit loops (0 = infinite)
    # Change this value to limit iterations, or set to 0 for infinite loop
    MAX_ITERATIONS = 10  # Will stop after 5 checks with no files

    try:
        await app.ainvoke({"max_iterations": MAX_ITERATIONS})
        print("\n" + "=" * 60)
        print("Watchdog completed successfully")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Watchdog stopped by user (Ctrl+C)")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
