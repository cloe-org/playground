#!/usr/bin/env python3
"""Generate a Mermaid diagram of the repository directory structure.

This script walks the repository tree (excluding hidden entries,
__pycache__, and common non-source artifacts) and produces a Mermaid
graph that is injected into the README between special markers.

Usage:
    python scripts/generate_structure.py          # preview to stdout
    python scripts/generate_structure.py --update  # update README.md in-place
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Directories to always skip
SKIP_DIRS = {
    ".git",
    ".github",
    "__pycache__",
    ".ipynb_checkpoints",
    ".vscode",
    "node_modules",
}

# Files to skip everywhere
SKIP_FILES = {".DS_Store", ".gitkeep"}

# File extensions to skip
SKIP_EXTENSIONS = {".pyc", ".fits", ".npy", ".npz", ".hdf5"}

# Markers inside README.md
START_MARKER = "<!-- REPO-STRUCTURE-START -->"
END_MARKER = "<!-- REPO-STRUCTURE-END -->"


def _should_skip(entry: Path, is_root: bool = False) -> bool:
    """Return True if *entry* should be excluded from the diagram."""
    name = entry.name
    if entry.is_dir() and name in SKIP_DIRS:
        return True
    if name in SKIP_FILES:
        return True
    if any(name.endswith(ext) for ext in SKIP_EXTENSIONS):
        return True
    # Skip hidden files / dotfiles
    if name.startswith("."):
        return True
    # At the repo root, skip non-directory files (LICENSE, README, etc.)
    # so the diagram focuses on the directory structure and its contents.
    if is_root and entry.is_file():
        return True
    return False


def _sanitize_id(name: str) -> str:
    """Turn a file/directory name into a safe Mermaid node identifier."""
    return name.replace(".", "_").replace("-", "_")


def _collect(
    root: Path,
    parent_id: str,
    is_root: bool = False,
) -> tuple[list[str], list[str]]:
    """Recursively collect Mermaid node declarations and edges."""
    declarations: list[str] = []
    edges: list[str] = []

    entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    for entry in entries:
        if _should_skip(entry, is_root=is_root):
            continue

        child_id = parent_id + "__" + _sanitize_id(entry.name)
        label = entry.name + ("/" if entry.is_dir() else "")
        declarations.append(f"    {child_id}[{label}]")
        edges.append(f"    {parent_id} --> {child_id}")

        if entry.is_dir():
            sub_decl, sub_edges = _collect(entry, child_id)
            declarations.extend(sub_decl)
            edges.extend(sub_edges)

    return declarations, edges


def generate_mermaid(repo_root: Path) -> str:
    """Return a Mermaid graph string for the repository structure."""
    declarations, edges = _collect(repo_root, parent_id="root", is_root=True)

    lines = ["```mermaid", "graph LR"]
    lines.append(f"    root[🗂 {repo_root.name}]")
    lines.extend(declarations)
    lines.append("")
    lines.extend(edges)
    lines.append("```")
    return "\n".join(lines)


def update_readme(repo_root: Path, mermaid_block: str) -> bool:
    """Replace the structure section in README.md. Return True if changed."""
    readme_path = repo_root / "README.md"
    content = readme_path.read_text()

    pattern = re.compile(
        rf"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}",
        re.DOTALL,
    )
    replacement = f"{START_MARKER}\n{mermaid_block}\n{END_MARKER}"
    new_content, count = pattern.subn(replacement, content)
    if count == 0:
        raise RuntimeError(
            f"Could not find {START_MARKER} / {END_MARKER} markers in README.md"
        )

    if new_content == content:
        return False

    readme_path.write_text(new_content)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Mermaid diagram of the repo structure."
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update README.md in-place between the marker comments.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    mermaid = generate_mermaid(repo_root)

    if args.update:
        changed = update_readme(repo_root, mermaid)
        if changed:
            print("README.md updated.")
        else:
            print("README.md is already up to date.")
    else:
        print(mermaid)


if __name__ == "__main__":
    main()
