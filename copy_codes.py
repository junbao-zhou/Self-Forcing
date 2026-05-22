#!/usr/bin/env python3
"""
Copy project source files while honoring .gitignore.

Features:
  * Parses .gitignore (uses pathspec if available, else fallback matcher)
  * Copies only "code" file extensions by default (toggle --all-files)
  * Destination may be local (filesystem) or remote ([user@]host:/path) via scp
  * Optional dry run
  * Optional automatic remote directory creation (via ssh) when using scp

Examples:
  Local copy (code files only):
    python copy_codes.py --dest /tmp/project_copy
  Include all non-ignored files:
    python copy_codes.py --dest /tmp/full_copy --all-files
  Remote copy (needs scp; destination path must exist unless --mkdir given):
    python copy_codes.py --dest user@server:/home/user/project_copy --mkdir
  Dry run:
    python copy_codes.py --dest /tmp/whatever --dry-run
"""

from __future__ import annotations
import argparse
import os
import shutil
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Callable

CODE_EXTENSIONS = {
    ".py",
    ".pyi",
    ".ipynb",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".psm1",
    ".pl",
    ".pm",
    ".lua",
    ".r",
    ".R",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".sql",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".json",
    ".md",
    ".txt",
    ".cu",
    ".gitignore",
    "LICENSE",
    "LICENSE.txt",
    "README",
    "README.md",
}


def load_gitignore(
    root: Path,
) -> List[str]:
    gi = root / ".gitignore"
    if not gi.is_file():
        return []
    lines: List[str] = []
    for line in gi.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip("\n")
        if not line or line.lstrip().startswith("#"):
            continue
        lines.append(line)
    return lines


class IgnoreMatcher:
    """Simplified .gitignore fallback (not fully spec compliant)."""

    def __init__(
        self,
        patterns: List[str],
    ):
        import fnmatch

        self.fnmatch = fnmatch
        self.rules = []
        for raw in patterns:
            neg = raw.startswith("!")
            pat = raw[1:] if neg else raw
            dir_only = pat.endswith("/")
            if dir_only:
                pat = pat[:-1]
            anchored = pat.startswith("/")
            if anchored:
                pat = pat[1:]
            self.rules.append((neg, dir_only, anchored, pat))

    def is_ignored(
        self,
        rel_path: str,
        is_dir: bool,
    ) -> bool:
        rel_path_norm = rel_path.replace("\\", "/")
        base = rel_path_norm.rsplit("/", 1)[-1]
        ignored = False
        for neg, dir_only, anchored, pat in self.rules:
            if dir_only and not is_dir:
                continue
            matched = False
            if anchored:
                if self.fnmatch.fnmatch(rel_path_norm, pat):
                    matched = True
            else:
                if self.fnmatch.fnmatch(
                    rel_path_norm, pat
                ) or self.fnmatch.fnmatch(base, pat):
                    matched = True
            if matched:
                ignored = not neg
        return ignored


def build_matcher(
    patterns: List[str],
) -> Callable[[str, bool], bool]:
    if not patterns:
        return lambda rel, is_dir: False
    try:
        import pathspec  # type: ignore

        spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
        return lambda rel, is_dir: spec.match_file(rel)
    except Exception:
        return IgnoreMatcher(patterns).is_ignored


def collect_files(
    root: Path,
    matcher,
    code_only: bool,
    include_git: bool,
) -> List[Path]:
    out: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""
        in_git = rel_dir == ".git" or rel_dir.startswith(".git/")
        # prune dirs
        keep = []
        for d in dirnames:
            rel = f"{rel_dir}/{d}" if rel_dir else d
            if d == ".git":
                if include_git:
                    keep.append(d)
                continue
            if in_git:
                keep.append(d)
                continue
            if matcher(rel, True):
                continue
            keep.append(d)
        dirnames[:] = keep
        # files
        for f in filenames:
            relf = f"{rel_dir}/{f}" if rel_dir else f
            p = Path(dirpath) / f
            if in_git:
                out.append(p)
                continue
            if matcher(relf, False):
                continue
            if code_only:
                if p.suffix in CODE_EXTENSIONS:
                    out.append(p)
            else:
                out.append(p)
    return out


def is_remote_destination(
    dest: str,
) -> bool:
    # Basic heuristic: pattern host:path or user@host:path
    if ":" not in dest:
        return False
    left, _ = dest.split(":", 1)
    if not left:
        return False
    # Disallow Windows drive letters like C:\ (unlikely on Linux)
    if len(left) == 1 and left.isalpha():
        return False
    return True


def ensure_remote_dir(
    dest: str,
) -> int:
    host, path = dest.split(":", 1)
    cmd = ["ssh", host, "mkdir", "-p", f"{path}"]
    return subprocess.call(cmd)


def copy_local(
    files: List[Path],
    root: Path,
    dest: Path,
    overwrite: bool,
):
    for src in files:
        rel = src.relative_to(root)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            continue
        shutil.copy2(src, target)


def stage_files(
    files: List[Path],
    root: Path,
    staging: Path,
):
    for src in files:
        rel = src.relative_to(root)
        target = staging / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)


def scp_transfer(
    staging: Path,
    dest: str,
    scp_args: List[str],
) -> int:
    entries = list(staging.iterdir())
    if not entries:
        return 0
    # Use separate paths to avoid nesting staging dir
    cmd = (
        ["scp", "-r"]
        + scp_args
        + [str(p) for p in entries]
        + [dest.rstrip("/") + "/"]
    )
    return subprocess.call(cmd)


def parse_args(
    argv: Optional[List[str]] = None,
):
    ap = argparse.ArgumentParser(
        description="Copy project code files honoring .gitignore."
    )
    ap.add_argument("--root", default=".", help="Project root (default: .)")
    ap.add_argument(
        "--dest",
        required=True,
        help="Destination directory (local path or [user@]host:/path).",
    )
    ap.add_argument(
        "--all-files",
        action="store_true",
        help="Include non-code files (still honors .gitignore).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local files.",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="List files without copying."
    )
    ap.add_argument(
        "--mkdir",
        action="store_true",
        help="Create remote destination directory via ssh before scp.",
    )
    ap.add_argument(
        "--include-git",
        action="store_true",
        help="Also copy the .git directory (full repo state).",
    )
    ap.add_argument(
        "--scp-args",
        default="",
        help="Extra arguments for scp (e.g. '-C -l 8192').",
    )
    return ap.parse_args(argv)


def main(
    argv: Optional[List[str]] = None,
) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"Root not found: {root}", file=sys.stderr)
        return 2
    patterns = load_gitignore(root)
    matcher = build_matcher(patterns)
    files = collect_files(
        root,
        matcher,
        code_only=not args.all_files,
        include_git=args.include_git,
    )
    files_sorted = sorted(files)
    print(f"Selected {len(files_sorted)} files.")
    if args.dry_run:
        for p in files_sorted:
            print(p.relative_to(root))
        return 0
    dest = args.dest
    remote = is_remote_destination(dest)
    if remote:
        if args.mkdir:
            rc = ensure_remote_dir(dest)
            if rc != 0:
                print("Failed to create remote directory.", file=sys.stderr)
                return rc
        with tempfile.TemporaryDirectory(prefix="copy_codes_") as tmpd:
            staging = Path(tmpd)
            stage_files(files_sorted, root, staging)
            extra = args.scp_args.split() if args.scp_args else []
            rc = scp_transfer(staging, dest, extra)
            if rc == 0:
                print("Remote copy complete.")
            else:
                print(f"scp failed with code {rc}", file=sys.stderr)
            return rc
    else:
        dest_path = Path(dest).resolve()
        dest_path.mkdir(parents=True, exist_ok=True)
        copy_local(files_sorted, root, dest_path, overwrite=args.overwrite)
        print(f"Local copy complete at {dest_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
