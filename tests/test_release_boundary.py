from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_FILENAMES = {
    "accuracy_by_clip.csv",
    "summary.csv",
    "manifest.json",
    "Subject Data.xlsx",
}

FORBIDDEN_SUFFIXES = {".avi", ".npy", ".npz", ".mat", ".parquet", ".pth"}


def test_release_tree_contains_no_row_level_or_gated_artifacts():
    """Fail if local subject rows, source data, caches, or weights enter a release."""
    violations = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or ".venv" in path.parts:
            continue
        if path.name in FORBIDDEN_FILENAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(str(path.relative_to(REPO_ROOT)))

    assert not violations, "private or licensed artifacts found: " + ", ".join(violations)
