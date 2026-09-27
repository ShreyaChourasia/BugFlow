"""US-06: per-commit JIT defect-prediction features (see master prompt §8) —
size, diffusion (files/dirs/subsystems/entropy), author experience, fix flag."""

import math
from pathlib import PurePosixPath

from bugflow_ml.mining.git_miner import ModifiedFileInfo


def compute_commit_features(
    files: list[ModifiedFileInfo],
    lines_added: int,
    lines_deleted: int,
    is_fix: bool,
    author_prior_commits: int,
) -> dict:
    files_changed = len(files)
    directories = {str(PurePosixPath(f.path).parent) for f in files if f.path}
    subsystems = {PurePosixPath(f.path).parts[0] for f in files if PurePosixPath(f.path).parts}

    lines_touched = [f.added_lines + f.deleted_lines for f in files]
    total_touched = sum(lines_touched)
    entropy = 0.0
    if total_touched > 0:
        for n in lines_touched:
            if n == 0:
                continue
            p = n / total_touched
            entropy -= p * math.log2(p)

    return {
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
        "churn": lines_added + lines_deleted,
        "files_changed": files_changed,
        "directories_touched": len(directories),
        "subsystems_touched": len(subsystems),
        "entropy": round(entropy, 4),
        "author_prior_commits": author_prior_commits,
        "is_fix": is_fix,
    }
