from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.studies.common.xinye_phase12_joint_runner import run_xinye_phase12_joint_study


if __name__ == "__main__":
    run_xinye_phase12_joint_study(Path(__file__).resolve().parent)
