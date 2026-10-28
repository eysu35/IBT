import os
from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parent
_REPO = _ANALYSIS.parent

SIM_DIR = Path(os.environ.get("BAYES_SIM_DIR", str(_REPO / "bayesian" / "data")))
FIG_DIR = Path(os.environ.get("BAYES_FIG_DIR", str(_ANALYSIS / "figures")))
OUT_DIR = Path(os.environ.get("BAYES_OUT_DIR", str(_ANALYSIS / "results")))
FIT_JSON = Path(
    os.environ.get(
        "BAYES_FIT_JSON", str(_REPO / "bayesian/fit/data/model_comparison.json")
    )
)

SIM_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
