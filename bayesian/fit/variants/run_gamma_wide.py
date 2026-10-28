import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fit_variants as FV

FV.F.HS_B = (1e-3, 20.0)
FV.stage_fit(
    n_folds=5,
    seed=42,
    variants=["full", "no_hier", "no_hier_oracle"],
    out_dir=HERE / "gamma_wide",
)
