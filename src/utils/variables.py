"""Repository paths, commit identifier, and figure colors."""

from pathlib import Path

from git import Repo

_current_file = Path(__file__).resolve()
_repo_root = _current_file.parent.parent.parent

ROOT_FOLDER = _repo_root
DATA_FOLDER = ROOT_FOLDER / "data"
SUVR_FOLDER = DATA_FOLDER / "processed" / "suvr"
GRAPHICAL_LASSO_FOLDER = DATA_FOLDER / "processed" / "graphical_lasso"
INDV_CONN_MATRIX_FOLDER = DATA_FOLDER / "processed" / "individual_conn_matrix"

try:
    GIT_TAG = Repo(ROOT_FOLDER).head.commit.hexsha
except:
    GIT_TAG = "no-github-repo"

COLORS = ["#FF0000", "#00FF00", "#0000FF", "#FF00FF", "#00FFFF", "#f39c12", "#9b59b6"]
