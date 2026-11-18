"""Execute the figure notebooks and save their refreshed outputs (figures/ and notebook cells).

Usage: python scripts/19_render_notebooks.py [Journal_Figures.ipynb EDA_Visuelle2.ipynb]
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def render(name):
    path = NOTEBOOKS / name
    nb = nbformat.read(path, as_version=4)

    def progress(cell, cell_index, **kwargs):
        print(f"{name}: cell {cell_index}", flush=True)

    NotebookClient(nb, timeout=900, resources={"metadata": {"path": str(NOTEBOOKS)}},
                   on_cell_execute=progress).execute()
    errors = [(i, o.get("evalue")) for i, c in enumerate(nb.cells)
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errors:
        raise RuntimeError(f"Rendering failed: {errors}")
    nbformat.write(nb, path)
    print(f"Completed {name}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for notebook in sys.argv[1:] or ["Journal_Figures.ipynb", "EDA_Visuelle2.ipynb"]:
        render(notebook)
