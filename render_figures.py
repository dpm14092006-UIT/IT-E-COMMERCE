"""Execute a notebook and save its refreshed figure outputs."""
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8')
for name in sys.argv[1:] or ['Journal_Figures.ipynb', 'EDA_Visuelle2.ipynb']:
    path = root / name
    nb = nbformat.read(path, as_version=4)
    def progress(cell, cell_index, **kwargs):
        print(f'{name}: cell {cell_index}', flush=True)
    client = NotebookClient(nb, timeout=900, resources={'metadata': {'path': str(root)}},
                            on_cell_execute=progress)
    client.execute()
    errors = [(i, o.get('evalue')) for i, c in enumerate(nb.cells)
              for o in c.get('outputs', []) if o.get('output_type') == 'error']
    if errors:
        raise RuntimeError(f'Rendering failed: {errors}')
    nbformat.write(nb, path)
    print(f'Completed {name}', flush=True)
