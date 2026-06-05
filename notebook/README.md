# notebook/ — manual testing playground

Jupyter notebooks for poking at the `ccga` package by hand. The venv lives at the
**project root** (`../.venv`).

## Layout

- `../.venv/` — virtualenv (deps from `../requirements.txt`).
- `scratch.ipynb` — starter notebook (imports `ccga`, demos `print_null`).

The `ccga` package is installed **editable** (`pip install -e .`, driven by the
project-root `pyproject.toml`), so `import ccga` works in any kernel/cwd that
uses this venv — edits to the source are picked up live.

## Launch

```sh
.venv/bin/jupyter lab notebook/scratch.ipynb
```

In the notebook, pick the **CCGA (.venv)** kernel (registered as `ccga`).

## Recreate from scratch (run from project root)

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # editable ccga + jupyter
.venv/bin/python -m ipykernel install --user --name ccga --display-name "CCGA (.venv)"
```
