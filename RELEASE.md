# Releasing to PyPI

The package is build-ready (`hatchling`, src layout, metadata + classifiers,
deps pinned). These are the final steps — **you run the upload** with your own
PyPI token.

## 1. Verify version
Bump `version` in `pyproject.toml` **and** `__version__` in
`src/fittok/__init__.py` (keep them in sync). Current: `0.3.0`.

## 2. Build
```bash
python -m pip install --upgrade build twine
rm -rf dist
python -m build            # creates dist/*.whl and dist/*.tar.gz
python -m twine check dist/*
```

## 3. (Recommended) Test on TestPyPI first
```bash
python -m twine upload --repository testpypi dist/*
# then in a clean venv:
pip install -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple fittok
```

## 4. Upload to PyPI
```bash
python -m twine upload dist/*
# username: __token__
# password: pypi-<your-API-token>
```

## 5. After publish — how users consume it
```bash
pip install fittok          # core (retrieval + embeddings)
pip install "fittok[ui]"    # + Gradio/pyvis graph visualizer
```
Register the MCP server (user scope, available in every repo):
```bash
claude mcp add fittok --scope user -- python -m fittok
```
Optional pre-warm so the first query is instant:
```bash
fittok-index /path/to/repo
```

## Notes / gotchas baked into the package
- `requires-python = ">=3.10"` (the `mcp` dep needs it; 3.9 fails).
- `transformers` pinned `<5` (5.x breaks llmlingua model loading).
- First use auto-indexes (graph + embeddings) and caches under `~/.cache/fittok`;
  embeddings are content-keyed so changes only re-embed what changed.
- The embedding model (`all-MiniLM-L6-v2`, ~90 MB) downloads from HuggingFace on
  first run — document this for users behind firewalls.
