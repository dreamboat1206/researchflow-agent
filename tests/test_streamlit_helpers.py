from __future__ import annotations

import ast
from pathlib import Path


def _load_helper(name: str):
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    module_ast = ast.parse(source)
    helper_nodes = [
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_figure_query_params", "_figure_image_path"}
    ]
    helper_module = ast.Module(body=helper_nodes, type_ignores=[])
    ast.fix_missing_locations(helper_module)
    namespace = {"Path": Path}
    exec(compile(helper_module, "streamlit_helpers", "exec"), namespace)
    return namespace[name]


def test_figure_query_params_builds_filters() -> None:
    helper = _load_helper("_figure_query_params")

    assert helper(" 7 ", " ZoomDet ") == {"paper_id": 7, "title": "ZoomDet"}
    assert helper("", "") == {}


def test_figure_image_path_maps_container_path() -> None:
    helper = _load_helper("_figure_image_path")
    local_path = Path("data/figures/demo.png")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(b"demo")

    assert helper("data/figures/demo.png") == str(local_path)
    assert helper("/app/data/figures/demo.png") == str(local_path)
    assert helper("D:/old/project/data/figures/demo.png") == str(local_path)
