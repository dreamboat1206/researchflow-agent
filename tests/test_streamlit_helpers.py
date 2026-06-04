from __future__ import annotations

import ast
from pathlib import Path


def _module_ast(path: str) -> ast.Module:
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def test_streamlit_navigation_includes_paper_reader() -> None:
    module_ast = _module_ast("app/streamlit_app.py")
    constants = [node.value for node in ast.walk(module_ast) if isinstance(node, ast.Constant)]

    assert "Paper Reader" in constants
    assert "Dashboard" in constants
    assert "System / Debug" in constants


def test_state_defaults_include_reader_fields() -> None:
    source = Path("app/ui/state.py").read_text(encoding="utf-8")

    assert "def init_state" in source
    assert "reader_selected_page" in source
    assert "reader_selected_chunk_id" in source
    assert "selected_paper_id" in source
    assert "last_qa_result" in source


def test_theme_and_card_components_exist() -> None:
    styles = Path("app/ui/styles.py").read_text(encoding="utf-8")

    assert "def apply_theme" in styles
    assert "#4B267A" in styles
    assert "#F5F0FF" in styles
    assert "rf-pill" in styles
    assert "rf-card" in styles
    assert Path("app/ui/components/hero.py").exists()
    assert Path("app/ui/components/metric_card.py").exists()
    assert Path("app/ui/components/feature_card.py").exists()


def test_dashboard_uses_hero_and_feature_cards() -> None:
    dashboard = Path("app/ui/pages/dashboard.py").read_text(encoding="utf-8")

    assert "Summarize, search and organize your papers" in dashboard
    assert "Add & Organize" in dashboard
    assert "Search Evidence" in dashboard
    assert "Ask with Citations" in dashboard


def test_no_disallowed_frontend_directories_created() -> None:
    assert not Path("src/ui").exists()
    assert not Path("frontend").exists()
    assert not Path("web").exists()
