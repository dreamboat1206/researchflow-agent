from fastapi import FastAPI

from api.routes_figure_qa import router as figure_qa_router
from api.routes_figure_search import router as figure_search_router
from api.routes_figures import router as figures_router
from api.routes_paper import router as paper_router
from api.routes_qa import router as qa_router


app = FastAPI(
    title="ResearchFlow-Agent API",
    description="Backend API for the ResearchFlow-Agent project.",
    version="0.1.0",
)

app.include_router(paper_router)
app.include_router(figure_search_router)
app.include_router(qa_router)
app.include_router(figure_qa_router)
app.include_router(figures_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
