from fastapi import FastAPI


app = FastAPI(
    title="ResearchFlow-Agent API",
    description="Backend API for the ResearchFlow-Agent project.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

