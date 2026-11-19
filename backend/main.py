
"""FastAPI entry point for PathSnap."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from .routes import router

app = FastAPI(
    title="PathSnap API",
    version="1.0.0",
    description="Map-matching and route reconstruction API"
)


@app.middleware("http")
async def normalize_double_slashes(request: Request, call_next):
    path = request.scope.get("path", "")
    if path.startswith("//"):
        request.scope["path"] = "/" + path.lstrip("/")
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "PathSnap API",
        "docs": "/docs"
    }