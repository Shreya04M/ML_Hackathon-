"""FastAPI entry point."""
from fastapi import FastAPI
from .routes import router

app = FastAPI(title="PathSnap API", version="0.1.0")
app.include_router(router)
