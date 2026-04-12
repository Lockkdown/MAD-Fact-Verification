"""FastAPI entry point for ViMAD web demo."""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from routers.predict import router as predict_router
from services.plm_service import plm_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    plm_service.load()
    yield
    plm_service.unload()


app = FastAPI(title="ViMAD Demo API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost", "http://localhost:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": plm_service.loaded_models}
