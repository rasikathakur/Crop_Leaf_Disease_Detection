from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.db.database import init_db, test_connection
from app.api.endpoints import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Backend API...")
    if test_connection():
        print("Database connected")
        init_db()
    yield
    print("Shutting down Backend API...")

app = FastAPI(
    title="Crop Disease Backend API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/health")
def health():
    return {"status": "healthy", "service": "backend"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
