import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine, Base, SessionLocal
from app.routers import complaints
from app.seed_data import seed_baseline_complaints

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pharma_copilot")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables
    logger.info("Initializing database schema...")
    Base.metadata.create_all(bind=engine)
    
    # Seed baseline records
    db = SessionLocal()
    try:
        seed_baseline_complaints(db)
    finally:
        db.close()
        
    logger.info("Backend initialized and ready.")
    yield

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="AI-Powered GxP Customer Complaint Management System for Pharma Manufacturing (API & FDF)",
    lifespan=lifespan
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(complaints.router)

@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "env": settings.APP_ENV,
        "database": settings.DATABASE_URL.split(":///")[0],
        "groq_configured": bool(settings.GROQ_API_KEY),
        "extraction_model": settings.GROQ_EXTRACTION_MODEL,
        "reasoning_model": settings.GROQ_REASONING_MODEL
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
