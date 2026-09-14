import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Pharma Complaint AI Copilot"
    APP_ENV: str = "development"
    DEBUG: bool = True
    
    # Database: SQLite by default for instant local execution, swap to PostgreSQL seamlessly
    DATABASE_URL: str = "sqlite:///./complaints.db"
    
    # Groq API Configuration
    GROQ_API_KEY: str = ""
    # Low-latency model for extraction, completeness check, and summary
    GROQ_EXTRACTION_MODEL: str = "llama-3.1-8b-instant"
    # High-reasoning model for risk classification rationale and CAPA direction
    GROQ_REASONING_MODEL: str = "llama-3.3-70b-versatile"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
