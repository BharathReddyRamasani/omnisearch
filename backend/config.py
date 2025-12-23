from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    app_name: str = "OmniSearch AI"
    version: str = "2.0.0"
    debug: bool = True
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    
    class Config:
        env_file = ".env"

settings = Settings()
