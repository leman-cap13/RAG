from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file =".env", extra = "ignore")

    gemini_api_key : str

    embed_model: str = "gemini-embedding-001"
    embed_batch_size: int = 100

    gen_model: str = "gemini-2.5-flash"
    gen_temperature: float = 0.2

    chrome_path: str = "chrome_db"
    data_dir: str = "data"

    chunk_size: int = 800

    chunk_overlap: int = 150

    top_k: int = 5

settings = Settings()
