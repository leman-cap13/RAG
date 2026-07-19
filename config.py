from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str

    embed_model: str = "gemini-embedding-001"
    embed_batch_size: int = 100

    gen_model: str = "gemini-2.5-flash"
    gen_temperature: float = 0.2

    chroma_path: str = "chroma_db"
    data_dir: str = "data"

    chunk_size: int = 800
    chunk_overlap: int = 150

    top_k: int = 4

    redis_url: str = 'redis://localhost:6379/0'
    cache_ttl: int = 3600
    cache_similarity_threshold: float = 0.15

    log_level: str = 'INFO'
    log_format: str = 'text'

    rabbitmq_url: str = 'amqp://guest:guest@localhost:5672/'
    rabbitmq_ask_queue: str = 'ask_queue'
    rabbitmq_rpc_timeout: int = 45
    rabbitmq_max_queue_depth: int = 300


settings = Settings()