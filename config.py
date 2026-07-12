from pydantic_settings import BaseSettings, SecretsSettingsSource

class Settings(BaseSettings):
    model_config = SecretsSettingsSource(env_file = ".env", extra = "ignore")

    gemini_api_key: str

    embed_model: str = "gemini-embedding-001"
    embed_batch_size: int = 100
    gen_model: str = "gemini-3.5-flash"
    gen_temperature: float = 0.2

    min_similarity: float = 0.25

    chroma_path: str = "chroma_db"
    data_dir: str = "data"

    chunk_size: int = 800
    chunk_overlap: int = 150

    default_prompt: str = """Sən tələbələrə universitet seçimində kömək edən səmimi, isti münasibətli bir köməkçisən.

        Qaydalar:
        - Cavabı YALNIZ aşağıdakı mənbələrdəki məlumata əsasən ver.
        - Rəsmi, quru dillə yox, tələbə ilə söhbət edir kimi mehriban və anlaşıqlı tonda yaz.
        - Əgər mənbələrdə sualın cavabını etibarlı şəkildə mənbələrdən çıxara bilməsən, bunu səmimi şəkildə bildir.
        - Uyğun olduğu yerlərdə istifadə etdiyin mənbəni [1], [2] və s. şəklində qeyd et.
        - Cavabı Azərbaycan dilində yaz.

        Mənbələr:
        {context}

        Sual: {question}

        Cavab:"""

    top_k: int = 4

settings = Settings()

