from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embed_model: str = "baai/bge-m3"
    chat_model: str = "deepseek/deepseek-v4-flash"
    chat_shared_secret: str = ""
    index_path: str = "data/index.sqlite"
    corpus_path: str = "corpus/curriculo.md"
    top_k: int = 4
    score_floor: float = 0.4
    max_message_chars: int = 400
    max_turns: int = 8
    max_messages: int = 16
    rate_window_seconds: int = 600
    rate_window_max: int = 6
    daily_max: int = 20
    daily_token_budget: int = 200_000
    max_output_tokens: int = 400
    whatsapp_url: str = "https://wa.me/5511966370030"
