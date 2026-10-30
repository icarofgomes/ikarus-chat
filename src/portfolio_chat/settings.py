from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_embed_model: str = "openai/text-embedding-3-small"
    openrouter_chat_model: str = "deepseek/deepseek-v4-flash"
    chat_shared_secret: str = ""
    index_path: str = "data/index.sqlite"
    corpus_path: str = "private/curriculo.md"
    job_match_secret: str = ""
    job_match_openrouter_api_key: str = ""
    job_match_embed_model: str = "openai/text-embedding-3-small"
    job_match_chat_model: str = "deepseek/deepseek-v4-flash"
    job_match_top_k: int = 8
    job_match_score_floor: float = 0.35
    job_match_max_text_chars: int = 8_000
    job_match_max_output_tokens: int = 800
    job_match_daily_token_budget: int = 100_000
    top_k: int = 4
    score_floor: float = 0.4
    max_message_chars: int = 400
    max_reply_chars: int = 2_000
    max_turns: int = 8
    max_messages: int = 16
    rate_window_seconds: int = 600
    rate_window_max: int = 6
    daily_max: int = 20
    token_minute_budget: int = 15_000
    token_hour_budget: int = 60_000
    daily_token_budget: int = 200_000
    prompt_token_ceiling: int = 2_000
    max_output_tokens: int = 160
    whatsapp_url: str = "https://wa.me/5511966370030"
