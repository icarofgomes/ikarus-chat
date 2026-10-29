from pathlib import Path

from portfolio_chat.settings import Settings


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "chat_shared_secret": "secret",
        "openai_api_key": "",
        "openrouter_api_key": "test-key",
        "index_path": str(tmp_path / "index.sqlite"),
        "score_floor": 0.5,
        "daily_token_budget": 100_000,
        "max_output_tokens": 50,
        "whatsapp_url": "https://wa.me/5511966370030",
    }
    values.update(overrides)
    return Settings.model_validate(values)
