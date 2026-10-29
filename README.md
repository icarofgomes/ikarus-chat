# Portfolio chat

RAG API for the public career of Icaro Ferreira Gomes. The portfolio page sends a question. This service retrieves the matching sections and asks a model to answer from those sections only.

The corpus in `corpus/curriculo.md` is public on purpose. It repeats what the portfolio already shows. Internal notes, incidents, revenue, and customer data do not belong in this file: a public repository shows the markdown even when the chat is closed.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Fill `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, and `CHAT_SHARED_SECRET` in `.env`. Then build the index once. Repeat when the markdown changes, and also when the embedding model changes.

```bash
python scripts/ingest.py
uvicorn portfolio_chat.main:create_app --factory --port 8080
```

`GET /health` returns `ready: true` after the index exists.

## Ask

```bash
curl -s http://127.0.0.1:8080/v1/chat \
  -H "content-type: application/json" \
  -H "x-chat-secret: $CHAT_SHARED_SECRET" \
  -H "x-chat-client-ip: 203.0.113.10" \
  -d '{"messages":[{"role":"user","content":"O que você fez na KingPanda?"}]}'
```

The portfolio proxy sends the visitor IP in `X-Chat-Client-Ip`. The shared secret stays on the server. Callers without it receive 401.

The panel keeps the open conversation and sends the recent turns with each question. This process does not store the text. Closing the panel ends the session.

A follow-up such as "E na SeguroBet?" is embedded together with the previous user turn, so the search keeps the subject.

## Limits

- 400 characters on each question. A previous answer may be up to 2,000 characters.
- 8 user turns, 16 messages.
- 6 requests every 10 minutes and 20 per day, per IP.
- Global token budgets, reserved before the model call: 15,000 per minute, 60,000 per hour, and 200,000 per day.
- Answers are capped at 160 tokens. The OpenRouter fallback keeps reasoning off.
- Below the similarity floor, the service returns a fixed sentence and does not call the chat model.
- On the 8th turn the answer points to WhatsApp. The next turn is refused.

The counters live in the same SQLite file as the index. They reset when the instance sleeps. That matches a host that scales to zero.

## Models

OpenAI is the primary provider for the embedding and the answer. OpenRouter runs only when that call fails. With no `OPENAI_API_KEY`, every call goes straight to OpenRouter.

- `OPENAI_EMBED_MODEL` defaults to `text-embedding-3-small`
- `OPENAI_CHAT_MODEL` defaults to `gpt-4o-mini`
- `OPENROUTER_EMBED_MODEL` defaults to `openai/text-embedding-3-small`
- `OPENROUTER_CHAT_MODEL` defaults to `deepseek/deepseek-v4-flash`

The two embedding ids must stay the same model. A fallback vector from another model does not match the index. Changing either embedding id requires a rebuild.

## Cloud Run

Build the index locally first. The image copies `data/index.sqlite`.

Deploy with minimum instances at 0 and request-based billing, so an idle service does not keep a machine running. The URL stays up. The first question after a pause waits for the container to start.

Set `CHAT_SHARED_SECRET`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, and the model ids as environment variables. Do not bake a key into the image.

## Tests

```bash
pytest
```

The tests do not call OpenAI or OpenRouter.
