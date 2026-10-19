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

Fill `OPENROUTER_API_KEY` and `CHAT_SHARED_SECRET` in `.env`. Then build the index once. Repeat only when the markdown changes.

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

- 400 characters per message, 8 user turns, 16 messages.
- 6 requests every 10 minutes and 20 per day, per IP.
- A daily token budget, checked before the model call.
- Answers are capped at 400 tokens. Reasoning is off.
- Below the similarity floor, the service returns a fixed sentence and does not call the chat model.
- On the 8th turn the answer points to WhatsApp. The next turn is refused.

The counters live in the same SQLite file as the index. They reset when the instance sleeps. That matches a host that scales to zero.

## Models

Both calls use OpenRouter. The ids are environment variables:

- `EMBED_MODEL` defaults to `baai/bge-m3`
- `CHAT_MODEL` defaults to `deepseek/deepseek-v4-flash`

Changing either id does not require a code change. Rebuild the index after changing the embedding model.

## Cloud Run

Build the index locally first. The image copies `data/index.sqlite`.

Deploy with minimum instances at 0 and request-based billing, so an idle service does not keep a machine running. The URL stays up. The first question after a pause waits for the container to start.

Set `CHAT_SHARED_SECRET`, `OPENROUTER_API_KEY`, and the model ids as environment variables. Do not bake the key into the image.

## Tests

```bash
pytest
```

The tests do not call OpenRouter.
