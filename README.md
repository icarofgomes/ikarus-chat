# Portfolio chat

RAG API for the public career of Icaro Ferreira Gomes. The portfolio page sends a question. This service retrieves the matching sections and asks a model to answer from those sections only.

`corpus/curriculo.md` is a public demonstration. It repeats what the portfolio already shows. The index used in production is built from `private/curriculo.md`, which stays out of git. Re-running ingest against the public file replaces that denser index.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Fill `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, and `CHAT_SHARED_SECRET` in `.env`. For job match, also fill `JOB_MATCH_SECRET` and `JOB_MATCH_OPENROUTER_API_KEY`. That key is the OpenRouter key already stored in OpenClaw. It is separate from the chat key. Then build the index once from `private/curriculo.md`. Repeat when that markdown changes, and also when the embedding model changes. If `private/curriculo.md` is missing, ingest stops and leaves the index untouched.

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

## Job match

`POST /v1/job-match` scores one job against the indexed resume. The notebook sends the job text and `x-job-match-secret`. The service does not store the job.

Optional fields classify the posting before the score: `required`, `also_accepts`, `differentials`, and `exclusive`. Empty lists mean the model infers those classes from the text. Feed chrome and a line that is mostly hashtags are removed before retrieval and before the prompt.

Retrieval runs two searches: the title plus those lists, and the cleaned body. Each search uses `JOB_MATCH_TOP_K`. The union keeps half of the slots for each search, then fills the rest by similarity, still capped at `JOB_MATCH_TOP_K`.

The score comes from `deepseek/deepseek-v4-flash`. An axis counts when at least one excerpt appears in a retrieved section. The domain axis drops out when the posting does not ask for a sector, and the other three weights scale back to 100. The response keeps `evidence` and adds `axes`, with the score, the points, and the excerpts that were accepted. With no retrieved section, the route returns `pular` and does not call the chat model.

This route does not use the public chat limits, the OpenAI chat model, or `CHAT_SHARED_SECRET`. Retrieval uses `openai/text-embedding-3-small` through the job-match OpenRouter key, the same model as the index.

If `JOB_MATCH_SECRET` or `JOB_MATCH_OPENROUTER_API_KEY` is empty, the public chat still starts and this route returns 503.

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

Set `CHAT_SHARED_SECRET`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, and the model ids as environment variables. Job match also needs `JOB_MATCH_SECRET` and `JOB_MATCH_OPENROUTER_API_KEY`. Do not bake a key into the image. The image still needs `private/curriculo.md` present when ingest runs, and that file is not part of the git checkout.

## Tests

```bash
pytest
```

The tests do not call OpenAI or OpenRouter.
