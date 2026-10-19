FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY corpus ./corpus
COPY scripts ./scripts
COPY data/index.sqlite /app/data/index.sqlite

RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn portfolio_chat.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8080}"]
