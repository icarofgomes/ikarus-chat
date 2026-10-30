# Continuidade do job-match

Arquivo temporário para a IA neste computador. Apague depois do deploy.

O código desta rota já está no GitHub. O currículo denso não está. Ele fica em `private/curriculo.md`, fora do git. Este checkout novo não traz `private/` nem `data/index.sqlite`.

## O que não pode acontecer

- Não dê commit em `private/curriculo.md`, em `.env` nem em chave nenhuma.
- Não rode o ingest em `corpus/curriculo.md`. Esse arquivo é a demonstração pública. Indexá-lo substitui o índice denso.
- Não use a chave da OpenAI nem o `gpt-4o-mini` no job-match.
- Não reutilize `CHAT_SHARED_SECRET` nem `OPENROUTER_API_KEY` do chat. O job-match tem segredo e chave próprios.
- Não ligue o script do job-finder nesta leva. O notebook ainda não chama esta URL.
- Não faça deploy antes do ingest. A imagem copia `data/index.sqlite`. Sem esse arquivo gerado daqui, a imagem não sobe.

## Variáveis

No `.env` local, além do que o chat já usa:

```bash
CORPUS_PATH=private/curriculo.md
JOB_MATCH_SECRET=
JOB_MATCH_OPENROUTER_API_KEY=
JOB_MATCH_EMBED_MODEL=openai/text-embedding-3-small
JOB_MATCH_CHAT_MODEL=deepseek/deepseek-v4-flash
```

`JOB_MATCH_OPENROUTER_API_KEY` é a chave da OpenRouter que já está no OpenClaw, não a chave do chat do portfólio.

`JOB_MATCH_EMBED_MODEL` tem de continuar `openai/text-embedding-3-small`. É o mesmo modelo dos vetores do índice. Outro modelo não casa com a busca.

No Cloud Run, as variáveis do chat continuam. Somam-se só estas duas, como ambiente, sem entrar na imagem:

- `JOB_MATCH_SECRET`
- `JOB_MATCH_OPENROUTER_API_KEY`

Se uma das duas faltar, o chat público sobe e `POST /v1/job-match` responde 503.

## Ordem

1. Confirme que `private/curriculo.md` existe nesta máquina e que o git o ignora: `git check-ignore -v private/curriculo.md`.
2. Rode `python scripts/ingest.py`. Se o privado não existir, o script para e não grava o índice.
3. Confira que `data/index.sqlite` foi gerado depois desse ingest.
4. Suba o serviço como já era feito. O `Dockerfile` copia esse sqlite e não copia `private/`.
5. Publique `JOB_MATCH_SECRET` e `JOB_MATCH_OPENROUTER_API_KEY` no Cloud Run.
6. O chat público tem de continuar respondendo em `POST /v1/chat`.

## Contrato da rota

`POST /v1/job-match`, header `x-job-match-secret`.

Entram `url`, `title`, `company`, `text`, `source`. Saem `match_percent`, `decision` (`aplicar`, `talvez`, `pular`), `evidence`, `gaps`, `summary`.

A nota é do `deepseek/deepseek-v4-flash`. A busca traz até 8 trechos. Uma citação só conta se o texto aparecer num trecho recuperado. Sem trecho, a resposta é 0 e `pular`, sem chamar o modelo. `aplicar` começa em 75. `talvez` começa em 50. O serviço não grava a vaga. O teto de tokens é outro, separado do chat, na tabela `job_match_spend`.
