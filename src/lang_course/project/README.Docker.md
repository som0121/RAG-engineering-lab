# Running the API with Docker

All commands run from `src/lang_course/project/`. The container reads its settings
(`GOOGLE_API_KEY`, `LANGSMITH_*`, …) from `.env`, which is never copied into the image.

## Build and start

```bash
docker compose up --build -d     # build the image and start it in the background
docker compose ps                # STATUS shows "healthy" about 20 s after start
docker compose logs -f           # follow the JSON logs (Ctrl+C to stop following)
```

The API is at http://localhost:8000 (docs at http://localhost:8000/docs).

```powershell
curl.exe http://localhost:8000/health
```

## Stop

```bash
docker compose down
```

## Rebuilding

| What changed | Command | Time |
|---|---|---|
| Code in `app/` | `docker compose up --build -d` | seconds (only the `COPY app/` layer is rebuilt) |
| `pyproject.toml` / `uv.lock` | `docker compose up --build -d` | dependencies are reinstalled (downloads are cached between builds) |
| Only `.env` | `docker compose up -d --force-recreate` | no rebuild needed |

Port 8000 must be free: stop a local `uvicorn` first, or change the left side of
`ports` in `compose.yaml` (for example `"8010:8000"`).

## What's in the image

- `python:3.12-slim` with `uv` 0.13.1 copied from `ghcr.io/astral-sh/uv`
- Runtime dependencies only (`uv sync --frozen --no-dev`), installed into `/app/.venv`
- The `app/` package, run by `uvicorn app.main:app` as the non-root user `appuser`
- A health check that calls `/health` every 30 s

Add a dependency with `uv add <package>` (this updates `pyproject.toml` and `uv.lock`),
then rebuild.
