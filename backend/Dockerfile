FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY ml/ ./ml/
# --extra-index-url: sentence-transformers pulls in torch, whose default PyPI
# wheel on Linux is the CUDA build (pulls several GB of nvidia-* packages
# this CPU-only container never uses). The CPU wheel index gives pip a
# CPU-only torch to prefer instead — same package, without the CUDA payload.
RUN pip install --no-cache-dir -e ./ml --extra-index-url https://download.pytorch.org/whl/cpu

COPY backend/pyproject.toml ./backend/pyproject.toml
COPY backend/app ./backend/app
RUN pip install --no-cache-dir -e ./backend

COPY backend/alembic.ini ./backend/alembic.ini
COPY backend/alembic ./backend/alembic
COPY scripts/ ./scripts/

WORKDIR /app/backend

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
