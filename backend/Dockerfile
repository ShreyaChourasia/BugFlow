FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY ml/ ./ml/
RUN pip install --no-cache-dir -e ./ml

COPY backend/pyproject.toml ./backend/pyproject.toml
COPY backend/app ./backend/app
RUN pip install --no-cache-dir -e ./backend

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
