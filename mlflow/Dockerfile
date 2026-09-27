FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mlflow>=2.14

RUN mkdir -p /mlflow/artifacts /mlflow/db

EXPOSE 5000

# --allowed-hosts '*': this server is only ever reached from inside the
# docker-compose network or via localhost port-forwarding, never exposed to
# the internet, so DNS-rebinding protection (which otherwise rejects the
# "mlflow:5000" Host header other containers use) isn't a real threat here.
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/db/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", "--port", "5000", "--allowed-hosts", "*"]
