.PHONY: up down logs test lint seed reproduce

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

test:
	cd backend && python -m pytest
	cd ml && python -m pytest
	cd frontend && npm test

lint:
	cd backend && ruff check . && mypy app
	cd frontend && npm run lint && npm run format

seed:
	docker compose exec api python /app/scripts/seed_demo.py

reproduce:
	docker compose exec api python /app/scripts/reproduce_run.py --run-id $(RUN_ID)
