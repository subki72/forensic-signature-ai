.PHONY: install test lint format run docker-build clean

install:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest tests/ -v

lint:
	ruff check api src tests

format:
	ruff format api src tests

run:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t forensic-signature-ai:latest .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
