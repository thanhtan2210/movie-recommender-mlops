# Makefile for Big-Data-MLOps-System

.PHONY: install test lint format run benchmark evaluate clean

install:
	.venv\Scripts\pip install -r requirements.txt

test:
	.venv\Scripts\python -m pytest tests/ -v

lint:
	.venv\Scripts\ruff check src/ tests/

format:
	.venv\Scripts\ruff format src/ tests/

run:
	.venv\Scripts\streamlit run app.py

benchmark:
	.venv\Scripts\python -m scripts.benchmark

evaluate:
	.venv\Scripts\python -m scripts.evaluate_offline

clean:
	@echo Cleaning cache files...
	-@powershell -Command "Remove-Item -Path .pytest_cache, src/serving/__pycache__, tests/__pycache__ -Recurse -ErrorAction SilentlyContinue"
