.PHONY: help install data-check bronze silver gold ml pipeline clean logs-rotate

help:
	@echo "Available targets:"
	@echo "  install       — install Python dependencies"
	@echo "  data-check    — verify raw CSV files exist in data/raw/"
	@echo "  bronze        — run bronze layer only"
	@echo "  silver        — run silver layer only"
	@echo "  gold          — run gold layer only"
	@echo "  ml            — run ML training only"
	@echo "  pipeline      — run full pipeline (bronze → silver → gold → ml)"
	@echo "  docker-build  — build Docker image"
	@echo "  docker-up     — start app + MLflow containers"
	@echo "  docker-down   — stop containers"
	@echo "  docker-logs   — tail container logs"
	@echo "  clean         — remove all data (raw CSV + delta tables)"
	@echo "  clean-logs    — remove log files"
	@echo "  mlflow        — open MLflow UI in browser"
	@echo "  eda           — open Jupyter notebook"

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

data-check:
	@if [ ! -f data/raw/flight_data_2018_2024.csv ] && [ -z "$$(ls data/raw/flights_20*.csv 2>/dev/null)" ]; then \
		echo "ERROR: Place flight_data_2018_2024.csv in data/raw/"; \
		echo "Download: https://www.kaggle.com/code/peymanradmanesh/flight-delay-analysis-2018-2024"; \
		exit 1; \
	fi

# ── Layers ────────────────────────────────────────────────────────────────────

bronze:
	python -m src.bronze

silver:
	python -m src.silver

gold:
	python -m src.gold

ml:
	python -m src.ml

delta-utils:
	python -m src.delta_utils

pipeline: data-check
	python src/pipeline.py

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker-compose build

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -rf data/raw/*.csv
	rm -rf data/delta/bronze data/delta/silver data/delta/gold
	rm -rf mlflow/

clean-logs:
	rm -f logs/*.log logs/*.png

# ── Utilities ─────────────────────────────────────────────────────────────────

mlflow:
	@echo "MLflow UI: http://localhost:5000"
	open http://localhost:5000

eda:
	jupyter notebook notebooks/eda.ipynb
