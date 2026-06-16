# Banking Transaction Prediction - Makefile
.PHONY: help install test lint format docker run-api run-dashboard run-all clean

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make docker        - Build Docker image"
	@echo "  make run-api       - Start Flask API"
	@echo "  make run-dashboard - Start Streamlit dashboard"
	@echo "  make run-all       - Run complete pipeline"
	@echo "  make clean         - Clean artifacts"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ app/ --max-line-length=100
	pylint src/ --disable=C0103

format:
	black src/ app/ --line-length=100
	isort src/ app/ --profile black

docker:
	docker build -t banking-prediction:latest .
	docker-compose up -d

run-api:
	python app/flask/app.py

run-dashboard:
	streamlit run app/streamlit/app.py

run-all:
	python main_pipeline.py --phase all --data data/raw/train.csv

clean:
	rm -rf logs/*.log
	rm -rf reports/figures/*.png
	rm -rf models/artifacts/*
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
