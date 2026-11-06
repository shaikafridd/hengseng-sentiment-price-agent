# Makefile for HengSeng Sentiment Price Prediction Agent
# Candidate: Shaik Afrid

.PHONY: demo test all ingest sentiment features train report clean help

help:
	@echo "Available commands:"
	@echo "  make demo        - Run complete offline demo pipeline (no API keys required)"
	@echo "  make test        - Run complete pytest validation suite"
	@echo "  make all         - Run complete live end-to-end pipeline with APIs"
	@echo "  make ingest      - Ingest market prices, social posts, and Exa news"
	@echo "  make sentiment   - Run multilingual extraction & sentiment scoring"
	@echo "  make features    - Build zero-leakage technical and sentiment feature matrix"
	@echo "  make train       - Execute walk-forward cross validation and ablation"
	@echo "  make report      - Generate interactive HTML & PDF reports"
	@echo "  make clean       - Remove temporary artifacts and python caches"

demo:
	python run.py demo

test:
	python -m pytest -v

all:
	python run.py all

ingest:
	python run.py ingest

sentiment:
	python run.py sentiment

features:
	python run.py features

train:
	python run.py train

report:
	python run.py report

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
