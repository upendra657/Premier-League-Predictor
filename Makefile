.PHONY: install data features train backtest report test lint all serve docker

install:
	pip install -r requirements-dev.txt

data:
	python -m src.data

features:
	python -m src.features

train:
	python -m src.train

backtest:
	python -m src.backtest

report:
	python -m src.report && python -m src.plots

test:
	pytest -q

lint:
	ruff check src/ tests/ --line-length 100

all: data features train backtest report test lint

serve:
	uvicorn src.main:app --reload

docker:
	docker build -t plde:2.0 . && docker run -p 8000:8000 plde:2.0
