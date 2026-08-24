.PHONY: install quality train all mlflow

install:
	python -m pip install -e '.[dev]'

quality:
	ruff check .
	pytest -q

train:
	credit-risk train

all:
	credit-risk all

mlflow:
	python -m pip install -e '.[mlops]'
	mlflow ui --backend-store-uri ./mlruns
