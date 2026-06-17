PYTHON = .venv/Scripts/python.exe

lint:
	$(PYTHON) -m ruff check src/ tests/

test:
	$(PYTHON) -m pytest tests/ -v

train-baselines:
	$(PYTHON) -m src.models.baselines

train-mlp:
	$(PYTHON) -m src.models.mlp

run-api:
	$(PYTHON) -m uvicorn src.api.main:app --reload

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

all: lint test
