# Raccourcis du projet : tape `make help` pour la liste.

ifeq ($(OS),Windows_NT)
PYTHON_311 := py -3.11
PY := .venv/Scripts/python.exe
else
PYTHON_311 := python3.11
PY := .venv/bin/python
endif

IMAGE := complaint-router
# `make space HF_TOKEN=hf_xxx` : le jeton doit être visible du script Python
export HF_TOKEN

.DEFAULT_GOAL := help
.PHONY: help setup download data baseline train export llm report pipeline serve docker \
	docker-run space test lint mlflow

help:
	@echo Installation
	@echo   make setup     : cree l environnement Python et installe tout
	@echo Pipeline, dans l ordre
	@echo   make download  : telecharge les reclamations, 925 Mo
	@echo   make data      : prepare les jeux train, validation et test
	@echo   make baseline  : entraine la reference TF-IDF, 1 minute
	@echo   make train     : affine le modele MiniLM en PyTorch, environ 2 h sur processeur
	@echo   make export    : exporte en ONNX, quantifie en 8 bits, mesure les temps de reponse
	@echo   make llm       : compare avec un LLM local, Ollama doit tourner
	@echo   make report    : tableau comparatif et graphique final
	@echo   make pipeline  : tout ce qui precede, sauf le LLM
	@echo Service
	@echo   make serve     : lance l API et la demo sur http://localhost:8000
	@echo   make docker    : construit l image Docker de l API
	@echo   make docker-run: lance l image sur http://localhost:8000
	@echo   make space     : publie la demo sur Hugging Face, avec HF_TOKEN=hf_xxx
	@echo Qualite
	@echo   make test      : lance les tests
	@echo   make lint      : verifie le style du code
	@echo   make mlflow    : ouvre l historique des entrainements sur http://localhost:5000

setup:
	$(PYTHON_311) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install --no-deps -e .

download:
	$(PY) -m complaint_router.download

data:
	$(PY) -m complaint_router.data

baseline:
	$(PY) -m complaint_router.baseline

train:
	$(PY) -m complaint_router.train

export:
	$(PY) -m complaint_router.export_onnx

llm:
	$(PY) -m complaint_router.llm_baseline

report:
	$(PY) -m complaint_router.report

pipeline: download data baseline train export report

serve:
	$(PY) -m uvicorn app.main:app --port 8000

docker:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)

space:
	$(PY) scripts/deploy_space.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

mlflow:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
