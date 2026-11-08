# Makefile - short commands for every step of the project.
# Usage: make <target>     e.g.  make train-all   make test   make api
# Change the API port if 8000 is busy:   make api PORT=8001

PY      ?= .venv/bin/python
PORT    ?= 8000
TAG     ?= model-v1
BUCKET  ?= YOUR-BUCKET
REGION  ?= us-east-1
PROFILE ?= dvc

.PHONY: help setup train train-all promote export predict test api mlflow-ui \
        dvc-remote dvc-push dvc-pull docker-build docker-run docker-stop \
        k8s-up k8s-down prometheus prometheus-stop traffic-normal traffic-drift drift clean

help:            ## Show all commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

setup:           ## Create .venv and install all libraries
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

# ---- Lab 1: training + MLflow ----
train:           ## Train the default (linear) model
	$(PY) train.py

train-all:       ## Train linear, tree and forest (compare them in MLflow)
	$(PY) train.py linear
	$(PY) train.py tree
	$(PY) train.py forest

promote:         ## Give the 'champion' alias to the lowest-MAE version and export it
	$(PY) export_model.py --promote

export:          ## Export the current champion to champion.pkl
	$(PY) export_model.py

predict:         ## Price one house with model.pkl
	$(PY) predict.py

mlflow-ui:       ## Open the MLflow dashboard on http://localhost:5000
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# ---- Lab 2: DVC + S3 ----
dvc-remote:      ## Connect DVC to S3:  make dvc-remote BUCKET=my-bucket REGION=us-east-1
	$(PY) -m dvc remote add -d -f storage s3://$(BUCKET)/dvc
	$(PY) -m dvc remote modify storage region $(REGION)
	$(PY) -m dvc remote modify --local storage profile $(PROFILE)

dvc-push:        ## Upload data to S3 (always BEFORE git push)
	$(PY) -m dvc push

dvc-pull:        ## Download data from S3
	$(PY) -m dvc pull

# ---- Lab 2: tests, API, Docker ----
test:            ## Run all tests (needs metrics.json, model.pkl, champion.pkl)
	$(PY) -m pytest -v

api:             ## Run the API on http://127.0.0.1:$(PORT)/docs
	$(PY) -m uvicorn app:app --reload --host 0.0.0.0 --port $(PORT)

docker-build:    ## Build the image house-api:$(TAG)
	docker build -t house-api:$(TAG) .

docker-run:      ## Run the image on port $(PORT)
	docker run -d --name house-api -p $(PORT):8000 house-api:$(TAG)

docker-stop:     ## Stop and remove the container
	docker rm -f house-api

# ---- Lab 3: Kubernetes ----
k8s-up:          ## Deploy to the LOCAL docker-desktop cluster
	kubectl config use-context docker-desktop
	kubectl apply -f k8s/
	kubectl rollout status deployment/house-api

k8s-down:        ## Remove the deployment and service
	kubectl delete -f k8s/

# ---- Lab 4: monitoring + drift ----
prometheus:      ## Start Prometheus on http://localhost:9090
	docker run -d --name prometheus -p 9090:9090 -v "$(PWD)/prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus:v3.0.0

prometheus-stop: ## Stop Prometheus
	docker rm -f prometheus

traffic-normal:  ## Send 50 normal requests to the API
	$(PY) send_traffic.py normal http://127.0.0.1:$(PORT)/predict

traffic-drift:   ## Send 50 luxury-house requests to the API
	$(PY) send_traffic.py drift http://127.0.0.1:$(PORT)/predict

drift:           ## Check for data drift (exit code 1 = drift)
	$(PY) check_drift.py

clean:           ## Delete everything the scripts re-create
	rm -rf mlruns mlartifacts mlflow.db model.pkl champion.pkl metrics.json predictions.csv .pytest_cache
