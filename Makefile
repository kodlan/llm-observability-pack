.PHONY: up down logs test lint test-vllm test-triton load-test up-vllm up-triton

COMPOSE_FILE := deploy/docker-compose.yml
PROFILE ?= vllm

up:
	docker compose -f $(COMPOSE_FILE) --profile $(PROFILE) up -d

up-vllm:
	docker compose -f $(COMPOSE_FILE) --profile vllm up -d

up-triton:
	docker compose -f $(COMPOSE_FILE) --profile triton up -d

down:
	docker compose -f $(COMPOSE_FILE) --profile vllm --profile triton down

logs:
	docker compose -f $(COMPOSE_FILE) logs -f

test:
	@echo "No tests yet"

lint:
	@echo "No lint yet"

test-vllm:
	./scripts/test-vllm.sh

test-triton:
	./scripts/test-triton.sh

load-test:
	./scripts/load-test.sh $(CONCURRENCY)