.PHONY: up down logs test lint test-vllm load-test

COMPOSE_FILE := deploy/docker-compose.yml

up:
	docker compose -f $(COMPOSE_FILE) up -d

down:
	docker compose -f $(COMPOSE_FILE) down

logs:
	docker compose -f $(COMPOSE_FILE) logs -f

test:
	@echo "No tests yet"

lint:
	@echo "No lint yet"

test-vllm:
	./scripts/test-vllm.sh

load-test:
	./scripts/load-test.sh $(CONCURRENCY)
