.PHONY: up down logs test lint test-vllm test-triton test-triton-trt load-test load-test-triton load-test-triton-trt up-vllm up-triton up-triton-trt compile-triton-trt

COMPOSE_FILE := deploy/docker-compose.yml
PROFILE ?= vllm

up:
	docker compose -f $(COMPOSE_FILE) --profile $(PROFILE) up -d

up-vllm:
	docker compose -f $(COMPOSE_FILE) --profile vllm up -d

up-triton:
	docker compose -f $(COMPOSE_FILE) --profile triton up -d

up-triton-trt:
	docker compose -f $(COMPOSE_FILE) --profile triton-trt up -d

down:
	docker compose -f $(COMPOSE_FILE) --profile vllm --profile triton --profile triton-trt down

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

load-test-triton:
	./scripts/load-test-triton.sh $(CONCURRENCY)

compile-triton-trt:
	./scripts/compile-triton-trt.sh $(MODEL_NAME) $(MAX_BATCH_SIZE) $(MAX_INPUT_LEN) $(MAX_OUTPUT_LEN)

test-triton-trt:
	./scripts/test-triton-trt.sh

load-test-triton-trt:
	./scripts/load-test-triton-trt.sh $(CONCURRENCY)