# llm-observability-pack

A ready-to-run observability stack for LLM inference servers. Supports both **vLLM** and **NVIDIA Triton** backends with a single command switch. Includes Prometheus metrics collection, pre-configured Grafana dashboards, and load testing tools to generate traffic and visualize key LLM performance indicators.

Features:
- **Dual backend support** — switch between vLLM and Triton using Docker Compose profiles
- **Pre-provisioned dashboards** — Grafana dashboards auto-load on startup
- **Load testing tools** — generate realistic traffic to populate metrics
- **GPU monitoring** — track utilization, memory, and inference throughput

## Stack

- **vLLM** — OpenAI-compatible inference server with Prometheus metrics
- **Triton** — NVIDIA Triton Inference Server with vLLM backend
- **Prometheus** — scrapes and stores metrics from inference servers
- **Grafana** — pre-provisioned dashboards for monitoring LLM performance

## Key Indicators

- **TTFT** — time to first token (streaming responsiveness)
- **Inter-token latency** — token generation cadence
- **Throughput** — requests/sec and tokens/sec
- **KV Cache usage** — memory pressure indicator

## SLOs

- Availability: 99.9% successful responses
- Responsiveness: p95 TTFT ≤ 2.0s for small prompts
- Token cadence: p95 inter-token latency ≤ 150ms

## Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA driver 580+ (for CUDA 13.0 support)
- GPU with compute capability >= 7.0 (RTX 20xx or newer)

Verify with:
```bash
nvidia-smi  # should show Driver 580+ and CUDA 13.0
```

## Quick Start

```bash
cp deploy/env.example deploy/.env  # configure model and settings
make up-vllm                       # start vLLM stack
# or
make up-triton                     # start Triton stack

make logs                          # view logs
make down                          # stop services
```

## Services

### vLLM Stack
| Service | URL |
|---------|-----|
| vLLM API | http://localhost:8000 |
| vLLM docs | http://localhost:8000/docs |
| vLLM metrics | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

### Triton Stack
| Service | URL |
|---------|-----|
| Triton HTTP API | http://localhost:8000 |
| Triton gRPC API | http://localhost:8001 |
| Triton metrics | http://localhost:8002/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

## Testing

Test API with sample requests:
```bash
make test-vllm    # test vLLM API
make test-triton  # test Triton API
```

## Load Testing

Generate continuous traffic to populate dashboards:

### vLLM
```bash
make load-test                  # default: 3 concurrent workers
make load-test CONCURRENCY=5    # custom concurrency
./scripts/load-test.sh 5        # or run directly
```

### Triton
```bash
make load-test-triton                  # default: 3 concurrent workers
make load-test-triton CONCURRENCY=5    # custom concurrency
./scripts/load-test-triton.sh 5        # or run directly
```

Press `Ctrl+C` to stop.

## Grafana Dashboards

Access Grafana at http://localhost:3000 (login: admin/admin).

### vLLM Overview
- Request rate (success/failure)
- Token throughput (generation + prompt tokens/s)
- Time to First Token (TTFT) — p50/p95/p99
- End-to-end request latency — p50/p95/p99
- Running/waiting requests
- KV cache usage

### Triton Overview
- Inference request rate (success/failure)
- Inference throughput
- GPU utilization
- GPU memory used
- Total requests
- Pending requests

## Prometheus Integration

Verify Prometheus is scraping metrics:

1. **Check targets** — http://localhost:9090/targets
   - `vllm` or `triton` target should show as "UP"

2. **Query metrics**:
   ```bash
   # vLLM metrics
   curl -s "http://localhost:9090/api/v1/label/__name__/values" | grep vllm

   # Triton metrics
   curl -s "http://localhost:9090/api/v1/label/__name__/values" | grep nv_
   ```

3. **Example vLLM metrics** (note the colon in metric names):
   - `vllm:request_success_total`
   - `vllm:prompt_tokens_total`
   - `vllm:generation_tokens_total`
   - `vllm:time_to_first_token_seconds`
   - `vllm:kv_cache_usage_perc`

4. **Example Triton metrics**:
   - `nv_inference_request_success`
   - `nv_inference_request_failure`
   - `nv_inference_count`
   - `nv_inference_pending_request_count`
   - `nv_gpu_utilization`
   - `nv_gpu_memory_used_bytes`