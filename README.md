# llm-observability-pack

A ready-to-run observability stack for LLM inference servers. Spin up vLLM with Prometheus metrics collection and Grafana dashboards using a single command. Includes load testing tools to generate traffic and visualize key LLM performance indicators.

## Stack

- **vLLM** — OpenAI-compatible inference server with Prometheus metrics
- **Prometheus** — scrapes and stores metrics from vLLM
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
make up                            # start services
make logs                          # view logs
make down                          # stop services
```

## Services

| Service | URL |
|---------|-----|
| vLLM API | http://localhost:8000 |
| vLLM docs | http://localhost:8000/docs |
| vLLM metrics | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

## Testing

Test vLLM API with sample requests:
```bash
make test-vllm
```

## Load Testing

Generate continuous traffic to populate dashboards:

```bash
# Default: 3 concurrent workers
make load-test

# Custom concurrency
make load-test CONCURRENCY=5

# Or run directly
./scripts/load-test.sh 5
```

Press `Ctrl+C` to stop.

## Grafana Dashboards

Access Grafana at http://localhost:3000 (login: admin/admin).

**vLLM Overview** dashboard includes:
- Request rate (success/failure)
- Token throughput (generation + prompt tokens/s)
- Time to First Token (TTFT) — p50/p95/p99
- End-to-end request latency — p50/p95/p99
- Running/waiting requests
- KV cache usage

## Prometheus Integration

Verify Prometheus is scraping vLLM metrics:

1. **Check targets** — http://localhost:9090/targets
   - `vllm` target should show as "UP"

2. **Query metrics**:
   ```bash
   curl -s "http://localhost:9090/api/v1/label/__name__/values" | grep vllm
   ```

3. **Example metrics** (note the colon in metric names):
   - `vllm:request_success_total`
   - `vllm:prompt_tokens_total`
   - `vllm:generation_tokens_total`
   - `vllm:time_to_first_token_seconds`
   - `vllm:kv_cache_usage_perc`