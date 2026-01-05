# llm-observability-pack

A reusable observability kit for LLM text-generation services. Provides SLOs, error budgets, burn-rate alerting, Grafana dashboards, and runbooks for vLLM and NVIDIA Triton Inference Server.

## Stack

- **Prometheus** — scrapes metrics from model servers and the derived-metrics adapter
- **Grafana** — pre-provisioned dashboards for SLOs, latency, throughput, and saturation

## Key Indicators

- **TTFT** — time to first token (streaming responsiveness)
- **Inter-token latency** — token generation cadence
- **Throughput** — requests/sec and tokens/sec
- **Availability** — successful response rate

## SLOs

- Availability: 99.9% successful responses
- Responsiveness: p95 TTFT ≤ 2.0s for small prompts
- Token cadence: p95 inter-token latency ≤ 150ms

## Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA driver 535+ (for CUDA 12.x support)
- GPU with compute capability >= 7.0 (RTX 20xx or newer)

Verify with:
```bash
nvidia-smi  # should show Driver 535+ and CUDA 12.x
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
