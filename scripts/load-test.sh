#!/bin/bash
# Simple load test for vLLM - sends continuous parallel requests
# Usage: ./load-test.sh [CONCURRENCY] [VLLM_URL]
# Stop with Ctrl+C

CONCURRENCY=${1:-3}
VLLM_URL=${2:-http://localhost:8000}

PROMPTS=(
  "Explain what Python is in 2 sentences."
  "What is machine learning? Brief answer."
  "Count from 1 to 10."
  "Say hello in 3 different languages."
  "What is 2 + 2? Just the number."
)

echo "Starting load test with $CONCURRENCY concurrent requests"
echo "Target: $VLLM_URL"
echo "Press Ctrl+C to stop"
echo ""

send_request() {
  local id=$1
  local prompt_idx=$((RANDOM % ${#PROMPTS[@]}))
  local prompt="${PROMPTS[$prompt_idx]}"

  while true; do
    start_time=$(date +%s.%N)

    response=$(curl -s -w "\n%{http_code}" "${VLLM_URL}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"Qwen/Qwen2.5-1.5B-Instruct\",
        \"messages\": [{\"role\": \"user\", \"content\": \"${prompt}\"}],
        \"max_tokens\": 50
      }" 2>/dev/null)

    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    http_code=$(echo "$response" | tail -1)

    if [ "$http_code" = "200" ]; then
      echo "[Worker $id] OK - ${duration}s"
    else
      echo "[Worker $id] ERROR $http_code"
    fi

    sleep 0.1
  done
}

# Start background workers
for i in $(seq 1 $CONCURRENCY); do
  send_request $i &
done

# Wait for Ctrl+C
trap "echo ''; echo 'Stopping...'; kill 0; exit" SIGINT
wait
