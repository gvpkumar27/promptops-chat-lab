#!/usr/bin/env sh
set -eu
MODEL="${1:-llama3.2:1b}"
ollama pull "$MODEL"
echo "Done"

