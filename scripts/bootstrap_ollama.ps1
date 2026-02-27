param(
  [string]$Model = "llama3.2:1b"
)

Write-Host "Checking Ollama..."
ollama --version | Out-Null
Write-Host "Pulling model $Model"
ollama pull $Model
Write-Host "Done."

