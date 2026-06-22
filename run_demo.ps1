$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
python (Join-Path $PSScriptRoot "run_demo.py") `
  --config (Join-Path $PSScriptRoot "configs/arc_fuse.example.json") `
  --corpus (Join-Path $PSScriptRoot "examples/corpus.jsonl") `
  --graph (Join-Path $PSScriptRoot "examples/graph.jsonl") `
  --questions (Join-Path $PSScriptRoot "examples/questions.jsonl") `
  --offline `
  --output (Join-Path $PSScriptRoot "output/demo_results.jsonl")
