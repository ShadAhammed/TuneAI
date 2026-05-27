# TuneAI Dashboard launcher
# Run this from the TuneAI project root:
#   powershell -ExecutionPolicy Bypass -File start_dashboard.ps1

$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
Set-Location $PSScriptRoot
python -m streamlit run app.py --server.port 8502 --server.headless true
