Set-Location "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"
$env:PYTHONPATH                     = $PWD.Path
$env:FL_ROUNDS                      = "10"
$env:FL_MIN_CLIENTS                 = "2"
$env:FL_LOCAL_EPOCHS                = "5"
$env:FLOWER_PORT                    = "8080"
$env:FLOWER_TLS_REQUIRE_CLIENT_CERT = "false"
Write-Output "=== SERVEUR FL (Python 3.12) ==="
& "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" federated/server.py
