Set-Location "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"
$env:PYTHONPATH           = $PWD.Path
$env:CLIENT_ID            = "2"
$env:FLOWER_SERVER_HOST   = "127.0.0.1"
$env:FLOWER_SERVER_PORT   = "8080"
$env:FL_CLIENT_CONTINUOUS = "true"
$env:DP_NOISE             = "1.5"
$env:FL_LR                = "0.0005"
$env:FL_BATCH_SIZE        = "256"
$env:FL_POS_WEIGHT        = "167.0"
Write-Output "=== BANQUE 2 (Python 3.12 + GPU) ==="
& "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" federated/client.py
