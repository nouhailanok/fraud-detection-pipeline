Set-Location 'C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline'
$env:PYTHONPATH = $PWD.Path
$env:CLIENT_ID = "3"
$env:FLOWER_SERVER_HOST = "100.71.140.30"
$env:FLOWER_SERVER_PORT = "8090"
$env:FL_CLIENT_CONTINUOUS = "true"
$env:DP_NOISE = "1.0"
$env:DP_EPSILON_TARGET = "1.0"
$env:FL_LR = "0.0005"
$env:FL_BATCH_SIZE = "256"
$env:FL_POS_WEIGHT = "100.0"
Write-Output "BANQUE 3 - Node 3"
& 'C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe' federated/client.py
Write-Output "Client 3 termine"
Read-Host "Appuyer sur Entree"
