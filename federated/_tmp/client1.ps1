Set-Location 'C:\Users\Imane\OneDrive\Bureau\IAGI-S4\Projet_Metier\fraud-detection-pipeline'
$env:PYTHONPATH = $PWD.Path
$env:CLIENT_ID = "1"
$env:FLOWER_SERVER_HOST = "100.71.140.30"
$env:FLOWER_SERVER_PORT = "8090"
$env:FL_CLIENT_CONTINUOUS = "true"
$env:DP_NOISE = "1.0"
$env:DP_EPSILON_TARGET = "1.0"
$env:FL_LR = "0.0005"
$env:FL_BATCH_SIZE = "256"
$env:FL_POS_WEIGHT = "100.0"
Write-Output "BANQUE 1 - Node 1"
& 'C:\Users\Imane\OneDrive\Bureau\IAGI-S4\Projet_Metier\fraud-detection-pipeline\venv\Scripts\python.exe' federated/client.py
Write-Output "Client 1 termine"
Read-Host "Appuyer sur Entree"
