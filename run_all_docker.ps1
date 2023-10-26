#Remove-Item datasets/* -r
#Remove-Item outputs/* -r

Write-Output "--------------------"
Write-Output "Emotion-classification-audio"
Write-Output "--------------------"
docker compose run --rm emotion-classification-audio


Write-Output "--------------------"
Write-Output "Fusion-layer"
Write-Output "--------------------"
docker compose run --rm fusion-layer


Write-Output "--------------------"
Write-Output "Ed-evaluation"
Write-Output "--------------------"
docker compose run --rm ed-evaluation
