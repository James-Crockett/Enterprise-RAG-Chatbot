param(
  [string]$Base = "http://127.0.0.1:8000",
  [string]$Email = "internal@demo.com",
  [string]$Password = "internal123"
)

Write-Host "1) Health..."
$health = Invoke-RestMethod -Method Get -Uri "$Base/health"
$health | Format-List

Write-Host "2) Login..."
$loginBody = @{ email=$Email; password=$Password } | ConvertTo-Json
$login = Invoke-RestMethod -Method Post -Uri "$Base/auth/login" -ContentType "application/json" -Body $loginBody
$token = $login.access_token
Write-Host "Got token."

Write-Host "3) Chat..."
$chatBody = @{
  query  = "How to setup VPN?"
  top_k  = 5
  filters = @{}
} | ConvertTo-Json -Depth 6

$headers = @{ Authorization = "Bearer $token" }

$resp = Invoke-RestMethod -Method Post -Uri "$Base/chat" -Headers $headers -ContentType "application/json" -Body $chatBody
$resp.answer
$resp.results | Select-Object chunk_id, score, title, department, source_path | Format-Table
