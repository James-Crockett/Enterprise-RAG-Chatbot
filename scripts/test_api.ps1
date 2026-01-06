$base = "http://127.0.0.1:8000"

Write-Host "Health..."
Invoke-RestMethod "$base/health" | Format-List

Write-Host "Login..."
$body = @{
  email = "internal@demo.com"
  password = "internal123"
} | ConvertTo-Json

try {
  $resp = Invoke-RestMethod -Method Post -Uri "$base/auth/login" -ContentType "application/json" -Body $body
  $resp | Format-List
} catch {
  Write-Host "Login failed:"
  $_.Exception.Response
}
