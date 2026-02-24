import http.client
import json

conn = http.client.HTTPSConnection("")
payload = json.dumps({
   "name": "My API Token",
   "expired_time": 1790995200,
   "remain_quota": 1000000,
   "unlimited_quota": False,
   "model_limits_enabled": True,
   "model_limits": "gemini-2.5-flash,gpt-5.2",
   "allow_ips": "192.168.1.1,10.0.0.1",
   "group": "default"
})
headers = {
   'Authorization': 'Bearer <token>',
   'New-Api-User': '<api-key>',
   'Content-Type': 'application/json'
}
conn.request("POST", "/api/token/", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))