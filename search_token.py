import http.client
import json

conn = http.client.HTTPSConnection("")
payload = ''
headers = {
   'Authorization': 'Bearer <token>',
   'New-Api-User': '<api-key>',
   'Content-Type': 'application/json'
}
conn.request("GET", "/api/token/search?keyword=api&token=sk-123", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))