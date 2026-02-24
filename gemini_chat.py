import http.client
import json

conn = http.client.HTTPSConnection("")
payload = json.dumps({
   "contents": [
      {
         "parts": [
            {
               "text": "Tell me about this instrument"
            },
            {
               "inline_data": {
                  "mime_type": "image/jpeg",
                  "data": "<这里填你图片的base64编码>"
               }
            }
         ]
      }
   ]
})
headers = {
   'Authorization': 'Bearer <token>',
   'Content-Type': 'application/json'
}
conn.request("POST", "/v1beta/models/:generateContent", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))