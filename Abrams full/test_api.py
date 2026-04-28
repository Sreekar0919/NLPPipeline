import urllib.request, json

try:
    url = "https://aihubapi.stanfordhealthcare.org/aws-bedrock/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/invoke"
    
    hdr = {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'api-key': '8a225a762d6b4aaaa0d5855e1aeeb821',
    }
    
    # Request body in Anthropic Messages API format
    data = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France? Answer in one sentence."
            }
        ]
    }
    
    data = json.dumps(data)
    req = urllib.request.Request(url, headers=hdr, data=bytes(data.encode("utf-8")))
    req.get_method = lambda: 'POST'
    
    response = urllib.request.urlopen(req)
    print("Status:", response.getcode())
    
    # Parse and print just the answer text
    response_body = json.loads(response.read())
    answer = response_body["content"][0]["text"]
    print("Answer:", answer)

except Exception as e:
    print("Error:", e)