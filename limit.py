import requests
import json
response = requests.get(
  url="https://openrouter.ai/api/v1/auth/key",
  headers={
    "Authorization": f"Bearer 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"
  }
)
print(json.dumps(response.json(), indent=2))