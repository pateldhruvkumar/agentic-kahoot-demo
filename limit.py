import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("Error: OPENROUTER_API_KEY environment variable is not set.")
    print("Please set it in your .env file or environment variables.")
    exit(1)

response = requests.get(
  url="https://openrouter.ai/api/v1/auth/key",
  headers={
    "Authorization": f"Bearer {api_key}"
  }
)
print(json.dumps(response.json(), indent=2))