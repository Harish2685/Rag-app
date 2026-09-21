"""
List all available models from NVIDIA API
"""
import os
from dotenv import load_dotenv
import requests

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    print("❌ No NVIDIA_API_KEY found in .env")
    exit(1)

print("Checking available models from NVIDIA API...\n")

headers = {"Authorization": f"Bearer {api_key}"}

# Try the models endpoint
try:
    response = requests.get(
        "https://integrate.api.nvidia.com/v1/models",
        headers=headers,
        timeout=5
    )
    print(f"Status: {response.status_code}")
    print(f"Response:\n{response.text}\n")
except Exception as e:
    print(f"Error: {e}\n")

# Also try with chat completions endpoint
print("=" * 60)
print("Testing current popular models:\n")

models_to_test = [
    "nvidia/llama-3.3-70b-instruct",
    "nvidia/llama-3.2-90b-vision-instruct",
    "nvidia/llama-3.2-70b-instruct",
    "nvidia/llama-3.2-1b-instruct",
    "nvidia/mistral-nemo",
    "meta/llama-3.1-405b-instruct",
    "google/gemma-2-27b-it",
    "qwen/qwen2-72b-instruct",
]

from langchain_nvidia_ai_endpoints import ChatNVIDIA

for model in models_to_test:
    try:
        print(f"Testing: {model}...", end=" ")
        llm = ChatNVIDIA(model=model, api_key=api_key, temperature=0.0)
        response = llm.invoke("Hi")
        print(f"✓ WORKS!\n   Response: {response[:100]}")
        print(f"\n✅ Use this model in config.py:")
        print(f"   LLM_MODEL = \"{model}\"\n")
        break
    except Exception as e:
        error_msg = str(e)
        if "410" in error_msg:
            print("❌ End of life")
        elif "404" in error_msg:
            print("❌ Not found")
        else:
            print(f"❌ {error_msg[:50]}")