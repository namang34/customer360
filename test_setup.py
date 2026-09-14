from dotenv import load_dotenv
load_dotenv()

import os, redis
from langchain_google_genai import ChatGoogleGenerativeAI

# Test Gemini
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
response = llm.invoke("say ok")
print("Gemini:", response.content)

# Test Redis
r = redis.Redis(host="localhost", port=6379)
r.set("test", "working")
print("Redis:", r.get("test"))