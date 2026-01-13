from dotenv import load_dotenv
import os

load_dotenv()

print(os.environ.get("GROQ_API_KEY") is not None)
print(os.environ.get("GROQ_BASE_URL"))
