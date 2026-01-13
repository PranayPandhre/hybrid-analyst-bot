from dotenv import load_dotenv
load_dotenv()   # <-- REQUIRED

from src.llm import chat_completion

response = chat_completion(
    [{"role": "user", "content": "Reply with exactly: LLM_OK"}],
    temperature=0
)

print(response)
