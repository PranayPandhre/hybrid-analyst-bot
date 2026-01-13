from dotenv import load_dotenv
load_dotenv()

from src.router import route_query

tests = [
    "What is the market cap of Tesla?",
    "Compare the revenue of Apple and Microsoft.",
    "What are the AI initiatives mentioned by Microsoft?",
    "What are the headwinds facing Apple's growth?",
]

for t in tests:
    print(t)
    print(route_query(t))
    print()
