from pathlib import Path
from dotenv import load_dotenv
import os

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

key = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY loaded?", bool(key), "length:", len(key) if key else None)
print("Starts with sk- ?", key[:3] if key else None)
