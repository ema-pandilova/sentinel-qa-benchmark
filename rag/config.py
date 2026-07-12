# rag/config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))

CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(DATA_DIR, "chroma"))
HNSW_SPACE = os.getenv("HNSW_SPACE", "cosine")

DEFAULT_K = int(os.getenv("DEFAULT_RETRIEVAL_K", "1"))

# default models (used if UI doesn't override)
ZEROSHOT_MODEL = os.getenv("ZEROSHOT_MODEL", "gpt-4o")
RAG_MODEL = os.getenv("RAG_MODEL", "gpt-4o")
DSPY_MODEL = os.getenv("DSPY_MODEL", "gpt-4o")

# models shown in UI dropdowns
OPENAI_MODELS = os.getenv("OPENAI_MODELS", "gpt-4o,gpt-4o-mini").split(",")
GEMINI_MODELS = os.getenv("GEMINI_MODELS", "gemini-1.5-pro,gemini-1.5-flash").split(",")
ANTHROPIC_MODELS = os.getenv("ANTHROPIC_MODELS", "claude-3-5-sonnet-20241022").split(",")
DEEPSEEK_MODELS = os.getenv("DEEPSEEK_MODELS", "deepseek-chat").split(",")
