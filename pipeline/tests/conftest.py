import os
import sys

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

import pytest
from dotenv import load_dotenv

load_dotenv(os.path.join(PIPELINE_DIR, ".env"))

HAS_GEMINI_KEY = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

requires_gemini = pytest.mark.skipif(
    not HAS_GEMINI_KEY, reason="No GEMINI_API_KEY/GOOGLE_API_KEY set in environment or .env"
)
