"""
Panjayet API — FastAPI app entry point.
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from routers.session import router as session_router
from routers.chat import router as chat_router

import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

class SecureLogFilter(logging.Filter):
    """Intercepts and masks sensitive query parameters in log output."""
    def __init__(self):
        super().__init__()
        self.pattern = re.compile(r"([?&]key=)([^&\s'\"]+)")

    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = self.pattern.sub(r"\1***MASKED***", record.msg)
        if record.args:
            masked_args = []
            for arg in record.args:
                try:
                    s_arg = str(arg)
                    if "?key=" in s_arg or "&key=" in s_arg:
                        masked_args.append(self.pattern.sub(r"\1***MASKED***", s_arg))
                    else:
                        masked_args.append(arg)
                except Exception:
                    masked_args.append(arg)
            record.args = tuple(masked_args)
        return True

secure_filter = SecureLogFilter()
logging.getLogger().addFilter(secure_filter)
logging.getLogger("httpx").addFilter(secure_filter)

for handler in logging.root.handlers:
    handler.addFilter(secure_filter)

app = FastAPI(
    title="Panjayet API",
    description=(
        "Multi-Agent Adversarial Research Platform. "
        "Zero database. Zero auth. You bring the keys."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    root_path="/panjayet",
)

# ── CORS ────────────────────────────────────────────────────────────────────
# Unified container policy: wildcard origins with credentials=False.
# This matches the hoptrace and pragma CORS configuration and satisfies
# the browser spec (allow_credentials cannot be True with allow_origins=["*"]).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Cannot be True when allow_origins=["*"] — browser spec violation
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session_router)
app.include_router(chat_router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Panjayet API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
