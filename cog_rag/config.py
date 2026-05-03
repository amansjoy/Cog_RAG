"""Centralised configuration.

All env-var reads happen here. If you change deployment names or auth, this
is the only file you edit.

The .env file (in the project root) is loaded automatically if python-dotenv
is installed. If you prefer PyCharm's run-config "Environment variables"
field, that also works — dotenv will simply not find a file and skip silently.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env from project root if present
except ImportError:  # python-dotenv is optional
    pass

from openai import AzureOpenAI


@dataclass(frozen=True)
class Settings:
    azure_endpoint: str
    api_key: str
    api_version: str
    chat_deployment: str
    embed_deployment: str

    @classmethod
    def from_env(cls) -> "Settings":
        missing = [
            name for name in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY")
            if not os.environ.get(name)
        ]
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {missing}. "
                "Set them in a .env file at the project root, or in your "
                "PyCharm run configuration's Environment variables field."
            )
        return cls(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            chat_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
            embed_deployment=os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large"),
        )


def make_client(settings: Settings | None = None) -> AzureOpenAI:
    """Return a configured AzureOpenAI client.

    Pass a Settings object for tests, or call with no args to read from env.
    """
    s = settings or Settings.from_env()
    return AzureOpenAI(
        azure_endpoint=s.azure_endpoint,
        api_key=s.api_key,
        api_version=s.api_version,
    )


# Convenience module-level constants for the rest of the package.
# These read from env on import; if you need test-time overrides, instantiate
# Settings explicitly and pass it through.
_settings = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
