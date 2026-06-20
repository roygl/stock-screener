"""Secrets → env bridge (relocated verbatim from app.py)."""

from __future__ import annotations

import os

import streamlit as st

from screener import agent


# --- secrets -> env bridge -----------------------------------------------
# The agent reads API keys from os.environ. When the app is launched OUTSIDE the
# interactive shell that exported the key (e.g. Streamlit Cloud, a desktop click,
# or a plain `streamlit run`), that key is invisible — so the NL search silently
# used the offline parser. Copy any provider key found in .streamlit/secrets.toml
# into os.environ ONCE, before anything reads availability, so a secrets.toml works
# regardless of launch context. Env always wins (we never overwrite a set var), and
# this is offline-first: no secrets file -> a harmless no-op (no key UI, ever).
def bridge_secrets_to_env() -> None:
    """Stage any provider key from secrets.toml into os.environ (env wins)."""
    names = {p.env_key for p in agent.PROVIDERS.values() if p.env_key} | {"GOOGLE_API_KEY"}
    for name in names:
        if os.environ.get(name):
            continue
        try:
            val = st.secrets.get(name)  # safe even when no secrets.toml exists
        except Exception:
            val = None
        if val:
            os.environ[name] = str(val)
