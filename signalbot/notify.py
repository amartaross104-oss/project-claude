"""Optional Telegram delivery. Credentials come from environment variables only."""
from __future__ import annotations

import os

import requests

API = "https://api.telegram.org/bot{token}/{method}"


def _token() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN (get it from @BotFather)")
    return token


def send_telegram(text: str) -> None:
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not chat:
        raise RuntimeError("Set TELEGRAM_CHAT_ID (run: python -m signalbot chatid)")
    r = requests.post(API.format(token=_token(), method="sendMessage"),
                      json={"chat_id": chat, "text": text}, timeout=15)
    r.raise_for_status()


def find_chat_ids() -> list[tuple[int, str]]:
    """Chats that recently messaged the bot (send it any message first, e.g. /start)."""
    r = requests.get(API.format(token=_token(), method="getUpdates"), timeout=15)
    r.raise_for_status()
    seen: dict[int, str] = {}
    for u in r.json().get("result", []):
        msg = u.get("message") or u.get("channel_post") or {}
        chat = msg.get("chat")
        if chat:
            seen[chat["id"]] = chat.get("title") or chat.get("username") or chat.get("first_name", "")
    return list(seen.items())
