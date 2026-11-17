import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent))
from rag import answer

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
LOG = Path("logs/queries.jsonl")
LOG.parent.mkdir(parents=True, exist_ok=True)

bot = Bot(TOKEN)
dp = Dispatcher()

def _reply(q):
    r = answer(q)
    src = r["source"] or "—"
    text = r["text"] + chr(10) + chr(10) + "Источник: " + src
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "q": q, "score": r["score"], "source": src}, ensure_ascii=False) + chr(10))
    return text[:4000]

@dp.message(Command("start"))
async def start(m: Message):
    await m.answer("Привет! Отправь /ask <вопрос> или просто напиши вопрос по Python Tutorial.")

@dp.message(Command("help"))
async def help_cmd(m: Message):
    await m.answer("Использование: /ask Как работают f-strings? — или просто напиши вопрос.")

@dp.message(Command("ask"))
async def ask(m: Message):
    q = (m.text or "").partition(" ")[2].strip()
    if not q:
        await m.answer("Формат: /ask <вопрос>")
        return
    await m.answer(_reply(q))

@dp.message()
async def any_text(m: Message):
    q = (m.text or "").strip()
    if not q:
        return
    await m.answer(_reply(q))

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
