from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from config import BOT_TOKEN
from handlers.data_handlers import register_data_handlers
from aiogram.fsm.storage.memory import MemoryStorage
from handlers import data_handlers
import asyncio
import logging
import os
from datetime import datetime

# Создаём папку logs, если нет
os.makedirs('logs', exist_ok=True)
# Уникальное имя файла для сессии
log_filename = f"logs/bot_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
    
async def main():
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    bot = Bot(token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await data_handlers.register_data_handlers(dp)
    # await ml_handlers.register_ml_handlers(dp)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Если событийный цикл уже работает, используем `ensure_future` для планирования задачи
            asyncio.ensure_future(main())
        else:
            # Если событийный цикл не работает, запускаем его
            asyncio.run(main())
    except RuntimeError as e:
        print(f"Error: {e}")
    
    