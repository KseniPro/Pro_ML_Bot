import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
if not BOT_TOKEN:
    raise ValueError("Не найден BOT_TOKEN в .env файле")

ADMINS = [1429293979]  # Ваш ID в Telegram