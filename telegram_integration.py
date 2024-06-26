import telegram
import os

def send_telegram_update(actual_price, predicted_price):
    bot = telegram.Bot(token=os.environ.get('TOKEN'))
    chat_id = '-2171247293'
    message = f"Actual price: {actual_price:.2f}\nPredicted price close: {predicted_price[0][0]:.2f}"
    bot.send_message(chat_id=chat_id, text=message)
