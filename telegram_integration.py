import telegram

def send_telegram_update(actual_price, predicted_price):
    bot = telegram.Bot(token='7318185424:AAHfIWCYPdDNWx-e_m0qxElxEdhS4opZM1s')
    chat_id = '-2171247293'
    message = f"Actual price: {actual_price:.2f}\nPredicted price close: {predicted_price[0][0]:.2f}"
    bot.send_message(chat_id=chat_id, text=message)
