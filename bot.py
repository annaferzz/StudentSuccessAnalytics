import telebot
import vk_api
import time
import threading
import re
from preprocessing import preprocessing_post
import subprocess
import asyncio
from preprocessing.translator import translate_to_english
import os

telegram_token = os.environ.get("TELEGRAM_TOKEN")
access_token = os.environ.get("ACCESS_TOKEN")

if not telegram_token or not access_token:
    raise EnvironmentError("TELEGRAM_TOKEN or ACCESS_TOKEN not set in environment variables")

bot = telebot.TeleBot(telegram_token)
users_in_process = {}


def get_user_groups(user_id, token):
    vk_session = vk_api.VkApi(token=token)
    try:
        groups_info = vk_session.method('groups.get',
                                        {'user_id': user_id, 'extended': 1, 'fields': 'name, type, date, '
                                                                                      'members_count, activity'})
        if groups_info and 'items' in groups_info:
            return groups_info['items']

        else:
            return []
    except vk_api.exceptions.ApiError as e:
        print(f"Ошибка при получении информации о группах пользователя: {e}")


def get_user_data(user_id, token):
    vk_session = vk_api.VkApi(token=token)
    try:
        user_info = vk_session.method('users.get', {'user_ids': user_id,
                                                    'fields': 'first_name,last_name,is_closed,bdate,schools,id,city,'
                                                              'universities,last_seen,counters'})
        if user_info:
            return user_info[0]
    except vk_api.exceptions.ApiError as e:
        if 'This profile is private' in str(e):
            return {'is_closed': True}
        else:
            print(f"Ошибка при получении информации о пользователе: {e}")
            return None


def get_user_posts(user_id, token):
    vk_session = vk_api.VkApi(token=token)
    try:
        posts = vk_session.method('wall.get', {'owner_id': user_id, 'count': 200})
        if posts and 'items' in posts:
            return [post['text'] for post in posts['items'] if 'text' in post]
        else:
            print("Нет постов у пользователя.")
            return []
    except vk_api.exceptions.ApiError as e:
        print(f"Ошибка при получении постов пользователя: {e}")
        return []


def process_posts(posts):
    preprocessed_posts = [preprocessing_post.lemmatize(post) for post in posts if preprocessing_post.lemmatize(post)]
    translated_posts = [translate_to_english(post) for post in preprocessed_posts]
    return translated_posts


def user_get(message, user_id):
    wait_message = bot.send_message(message.chat.id,
                                    "Анализ может занять несколько минут. Пожалуйста, подождите.")
    user_data = get_user_data(user_id, access_token)
    if not user_data:
        bot.send_message(message.chat.id, f"Пользователь с ID {user_id} не найден.")
        return

    if user_data.get('is_closed'):
        markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        item_open = telebot.types.KeyboardButton('Теперь профиль открыт, начать анализ')
        item_back = telebot.types.KeyboardButton('Ввести другой профиль')
        markup.add(item_open, item_back)
        bot.send_message(message.chat.id, 'Этот профиль закрыт, невозможно провести анализ', reply_markup=markup)
        return

    user_name = [f"{user_data['first_name']} {user_data['last_name']}"]

    groups_data = get_user_groups(user_data['id'], access_token)

    async def send_messages(chat_id, user_name, full_response, personality_predictions):
        message_text = (
            f"Анализ профиля пользователя {' '.join(user_name)}\n"
            f"Характеристики личности:\n{personality_predictions}"
        )
        bot.send_message(chat_id, message_text)

    async def run_predict():
        try:
            process = await asyncio.create_subprocess_exec('python', 'predict/predict.py', stdout=subprocess.PIPE,
                                                           stderr=subprocess.PIPE)
            stdout, stderr = await process.communicate()
            return stdout.decode()
        except Exception as e:
            return f"Ошибка при запуске predict.py: {e}"

    async def main(message, user_id, user_name, full_response):
        personality_predictions = await run_predict()
        bot.delete_message(message.chat.id, wait_message.message_id)
        await send_messages(message.chat.id, user_name, full_response, personality_predictions)

    posts = get_user_posts(user_id, access_token)
    if posts:
        processed_posts = process_posts(posts)
        with open('predict/processed_posts.txt', 'w', encoding='utf-8') as f:
            for post in processed_posts:
                f.write(post + '\n')
        # Запуск асинхронной функции main
        if groups_data:
            full_response = ''
            asyncio.run(main(message, user_id, user_name, full_response))
    else:
        bot.delete_message(message.chat.id, wait_message.message_id)
        bot.send_message(message.chat.id, f"Недостаточно информации для анализа")


@bot.message_handler(commands=['start'])
def welcome(message):
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    item_inform = telebot.types.KeyboardButton('Информация❓')
    item_analysis = telebot.types.KeyboardButton('Анализ профиля🔎')
    markup.add(item_inform, item_analysis)
    bot.send_message(message.chat.id, 'Здравствуйте. Готовы сделать анализ профиля ВКонтакте?', reply_markup=markup)


@bot.message_handler(func=lambda message: True)
def handle_text(message):
    if message.text == 'Информация❓':
        bot.send_message(message.chat.id, 'Это бот для учебного проекта. По всем вопросам обращаться @anna_ferzz')
    elif message.text == 'Анализ профиля🔎':
        bot.send_message(message.chat.id, 'Введите ID пользователя ВКонтакте:')
        bot.register_next_step_handler(message, process_vk_id_input)
    elif message.text == 'Теперь профиль открыт, начать анализ':
        user_id = users_in_process.get(message.chat.id)
        if user_id:
            user_get(message, user_id)
    elif message.text == 'Ввести другой профиль':
        bot.send_message(message.chat.id, 'Введите ID пользователя ВКонтакте:')
        bot.register_next_step_handler(message, process_vk_id_input)
    else:
        bot.send_message(message.chat.id, 'Извините, я не понял Ваш запрос. Выберите один из вариантов на клавиатуре.')


def validate_id(user_id):
    if re.match("^[a-zA-Z0-9_.]+$", user_id):
        return False
    return True


def process_vk_id_input(message):
    user_id = message.text
    if validate_id(user_id):
        bot.send_message(message.chat.id, "Ошибка: Неверный формат ID профиля. Пожалуйста, введите корректный ID.")
        bot.register_next_step_handler(message, process_vk_id_input)
        return
    else:
        users_in_process[message.chat.id] = user_id
        threading.Thread(target=user_get, args=(message, user_id)).start()


while True:
    try:
        bot.polling(none_stop=True)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)
