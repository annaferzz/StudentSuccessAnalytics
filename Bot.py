import telebot
import vk_api
import time
import threading
import re
import g4f
from g4f import Provider


from Config import telegram_token, access_token

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

    if groups_data:
        groups_list = [f"{group['activity']}" for group in groups_data]
        with open('prompt.txt', 'r', encoding='utf-8') as file:
            prompt_text = file.read()
        gpt_input_text = prompt_text.format(groups_list=groups_list)
        try:
            response = g4f.ChatCompletion.create(
                model=g4f.models.default,
                provider=g4f.Provider.Liaobots,
                messages=[{"role": "user", "content": gpt_input_text}],
                stream=False,
            )

            full_response = ''
            for msg in response:
                full_response += msg
            bot.delete_message(message.chat.id, wait_message.message_id)
            bot.send_message(message.chat.id, f"Анализ профиля пользователя {' '.join(user_name)}\n{full_response}")
        except Exception as e:
            bot.send_message(message.chat.id, f"Error: {e}")


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
