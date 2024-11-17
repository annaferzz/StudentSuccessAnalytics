import subprocess
import time


def main():
    try:
        # Запуск bot.py
        bot_process = subprocess.Popen(['python', 'bot.py'])
        print("bot.py запущен")

        time.sleep(5)

        # Запуск predict.py
        predict_process = subprocess.Popen(['python', 'predict/predict.py'])
        print("predict.py запущен")

        bot_process.wait()
        predict_process.wait()

    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()
