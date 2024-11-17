from deep_translator import GoogleTranslator

def translate_to_english(text):
    try:
        translator = GoogleTranslator(source='auto', target='en')
        translation = translator.translate(text)
        return translation
    except Exception as e:
        print(f"Ошибка при переводе текста: {e}")
        return text
