from nlp.translate import TranslatorWrapper

def test_translate():
    wrapper = TranslatorWrapper()
    chunk = "Телекомплект-Оптик ООД работа, свободни позиции и заплати -ID: [122052] - БУЛСТАТ: 201395716, — Zaplata.bg"
    lang = "bg"
    translated_chunk = wrapper.translate(chunk, lang, method="api")
    translated_chunk = wrapper.translate(chunk, lang, method="offline") 
    assert translated_chunk == "Telekomplekt-Optik OOD jobs, vacancies and wages -ID: [122052] - BULSTAT: 210118898, — Zaplata.bg"
