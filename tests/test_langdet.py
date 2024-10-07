from nlp.lang import detect_language


def test_detect_language():
    assert detect_language("Hello, world!") == "en"
    assert detect_language("Bonjour tout le monde!") == "fr"
    assert detect_language("Olá, mundo!") == "pt"
    assert detect_language("こんにちは、世界！") == "ja"
    assert detect_language("안녕하세요, 세계!") == "ko"
    assert detect_language("你好，世界！") == "zh-cn"
