from pypinyin import lazy_pinyin


class T9PinyinHanziConverter:
    @staticmethod
    def hanzi2pinyin(hanzi: str) -> str:
        return "".join(lazy_pinyin(hanzi))

    @staticmethod
    def pinyin2t9(pinyin: str) -> str:
        t9_map = {
            'a': '2', 'b': '2', 'c': '2',
            'd': '3', 'e': '3', 'f': '3',
            'g': '4', 'h': '4', 'i': '4',
            'j': '5', 'k': '5', 'l': '5',
            'm': '6', 'n': '6', 'o': '6',
            'p': '7', 'q': '7', 'r': '7', 's': '7',
            't': '8', 'u': '8', 'v': '8',
            'w': '9', 'x': '9', 'y': '9', 'z': '9',
        }
        return ''.join(t9_map.get(char, '') for char in pinyin.lower() if char.isalpha())

    @staticmethod
    def hanzi2t9(hanzi: str) -> str:
        pinyin = T9PinyinHanziConverter.hanzi2pinyin(hanzi)
        t9 = T9PinyinHanziConverter.pinyin2t9(pinyin)
        return t9
