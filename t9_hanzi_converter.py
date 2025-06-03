from pypinyin import lazy_pinyin


class T9PinyinHanziConverter:
    @staticmethod
    def hanzi2pinyin(hanzi: str) -> list[str]:
        """
        Convert a Chinese string into its corresponding pinyin sequence without tone marks.

        Args:
            hanzi: A string of Chinese characters.

        Returns:
            A string of pinyin syllables separated by spaces.
        """
        return lazy_pinyin(hanzi)

    @staticmethod
    def pinyin2t9(pinyin: str) -> str:
        """
        Convert a string of pinyin syllables into a T9 digit string.

        Args:
            pinyin: A string of pinyin syllables (e.g., 'ni hao').

        Returns:
            A string of digits representing the T9 encoding (e.g., '64426').
        """
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
