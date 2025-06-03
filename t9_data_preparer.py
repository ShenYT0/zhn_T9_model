from pathlib import Path
import re
import xml.etree.ElementTree as ET

from file_handler import FileHandler
from t9_hanzi_converter import T9PinyinHanziConverter


class T9DataPreparer:
    def __init__(self, pinyin_dir: Path = Path("./data/xml/pinyin"), char_dir: Path = Path("data/xml/character")) -> None:
        """Initialize T9DataPreparer with directories for pinyin and character XML files."""
        self.pinyin_dir = pinyin_dir
        self.char_dir = char_dir

    def extract_samples_from_file(self, pinyin_path: Path, char_path: Path) -> list[tuple[str, str, str]]:
        """Extract T9 training samples from a pair of aligned pinyin and character XML files.

        Args:
            pinyin_path: Path to the pinyin XML file.
            char_path: Path to the character XML file.

        Returns:
            A list of tuples, each containing (digit sequence, pinyin sequence, Chinese characters).
        """
        samples = []
        root_pinyin = ET.parse(pinyin_path).getroot()
        root_character = ET.parse(char_path).getroot()

        for sp, sc in zip(root_pinyin.iter('s'), root_character.iter('s')):
            pinyin_words = [w.text.strip() for w in sp.findall('w')]
            char_words = [w.text.strip() for w in sc.findall('w')]

            if len(pinyin_words) != len(char_words):
                continue
            for pinyin, hanzi in zip(pinyin_words, char_words):
                syllables = T9DataPreparer.split_polyphonic(pinyin)
                normalized = T9DataPreparer.remove_tone_numbers(T9DataPreparer.normalize_pinyin(syllables))
                pinyin_seq = ''.join(normalized)
                digit_seq = T9PinyinHanziConverter.pinyin2t9(''.join(normalized))
                if digit_seq and hanzi:
                    samples.append((digit_seq, pinyin_seq, hanzi))
        return samples

    def extract_sentences_from_file(self, filepath: Path) -> list[str]:
        """Extract full sentences from a character XML file.

        Args:
            filepath: Path to the character XML file.

        Returns:
            A list of full Chinese sentences.
        """
        sentences = []
        root = ET.parse(filepath).getroot()
        for s in root.iter('s'):
            tokens = []
            for elem in s:
                if elem.tag in ('w', 'c'):
                    text = elem.text.strip() if elem.text else ''
                    if text:
                        tokens.append(text)
            if tokens:
                sentence = ''.join(tokens)
                sentences.append(sentence)
        return sentences

    def extract_samples_from_files(self, unique=True) -> list[tuple[str, str, str]]:
        """Extract samples from all matching pinyin and character XML files in the directories."""
        samples = []
        for filename in FileHandler.list_xml_files(self.pinyin_dir):
            pinyin_path = self.pinyin_dir / filename
            char_path = self.char_dir / filename
            if char_path.exists():
                file_samples = self.extract_samples_from_file(pinyin_path, char_path)
                samples.extend(file_samples)
        if unique:
            samples = list(set(samples))
        return samples

    def extract_sentences_from_files(self) -> list[str]:
        """Extract sentences from all character XML files in the character directory."""
        sentences = []
        for filename in FileHandler.list_xml_files(self.char_dir, extension=".XML"):
            filepath = self.char_dir / filename
            if filepath.exists():
                file_sentences = self.extract_sentences_from_file(filepath)
                sentences.extend(file_sentences)
        return sentences

    @staticmethod
    def remove_tone_numbers(syllables: list[str]) -> list[str]:
        """Remove trailing tone digits from each pinyin syllable."""
        return [re.sub(r'\d$', '', syllable) for syllable in syllables]

    @staticmethod
    def normalize_pinyin(syllables: list[str]) -> list[str]:
        """Normalize pinyin by replacing 'uu' with 'v'."""
        return [syllable.replace('uu', 'v') for syllable in syllables]

    @staticmethod
    def split_polyphonic(word: str) -> list[str]:
        """Split a polyphonic pinyin word into its component syllables."""
        return re.findall(r'[a-z]+[1-5]?', word)
