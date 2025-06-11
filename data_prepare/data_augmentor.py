"""
Module for augmenting Chinese sentences using cumulative segmentation logic.
Provides utilities to load space-separated sentences and generate augmented sequences.
"""
from datasets import Dataset

from tools.t9_hanzi_converter import T9PinyinHanziConverter

from pathlib import Path
import json


class DataAugmentor:
    def __init__(self) -> None:
        self.data: list[tuple[list[str], str]] | None = None
        self.data_copy: list[tuple[list[str], str]] | None = None
        self.t9: list[str] | None = None


    def load_sentences(self, corpus_path=Path("../data/zh_sent_dataset.tsv")) -> None:
        """
        Loads space-separated Chinese sentences from a UTF-8 encoded text file.

        Args:
            corpus_path (Path): Path to the text corpus file.
        """
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.data = [([line], None) for line in f.readlines()]
            self.data_copy = self.data.copy()

    def load_from_copy(self) -> "DataAugmentor":
        self.data = self.data_copy.copy()
        return self

    def prefix_span_sentences(self) -> "DataAugmentor":
        data = []
        for line, _ in self.data:
            line_str = line[0]
            spanned_lines = DataAugmentor.prefix_span_sentence(line_str)
            data.extend(spanned_lines)
        self.data = [(line, None) for line in data]
        return self

    @staticmethod
    def prefix_span_sentence(sentence: str, sep=" ") -> list[list[str]]:
        """
        input example: '我 喜欢 中文'
        output example: [['我'], ['我', '喜欢'], ['我', '喜欢', '中文']]
        """
        tokens = sentence.strip().split(sep=sep)
        sequences = []

        # Forward prefix span
        for i in range(2, len(tokens) + 1):
            if tokens[i - 1].isalpha():
                sequences.append(tokens[:i])

        return sequences

    def generate_feature_label_pair(self) -> "DataAugmentor":
        self.data = [(sentence[:-1], sentence[-1]) for sentence, _ in self.data]
        return self

    def generate_t9_from_labels(self) -> "DataAugmentor":
        self.t9 = [DataAugmentor.label_to_t9(label) for _, label in self.data]
        return self

    @staticmethod
    def label_to_t9(label: str) -> str:
        pinyin = "".join(T9PinyinHanziConverter.hanzi2pinyin(label))
        return T9PinyinHanziConverter.pinyin2t9(pinyin)

    def augment_sentences_with_t9(self) -> "DataAugmentor":
        data = []
        for (sentence, label), t9 in zip(self.data, self.t9):
            data.extend(DataAugmentor.augment_sentence_with_t9(sentence, label, t9))
        self.data = data
        return self

    @staticmethod
    def augment_sentence_with_t9(sentence: list[str], label: str, t9: str) -> list[tuple[str, str]]:
        data = []
        digits = list(t9)
        for i in range(1, len(digits) + 1):
            new_sentence = sentence + digits[:i]
            data.append((new_sentence, label))
        return data

    def prefix_span_last_numbers(self) -> "DataAugmentor":
        for sentences in self.data:
            for sentence in sentences:
                last_word = sentence[-1]
                if last_word.isdigit():
                    sentence.pop()
                    sentence.extend(list(last_word))
        return self

    def truncate_sentences(self, max_len=32) -> "DataAugmentor":
        for i, (sentence, label) in enumerate(self.data):
            if len(sentence) > max_len:
                self.data[i] = (DataAugmentor.left_truncate_sentence(sentence, max_len), label)
        return self

    @staticmethod
    def left_truncate_sentence(sentence: list[str], max_len: int) -> list[str]:
        return sentence[-max_len:]

    def augment_data(self, max_length=32) -> "DataAugmentor":
        self.load_from_copy().prefix_span_sentences().generate_feature_label_pair().generate_t9_from_labels().augment_sentences_with_t9().truncate_sentences(max_length)
        return self

    def output_json(self, json_path: Path=Path("../data/training_data/whole_corpus.json")) -> None:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False)

    def output_parquet(self, parquet_path=Path("../data/training_data/training_corpus.parquet")) -> None:
        formatted_data = [{"input": sentence, "label": label} for sentence, label in self.data]
        dataset = Dataset.from_list(formatted_data)
        dataset.to_parquet(parquet_path)

    def get_data(self) -> list[tuple[list[str], str]]:
        return self.data
