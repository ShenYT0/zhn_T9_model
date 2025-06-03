"""
Module for augmenting Chinese sentences using cumulative segmentation logic.
Provides utilities to load space-separated sentences and generate augmented sequences.
"""

from pathlib import Path


class DataAugmentor:
    def __init__(self) -> None:
        self.data = None

    def load_sentences(self, corpus_path=Path("../data/zh_sent_dataset.tsv")) -> None:
        """
        Loads space-separated Chinese sentences from a UTF-8 encoded text file.

        Args:
            corpus_path (Path): Path to the text corpus file.
        """
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.data = f.readlines()

    def augment_sentences(self):
        self.data = list(map(DataAugmentor.augment_sentence, self.data))

    @staticmethod
    def augment_sentence(sentence: str) -> list[list[str]]:
        tokens = sentence.strip().split()
        sequences = []

        # Forward cumulative segments
        for i in range(1, len(tokens) + 1):
            sequences.append(tokens[:i])

        # Backward cumulative segments
        for i in range(1, len(tokens)):
            sequences.append(tokens[i:])

        return sequences

    def get_data(self):
        return self.data
