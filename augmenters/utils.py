import sys

import pandas as pd

from augmenters import Sentence, Token


def load_file(file: str, text_column_index: int, tag_column_index: int):
    data = pd.read_csv(file, sep="\t", header=None, encoding="utf-8")

    if text_column_index >= len(data.columns) or tag_column_index >= len(data.columns):
        raise ValueError("Column indices are out of range")

    lines = file.read().split("\n\n")

    sentences = []
    for line in lines:
        sentence = Sentence()
        for token_line in line.split("\n"):
            if token_line.strip() == "":  # Skip empty lines
                continue
            columns = token_line.split("\t")
            word = columns[text_column_index]
            ner_tag = columns[tag_column_index]
            sentence.add_token(Token(word, ner_tag))
        sentences.append(sentence)
