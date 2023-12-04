import re

from spacy.tokenizer import Tokenizer


def conll_2003_tokenizer(nlp):
    # Regular expression that matches any whitespace or standalone punctuation
    split_pattern = (
        r"\s+|(?<!\w)[\.\,\!\?\:\;\"\'\-\(\)]|[\.\,\!\?\:\;\"\'\-\(\)](?!\w)"
    )

    def tokenize(text):
        tokens = []
        start = 0
        for match in re.finditer(split_pattern, text):
            end = match.start()
            if start < end:  # Non-empty token
                tokens.append(text[start:end])
            tokens.append(
                text[match.start() : match.end()]
            )  # Include the punctuation as a separate token
            start = match.end()
        if start < len(text):  # Add any remaining text as a token
            tokens.append(text[start:])
        return tokens

    return Tokenizer(nlp.vocab, token_match=tokenize)
