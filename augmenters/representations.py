import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Token:
    def __init__(self, word: str, ner_tag: str, pos_tag: str = None):
        self.word = word
        self.ner_tag = ner_tag
        self.pos_tag = pos_tag

    def get_word(self):
        return self.word

    def set_word(self, word: str):
        self.word = word

    def get_ner_tag(self):
        return self.ner_tag

    def __len__(self):
        return len(self.word)

    def __repr__(self):
        return f"Token(word='{self.word}', tag='{self.ner_tag}')"


class Sentence:
    def __init__(self):
        self.tokens = []

    def add_token(self, token):
        self.tokens.append(token)

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        return f"Sentence(tokens={self.tokens})"


class Corpus:
    def __init__(
        self, file_path: str, text_column_index: int = 0, ner_column_index: int = 1
    ):
        self.sentences = []

        if file_path:
            self._load_data(file_path, text_column_index, ner_column_index)

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def _load_data(self, file_path, text_column_index, ner_column_index):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.read().split("\n\n")

        for line in lines:
            sentence = Sentence()
            for token_line in line.split("\n"):
                if token_line.strip() == "" or token_line.startswith("-DOCSTART-"):
                    continue
                columns = (
                    token_line.split()
                )  # Split by spaces if your data is space-separated
                if len(columns) > max(text_column_index, ner_column_index):
                    word = columns[text_column_index]
                    ner_tag = columns[ner_column_index]
                    sentence.add_token(Token(word, ner_tag))
            if sentence.tokens:
                self.add_sentence(sentence)

        logging.info(f"{len(self.sentences)} sentences loaded into the corpus.")

    def __repr__(self):
        return f"Corpus(sentences={self.sentences})"

    def filter_empty_sentences(self):
        self.sentences = [sentence for sentence in self.sentences if len(sentence) > 0]

    def __len__(self):
        return len(self.sentences)
