class Token:
    def __init__(self, word, ner_tag):
        self.word = word
        self.ner_tag = ner_tag

    def get_word(self):
        return self.word

    def set_word(self, word):
        self.word = word

    def get_ner_tag(self):
        return self.ner_tag

    def __repr__(self):
        return f"Token(word='{self.word}', tag='{self.ner_tag}')"


class Sentence:
    def __init__(self):
        self.tokens = []

    def add_token(self, token):
        self.tokens.append(token)

    def __repr__(self):
        return f"Sentence(tokens={self.tokens})"


class Corpus:
    def __init__(self):
        self.sentences = []

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def __repr__(self):
        return f"Corpus(sentences={self.sentences})"

    def __len__(self):
        return len(self.sentences)
