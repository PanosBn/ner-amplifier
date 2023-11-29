class Token:
    def __init__(self, word: str, ner_tag: str, pos_tag: str):
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
    def __init__(self):
        self.sentences = []

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def __repr__(self):
        return f"Corpus(sentences={self.sentences})"
    
    def filter_empty_sentences(self):
        self.sentences = [sentence for sentence in self.sentences if len(sentence) > 0]

    def __len__(self):
        return len(self.sentences)
    
    