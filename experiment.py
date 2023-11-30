from augmenters import Corpus

corpus = Corpus(file_path="tests/bio.txt", text_column_index=0, ner_column_index=3)
print(corpus.sentences[5])
