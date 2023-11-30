import spacy

from amplifier import Corpus
from amplifier.augmenters import Augmenter

corpus = Corpus(file_path="tests/bio.txt", text_column_index=0, ner_column_index=3)
# print(corpus.sentences[5])

# for sentence in corpus.sentences:
#     print(sentence)


nlp = spacy.load("en_core_web_md", disable=["lemmatizer", "ner"])

augmenter = Augmenter(nlp)

print("Before augmentation")
for token in corpus.sentences[5].tokens:
    print(token.word)

augmenter.noun_augment_wordnet(corpus.sentences[5])

print("\nAfter augmentation")
for token in corpus.sentences[5].tokens:
    print(token.word)
