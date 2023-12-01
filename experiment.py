import spacy
import spacy_conll

from amplifier import Corpus
from amplifier.augmenters import NounAugmenter

corpus = Corpus(file_path="tests/bio.txt", text_column_index=0, ner_column_index=3)
# print(corpus.sentences[5])

# for sentence in corpus.sentences:
#     print(sentence)


nlp = spacy.load("en_core_web_md", disable=["lemmatizer", "ner"])
nlp.add_pipe("conll_formatter", last=True)

augmenter = NounAugmenter(nlp)

print("Before augmentation")
for token in corpus.sentences[2].tokens:
    print(token.word, token.get_ner_tag(), token.get_pos_tag())

augmenter.noun_augment_sense2vec(corpus.sentences[2], model_path="s2v_old/")

print("\nAfter augmentation")
for token in corpus.sentences[2].tokens:
    print(token.word, token.get_ner_tag(), token.get_pos_tag())
