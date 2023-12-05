import spacy

# import
from amplifier import Corpus
from amplifier.augmenters import NounAugmenter

# corpus = Corpus(file_path="tests/bio.txt", text_column_index=0, ner_column_index=3)

column_mapping = {"word": 0, "ner": 3}
corpus = Corpus(file_path="tests/dev.txt", column_mapping=column_mapping)


augmenter = NounAugmenter()

# augmenter.noun_augment_sense2vec(corpus, model_path="s2v_old/")
augmenter.noun_augment_wordnet(corpus)

# for sentence in corpus.sentences:
#     for token in sentence.tokens:
#         print(f"Original: {token.get_attribute('word')}, Augmented: {token.get_attribute('augmented')}")


# augmenter.noun_augment_sense2vec(corpus.sentences[0], model_path="path_to_sense2vec_model")

# Example to print the original and augmented forms
# for sentence in corpus.sentences:
#     for token in sentence.tokens:
#         print(
#             f"Original: {token.get_attribute('word')}, Augmented: {token.get_attribute('augmented')}"
#         )

corpus.export_to_conll("augmented.txt", delimiter="\t")
#
