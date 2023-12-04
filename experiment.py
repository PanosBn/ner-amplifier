import spacy

# import
from amplifier import Corpus
from amplifier.augmenters import NounAugmenter

# corpus = Corpus(file_path="tests/bio.txt", text_column_index=0, ner_column_index=3)

column_mapping = {"word": 0, "ner": 3}
corpus = Corpus(file_path="tests/bio.txt", column_mapping=column_mapping)

# for sentence in corpus.sentences:
#     for token in sentence.tokens:
#         print(
#             f"Word: {token.get_attribute('word')}, POS: {token.get_attribute('pos')}, NER: {token.get_attribute('ner')}"
#         )
#     print()  # Print a newline after each sentence for clarity


# Assuming `corpus` is an instance of the Corpus class
augmenter = NounAugmenter()

# Augment using WordNet or Sense2Vec
# augmenter.noun_augment_sense2vec(corpus, model_path="s2v_old/")# Or
# augmenter.noun_augment_sense2vec(corpus.sentences[0], model_path="path_to_sense2vec_model")
augmenter.noun_augment_wordnet(corpus)
# Example to print the original and augmented forms
for sentence in corpus.sentences:
    for token in sentence.tokens:
        print(
            f"Original: {token.get_attribute('word')}, Augmented: {token.get_attribute('augmented')}"
        )

#
