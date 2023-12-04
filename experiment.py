import spacy

# import
from amplifier import Corpus
from amplifier.augmenters import NounAugmenter

# corpus = Corpus(file_path="tests/bio.txt", text_column_index=0, ner_column_index=3)

column_mapping = {"word": 0, "ner": 3}
corpus = Corpus(file_path="tests/bio.txt", column_mapping=column_mapping)

for sentence in corpus.sentences:
    for token in sentence.tokens:
        print(
            f"Word: {token.get_attribute('word')}, POS: {token.get_attribute('pos')}, NER: {token.get_attribute('ner')}"
        )
    print()  # Print a newline after each sentence for clarity


# print(corpus.sentences[5])

# for sentence in corpus.sentences:
#     print(sentence)

# nlp = spacy.load("en_core_web_md", disable=["lemmatizer", "ner"])

# nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)
# nlp.tokenizer = conll_2003_tokenizer(nlp)
# # print(nlp.tokenizer.explain("AL-AIN, United Arab Emirates 1996-12-06"))


# augmenter = NounAugmenter(nlp)

# # print("Before augmentation")
# # for token in corpus.sentences[4].tokens:
# #     print(token.word, token.get_ner_tag(), token.get_pos_tag())

# augmenter.noun_augment_sense2vec(corpus.sentences[4], model_path="s2v_old/")

# print("\nAfter augmentation")
# for token in corpus.sentences[4].tokens:
#     print(token.word, token.get_ner_tag(), token.get_pos_tag())