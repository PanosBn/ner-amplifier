import pytest

from amplifier.representations import Corpus, Sentence, Token


def test_corpus_initialization_with_valid_data():
    column_mapping = {"word": 0, "ner": 3}
    corpus = Corpus(file_path="tests/test_data.txt", column_mapping=column_mapping)
    assert len(corpus.sentences) == 2
    assert len(corpus.sentences[0].tokens) > 0


def test_corpus_initialization_with_invalid_path():
    column_mapping = {"word": 0, "ner": 3}
    with pytest.raises(FileNotFoundError):
        Corpus(file_path="wololololo.txt", column_mapping=column_mapping)


# def test_corpus_initialization_with_malformed_data():
#     column_mapping = {"word": 0, "ner": 3}
#     corpus = Corpus(file_path="path/to/malformed_data.txt", column_mapping=column_mapping)
#     assert len(corpus.sentences) == 0 or all(len(sentence.tokens) == 0 for sentence in corpus.sentences)


def test_adding_pos_tags_when_missing():
    column_mapping = {"word": 0, "ner": 3}
    corpus = Corpus(file_path="tests/test_data.txt", column_mapping=column_mapping)
    for sentence in corpus.sentences:
        for token in sentence.tokens:
            assert token.get_attribute("pos") is not None


def test_entity_index_creation():
    column_mapping = {"word": 0, "ner": 3}
    corpus = Corpus(file_path="tests/test_data.txt", column_mapping=column_mapping)
    assert "PER" in corpus._entity_index
    assert "LOC" in corpus._entity_index


def test_span_identification():
    column_mapping = {"word": 0, "ner": 3}
    corpus = Corpus(file_path="tests/test_data.txt", column_mapping=column_mapping)
    assert len(corpus.sentences[0].get_spans()) > 0
