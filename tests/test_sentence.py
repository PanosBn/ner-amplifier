import pytest

from amplifier.representations import Sentence, Token


def test_init():
    sentence = Sentence()
    assert sentence.tokens == []
    assert sentence.spans == []


def test_add_token():
    sentence = Sentence()
    token = Token(word="test", ner="O")
    sentence.add_token(token)
    assert sentence.tokens == [token]


def test_identify_spans():
    sentence = Sentence()
    token1 = Token(word="test1", ner="O")
    token2 = Token(word="test2", ner="B-PER")
    token3 = Token(word="test3", ner="I-PER")
    sentence.add_token(token1)
    sentence.add_token(token2)
    sentence.add_token(token3)
    sentence._identify_spans()
    assert len(sentence.spans) == 1
    assert sentence.spans[0].tokens == [token2, token3]


def test_get_spans():
    sentence = Sentence()
    token1 = Token(word="test1", ner="B-PER")
    token2 = Token(word="test2", ner="I-PER")
    sentence.add_token(token1)
    sentence.add_token(token2)
    sentence._identify_spans()
    assert sentence.get_spans() == sentence.spans


def test_str():
    sentence = Sentence()
    token1 = Token(word="test1", ner="O")
    token2 = Token(word="test2", ner="O")
    sentence.add_token(token1)
    sentence.add_token(token2)
    assert str(sentence) == "test1 test2"


def test_iter():
    sentence = Sentence()
    token1 = Token(word="test1", ner="O")
    token2 = Token(word="test2", ner="O")
    sentence.add_token(token1)
    sentence.add_token(token2)
    for i, token in enumerate(sentence):
        assert token == sentence.tokens[i]


def test_len():
    sentence = Sentence()
    token1 = Token(word="test1", ner="O")
    token2 = Token(word="test2", ner="O")
    sentence.add_token(token1)
    sentence.add_token(token2)
    assert len(sentence) == 2


def test_repr():
    sentence = Sentence()
    token = Token(word="test", ner="O")
    sentence.add_token(token)
    assert (
        repr(sentence)
        == "Sentence(tokens=[Token({'word': 'test', 'ner': 'O', 'augmented': 'test'})])"
    )
