import pytest

import amplifier
from amplifier.representations import Token


def test_init():
    token = Token(word="test", ner="O")
    assert token.get_attribute("word") == "test"
    assert token.get_attribute("ner") == "O"
    assert token.get_attribute("augmented") == "test"


def test_get_attribute():
    token = Token(word="test", ner="O")
    assert token.get_attribute("word") == "test"


def test_set_attribute():
    token = Token(word="test", ner="O")
    token.set_attribute("word", "new_test")
    assert token.get_attribute("word") == "new_test"


def test_get_ner_tag():
    token = Token(word="test", ner="O")
    assert token.get_ner_tag() == "O"


def test_str():
    token = Token(word="test", ner="O")
    assert str(token) == "test"


def test_len():
    token = Token(word="test", ner="O")
    assert len(token) == 4


def test_repr():
    token = Token(word="test", ner="O")
    assert repr(token) == "Token({'word': 'test', 'ner': 'O', 'augmented': 'test'})"
