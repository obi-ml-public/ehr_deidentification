import spacy
from typing import Tuple, Iterable, Mapping, Dict, Union


class SpacyTokenizer(object):
    """
    This class is used to read text and return the tokens
    present in the text (and their start and end positions)
    using spacy
    """

    def __init__(self, spacy_model: str):
        """
        Initialize a spacy model to read text and split it into 
        tokens.
        Args:
            spacy_model (str): Name of the spacy model
        """
        self._nlp = spacy.load(spacy_model)

    @staticmethod
    def __get_start_and_end_offset(token: spacy.tokens.Token) -> Tuple[int, int]:
        """
        Return the start position of the token in the entire text
        and the end position of the token in the entire text
        Args:
            token (spacy.tokens.Token): The spacy token object
        Returns:
            start (int): the start position of the token in the entire text
            end (int): the end position of the token in the entire text
        """
        start = token.idx
        end = start + len(token)
        return start, end

    def get_tokens(self, text: str) -> Iterable[Dict[str, Union[str, int]]]:
        """
        Return an iterable that iterates through the tokens in the text
        Args:
            text (str): The text to annotate
        Returns:
            (Iterable[Mapping[str, Union[str, int]]]): Yields a dictionary that contains the text of the token
                                                       the start position of the token in the entire text
                                                       and the end position of the token in the entire text
        """
        document = self._nlp(text)
        for token in document:
            start, end = SpacyTokenizer.__get_start_and_end_offset(token)
            yield {'text': token.text, 'start': start, 'end': end}
