from typing import Iterable, Dict, Union

import spacy


class SpacySentencizer(object):
    """
    This class is used to read text and split it into 
    sentences (and their start and end positions)
    using a spacy model
    """

    def __init__(self, spacy_model: str):
        """
        Initialize a spacy model to read text and split it into 
        sentences.
        Args:
            spacy_model (str): Name of the spacy model
        """
        self._nlp = spacy.load(spacy_model)

    def get_sentences(self, text: str) -> Iterable[Dict[str, Union[str, int]]]:
        """
        Return an iterator that iterates through the sentences in the text
        Args:
            text (str): The text
        Returns:
            (Iterable[Dict[str, Union[str, int]]]): Yields a dictionary that contains the text of the sentence
                                                    the start position of the sentence in the entire text
                                                    and the end position of the sentence in the entire text
        """
        document = self._nlp(text)
        for sentence in document.sents:
            yield {'text': sentence.text,
                   'start': sentence.start_char,
                   'end': sentence.end_char,
                   'last_token': None}
