from typing import Iterable, Dict, Union


class NoteSentencizer(object):
    """
    This class is used to read text and split it into 
    sentences (and their start and end positions)
    This class considers an entire note or text as
    a single sentence
    """

    def __init__(self):
        """
        Nothing to be initialized
        """

    def get_sentences(self, text: str) -> Iterable[Dict[str, Union[str, int]]]:
        """
        Return an iterator that iterates through the sentences in the text.
        In this case it just returns the text itself.
        Args:
            text (str): The text
        Returns:
            (Iterable[Dict[str, Union[str, int]]]): Yields a dictionary that contains the text of the sentence
                                                    the start position of the sentence in the entire text
                                                    and the end position of the sentence in the entire text
        """
        yield {
            'text': text,
            'start': 0,
            'end': len(text),
            'last_token': None
        }
