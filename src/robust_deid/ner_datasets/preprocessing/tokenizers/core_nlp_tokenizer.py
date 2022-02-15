import json
from typing import Iterable, Mapping, Dict, Union

from pycorenlp import StanfordCoreNLP


class CoreNLPTokenizer(object):
    """
    This class is used to read text and return the tokens
    present in the text (and their start and end positions)
    using core nlp tokenization
    """

    def __init__(self, port: int = 9000):
        """
        Initialize a core nlp server to read text and split it into 
        tokens using the core nlp annotators
        Args:
            port (int): The port to run the server on
        """
        self._core_nlp = StanfordCoreNLP('http://localhost:{0}'.format(port))

    def get_stanford_annotations(self, text: str, annotators: str = 'tokenize,ssplit,pos,lemma') -> Dict:
        """
        Use the core nlp server to annotate the text and return the
        results as a json object
        Args:
            text (str): The text to annotate
            annotators (str): The core nlp annotations to run on the text
        Returns:
            output (Dict): The core nlp results
        """
        output = self._core_nlp.annotate(text, properties={
            "timeout": "50000",
            "ssplit.newlineIsSentenceBreak": "two",
            'annotators': annotators,
            'outputFormat': 'json'
        })
        if type(output) is str:
            output = json.loads(output, strict=False)
        return output

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
        stanford_output = self.get_stanford_annotations(text)
        for sentence in stanford_output['sentences']:
            for token in sentence['tokens']:
                yield {'text': token['originalText'],
                       'start': token['characterOffsetBegin'],
                       'end': token['characterOffsetEnd']}
