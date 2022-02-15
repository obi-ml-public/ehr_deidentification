from typing import Mapping, Union, Sequence, List
from .mismatch_error import MismatchError


class NERTokenLabels(object):
    """
    This class is used to align tokens with the spans
    Each token is assigned one of the following labels
    'B-LABEL', 'I-LABEL', 'O'. For example the text
    360 Longwood Avenue is 2 tokens - [360, Longwood, Avenue]
    and each token would be assigned the following labels
    [B-LOC, I-LOC, I-LOC] (this would also depend on what
    notation we are using). Generally the data after prodigy
    annotation has all the tokens and all the spans.
    We would have tokens:[tok1, tok2, ... tokn]
    and spans:[span1:[tok1, tok2, tok3], span2:[tok7], ... span k]
    This would be used to convert into the format we are using
    which is assign the label to each token based on which span it
    belongs to.
    """

    def __init__(
            self,
            spans: List[Mapping[str, Union[str, int]]],
            notation: str
    ):
        """
        Initialize variables that will be used to align tokens
        and span labels. The spans variable will contain all the spans
        in the note. Notation is whether we would like to use BIO, IO, BILOU, 
        when assigning the label to each token based on which span it belongs to.
        Keep track of the total number of spans etc.
        Args:
            spans (Sequence[Mapping[str, Union[str, int]]]): List of all the spans in the text
            notation (str): NER label notation
        """
        # Keeps track of all the spans (list) in the text (note)
        self._spans = spans
        for span in self._spans:
            if type(span['start']) != int or type(span['end']) != int:
                raise ValueError('The start and end keys of the span must be of type int')
        self._spans.sort(key=lambda _span: (_span['start'], _span['end']))
        # The current span is the first element of the list
        self._current_span = 0
        # Boolean variable that indicates whether the token is inside
        # the span (I-LABEL)
        self._inside = False
        # Total number of spans
        self._span_count = len(self._spans)
        # Depending on the notation passed, we will return the label for
        # the token accordingly
        if notation == 'BIO':
            self._prefix_single = 'B-'
            self._prefix_begin = 'B-'
            self._prefix_inside = 'I-'
            self._prefix_end = 'I-'
            self._prefix_outside = 'O'
        elif notation == 'BIOES':
            self._prefix_single = 'S-'
            self._prefix_begin = 'B-'
            self._prefix_inside = 'I-'
            self._prefix_end = 'E-'
            self._prefix_outside = 'O'
        elif notation == 'BILOU':
            self._prefix_single = 'U-'
            self._prefix_begin = 'B-'
            self._prefix_inside = 'I-'
            self._prefix_end = 'L-'
            self._prefix_outside = 'O'
        elif notation == 'IO':
            self._prefix_single = 'I-'
            self._prefix_begin = 'I-'
            self._prefix_inside = 'I-'
            self._prefix_end = 'I-'
            self._prefix_outside = 'O'

    def __check_begin(self, token: Mapping[str, Union[str, int]]) -> str:
        """
        Given a token, return the label (B-LABEL) and check whether the token
        covers the entire span or is a sub set of the span
        Args:
            token (Mapping[str, Union[str, int]]): Contains the token text, start and end position of the token
                                                   in the text
        Returns:
            (str): The label - 'B-LABEL'
        """
        # Set the inside flag to true to indicate that the next token that is checked
        # will be checked to see if it belongs 'inside' the span
        self._inside = True
        if token['end'] > int(self._spans[self._current_span]['end']):
            raise MismatchError('Span and Token mismatch - Begin Token extends longer than the span')
        # If this token does not cover the entire span then we expect another token
        # to be in the span and that token should be assigned the I-LABEL
        elif token['end'] < int(self._spans[self._current_span]['end']):
            return self._prefix_begin + self._spans[self._current_span]['label']
        # If this token does cover the entire span then we set inside = False
        # to indicate this span is complete and increment the current span
        # to move onto the next span in the text
        elif token['end'] == int(self._spans[self._current_span]['end']):
            self._current_span += 1
            self._inside = False
            return self._prefix_single + self._spans[self._current_span - 1]['label']

    def __check_inside(self, token: Mapping[str, Union[str, int]]) -> str:
        """
        Given a token, return the label (I-LABEL) and check whether the token
        covers the entire span or is still inside the span. 
        Args:
            token (Mapping[str, Union[str, int]]): Contains the token text, start and end position of the token
                                                   in the text
        Returns:
            (str): The label - 'I-LABEL'
        """

        if (token['start'] >= int(self._spans[self._current_span]['end'])
                or token['end'] > int(self._spans[self._current_span]['end'])):
            raise MismatchError('Span and Token mismatch - Inside Token starts after the span ends')
        # If this token does not cover the entire span then we expect another token
        # to be in the span and that token should be assigned the I-LABEL
        elif token['end'] < int(self._spans[self._current_span]['end']):
            return self._prefix_inside + self._spans[self._current_span]['label']
        # If this token does cover the entire span then we set inside = False
        # to indicate this span is complete and increment the current span
        # to move onto the next span in the text
        elif token['end'] == int(self._spans[self._current_span]['end']):
            self._current_span += 1
            self._inside = False
            return self._prefix_end + self._spans[self._current_span - 1]['label']

    def get_labels(self, token: Mapping[str, Union[str, int]]) -> str:
        """
        Given a token, return the label (B-LABEL, I-LABEL, O) based on
        the spans present in the text & the desired notation.
        Args:
            token (Mapping[str, Union[str, int]]): Contains the token text, start and end position of the token
                                                   in the text
        Returns:
            (str): One of the labels according to the notation - 'B-LABEL', 'I-LABEL', 'O'
        """
        # If we have iterated through all the spans in the text (note), all the tokens that
        # come after the last span will be marked as 'O' - since they don't belong to any span
        if self._current_span >= self._span_count:
            return self._prefix_outside
        # Check if the span can be assigned the B-LABEL
        if token['start'] == int(self._spans[self._current_span]['start']):
            return self.__check_begin(token)
        # Check if the span can be assigned the I-LABEL
        elif token['start'] > int(self._spans[self._current_span]['start']) and self._inside is True:
            return self.__check_inside(token)
        # Check if the token is outside a span
        elif self._inside is False and (token['end'] <= int(self._spans[self._current_span]['start'])):
            return self._prefix_outside
        else:
            raise MismatchError(
                'Span and Token mismatch - the span and tokens don\'t line up. There might be a tokenization issue '
                'that needs to be fixed')
