import random
import re
from typing import Iterable, Dict, Sequence, Union, Mapping, Optional, List

from .labels import NERTokenLabels, NERPredictTokenLabels, MismatchError

random.seed(41)


class Dataset(object):
    """
    Build a NER token classification dataset. Each token should have a corresponding label
    based on the annotated spans
    For training we will build the dataset using the annotated spans (e.g from prodigy)
    For predictions we will assign default labels. to keep the format of the dataset the same
    The dataset is on a sentence level, i.e each note is split into sentences and the
    task is run on a sentence level. Even the predictions are run on a sentence level
    The dataset would be something like:
    Tokens: [tok1, tok2, ... tok n]
    Labels: [lab1, lab2, ... lab n]
    For the prediction mode the labels would be: [default, default, default .... default]
    This script can also be used for predictions, the Labels will be filled with some
    default value. This is done so that we can use the same script for building a dataset to train a model
    and a dataset to obtain predictions using a model
    """

    def __init__(
            self,
            sentencizer,
            tokenizer
    ):
        """
        Build a NER token classification dataset
        For training we will build the dataset using the annotated spans (e.g from prodigy)
        For predictions we will assign default labels.
        The dataset is on a sentence level, i.e each note is split into sentences and the de-id
        task is run on a sentence level. Even the predictions are run on a sentence level
        The dataset would be something like:
        Tokens: [tok1, tok2, ... tok n]
        Labels: [lab1, lab2, ... lab n]
        This script can also be used for predictions, the Labels will be filled with some
        default value. This is done so that we can use the same script for building a dataset to train a model
        and a dataset to obtain predictions using a model 
        Args:
            sentencizer (Union[SpacySentencizer, MimicStanzaSentencizer, NoteSentencizer]): The sentencizer to use for 
                                                                                            splitting notes into
                                                                                            sentences
            tokenizer (Union[ClinicalSpacyTokenizer, SpacyTokenizer, CoreNLPTokenizer]): The tokenizer to use for
                                                                                         splitting text into tokens
        """
        self._sentencizer = sentencizer
        self._tokenizer = tokenizer

    def get_tokens(
            self,
            text: str,
            spans: Optional[List[Mapping[str, Union[str, int]]]] = None,
            notation: str = 'BIO',
            token_text_key: str = 'text',
            label_key: str = 'label'
    ) -> Iterable[Sequence[Dict[str, Union[str, int]]]]:
        """
        Get a nested list of tokens where the the inner list represents the tokens in the
        sentence and the outer list will contain all the sentences in the note
        Args:
            text (str): The text present in the note
            spans (Optional[List[Mapping[str, Union[str, int]]]]): The NER spans in the note. This will be none if
                                                                   building the dataset for prediction
            notation (str): The notation we will be using for the label scheme (e.g BIO, BILOU etc)
            token_text_key (str): The key where the note text is present
            label_key (str): The key where the note label for each token is present
        Returns:
            Iterable[Sequence[Dict[str, Union[str, int]]]]: Iterable that iterates through all the sentences 
                                                            and yields the list of tokens in each sentence
        """
        # Initialize the object that will be used to align tokens and spans based on the notation
        # as mentioned earlier - this will be used only when mode is train - because we have
        # access to labelled spans for the notes
        if spans is None:
            label_spans = NERPredictTokenLabels('O')
        else:
            label_spans = NERTokenLabels(spans=spans, notation=notation)
        # Iterate through the sentences in the note
        for sentence in self._sentencizer.get_sentences(text=text):
            # This is used to determine the position of the tokens with respect to the entire note
            offset = sentence['start']
            # Keeps track of the tokens in the sentence
            tokens = list()
            for token in self._tokenizer.get_tokens(text=sentence['text']):
                # Get the token position (start, end) in the note
                token['start'] += offset
                token['end'] += offset
                if token[token_text_key].strip() in ['\n', '\t', ' ', ''] or token['start'] == token['end']:
                    continue
                # Shorten consecutive sequences of special characters, this can prevent BERT from truncating
                # extremely long sentences - that could arise because of these characters
                elif re.search('(\W|_){9,}', token[token_text_key]):
                    print('WARNING - Shortening a long sequence of special characters from {} to 8'.format(
                        len(token[token_text_key])))
                    token[token_text_key] = re.sub('(?P<specchar>(\W|_)){8,}', '\g<specchar>' * 8,
                                                   token[token_text_key])
                elif len(token[token_text_key].split(' ')) != 1:
                    print('WARNING - Token contains a space character - will be replaced with hyphen')
                    token[token_text_key] = token[token_text_key].replace(' ', '-')
                # Get the labels for each token based on the notation (BIO)
                # In predict mode - the default label (e.g O) will be assigned
                try:
                    # Get the label for the token - based on the notation
                    label = label_spans.get_labels(token=token)
                    if label[2:] == 'OTHERISSUE':
                        raise ValueError('Fix OTHERISSUE spans')
                # Check if there is a token and span mismatch, i.e the token and span does not align
                except MismatchError:
                    print(token)
                    raise ValueError('Token-Span mismatch')
                token[label_key] = label
                tokens.append(token)
            if tokens:
                yield tokens
