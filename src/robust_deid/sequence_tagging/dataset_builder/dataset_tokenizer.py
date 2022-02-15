from typing import Mapping, Sequence, List, Union, Optional, NoReturn
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer


class DatasetTokenizer(object):
    """
    The main goal of this class is to solve the problem described below.
    Most of the comments have been copied from the huggingface webpage.
    What this class does is initialize a tokenizer with the desired parameters
    and then tokenize our dataset and align the tokens with the labels
    while keeping in mind the problem & solution described below. We can use this
    function to train and for predictions - we just assume the predictions dataset
    will have a label column filled with some values (so this code can be re-used).
    Now we arrive at a common obstacle with using pre-trained models for 
    token-level classification: many of the tokens in the dataset may not 
    be in the tokenizer vocabulary. Bert and many models like it use a method 
    called WordPiece Tokenization, meaning that single words are split into multiple 
    tokens such that each token is likely to be in the vocabulary. For example, 
    the tokenizer would split the date (token) 2080 into the tokens ['208', '##0']. 
    This is a problem for us because we have exactly one tag per token (2080 -> B-DATE). 
    If the tokenizer splits a token into multiple sub-tokens, then we will end up with 
    a mismatch between our tokens and our labels (208, 0) - two tokens but one label (B-DATE). 
    One way to handle this is to only train on the tag labels for the first subtoken of a 
    split token. We can do this in huggingface Transformers by setting the labels 
    we wish to ignore to -100. In the example above, if the label for 2080 is B-DATE
    and say the id (from the label to id mapping) for B-DATE is 3, we would set the labels 
    of ['208', '##0'] to [3, -100]. This tells the model to ignore the tokens labelled with
    -100 while updating the weights etc.
    """

    def __init__(
            self,
            tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
            token_column: str,
            label_column: str,
            label_to_id: Mapping[str, int],
            b_to_i_label: Sequence[int],
            padding: Union[bool, str],
            truncation: Union[bool, str],
            is_split_into_words: bool,
            max_length: Optional[int],
            label_all_tokens: bool,
            token_ignore_label: Optional[str]
    ) -> NoReturn:
        """
        Set the tokenizer we are using to subword tokenizer our dataset. The name of the 
        column that contains the pre-split tokens, the name of the column that contains
        the labels for each token, label to id mapping.
        Set the padding strategy of the input. Set whether to truncate the input tokens.
        Indicate whether the input is pre-split into tokens. Set the max length of the 
        input tokens (post subword tokenization). This will be used in conjunction with truncation.
        Set whether we want to label even the sub tokens
        In the description above we say for 2080 (B-DATE) - [208, ##0]
        We do [3, -100] - which says assume to label of token 2080 is the one 
        predicted for 208 or we can just label both sub tokens
        in which case it would be [3, 3] - so we would label 208 as DATE
        and ##0 as DATE - then we would have to figure out how to merge these
        labels etc
        Args:
            tokenizer (Union[PreTrainedTokenizerFast, PreTrainedTokenizer]): Tokenizer from huggingface
            token_column (str): The column that contains the tokens in the dataset
            label_column (str): The column that contains the labels in the dataset
            label_to_id (Mapping[str, int]): The mapping between labels and ID
            b_to_i_label (Sequence[int]): The mapping between labels and ID
            padding (Union[bool, str]): Padding strategy
            truncation (Union[bool, str]): Truncation strategy
            is_split_into_words (bool): Is the input pre-split(tokenized)
            max_length (Optional[int]): Max subword tokenized length for the model
            label_all_tokens (bool): Whether to label sub words
            token_ignore_label (str): The value of the token ignore label - we ignore these in the loss computation
        """
        self._tokenizer = tokenizer
        self._token_column = token_column
        self._label_column = label_column
        self._label_to_id = label_to_id
        self._b_to_i_label = b_to_i_label
        # We can tell the tokenizer that we’re dealing with ready-split tokens rather than full
        # sentence strings by passing is_split_into_words=True.
        # Set the following parameters using the kwargs
        self._padding = padding
        self._truncation = truncation
        self._is_split_into_words = is_split_into_words
        self._max_length = max_length
        self._label_all_tokens = label_all_tokens
        self._token_ignore_label = token_ignore_label
        self._ignore_label = -100

    def tokenize_and_align_labels(self, dataset: Dataset) -> Dataset:
        """
        This function is the one that is used to read the input dataset
        Run the subword tokenization on the pre-split tokens and then
        as mentioned above align the subtokens and labels and add the ignore
        label. This will read the input - say [60, year, old, in, 2080]
        and will return the subtokens - [60, year, old, in, 208, ##0]
        some other information like token_type_ids etc
        and the labels [0, 20, 20, 20, 3, -100] (0 - corresponds to B-AGE, 20 corresponds to O
        and 3 corresponds to B-DATE. This returned input serves as input for training the model
        or for gathering predictions from a trained model. 
        Another important thing to note is that we have mentioned before that
        we add chunks of tokens that appear before and after the current chunk for context. We would
        also need to assign the label -100 (ignore_label) to these chunks, since we are using them
        only to provide context. Basically if a token has the label NA, we don't use it for
        training or evaluation. For example the input would be something
        like tokens: [James, Doe, 60, year, old, in, 2080, BWH, tomorrow, only], 
        labels: [NA, NA, B-DATE, O, O, O, B-DATE, NA, NA, NA]. NA represents the tokens used for context
        This function would return some tokenizer info (e.g attention mask etc), along with
        the information that maps the tokens to the subtokens - 
        [James, Doe, 60, year, old, in, 208, ##0, BW, ##h, tomorrow, only]
        and the labels - [-100, -100, 0, 20, 20, 20, 3, -100, -100, -100] 
        (if label_all_tokens was true, we would return [-100, -100, 0, 20, 20, 20, 3, 3, -100, -100]). 
        Args:
            dataset (Dataset): The pre-split (tokenized dataset) that contain labels
        Returns:
            tokenized_inputs (Dataset): Subword tokenized and label aligned dataset
        """
        # Run the tokenizer - subword tokenization
        tokenized_inputs = self._tokenizer(
            dataset[self._token_column],
            padding=self._padding,
            truncation=self._truncation,
            max_length=self._max_length,
            is_split_into_words=self._is_split_into_words,
        )
        # Align the subwords and tokens
        labels = [self.__get_labels(
            labels,
            tokenized_inputs.word_ids(batch_index=index)
        ) for index, labels in enumerate(dataset[self._label_column])]
        tokenized_inputs[self._label_column] = labels

        return tokenized_inputs

    def __get_labels(
            self,
            labels: Sequence[str],
            word_ids: Sequence[int]
    ) -> List[int]:
        """
        Go thorough the subword tokens - which are given as word_ids. 2 different tokens
        2080 & John will have different word_ids, but the subword tokens 2080 & ##0 will
        have the same word_id, we use this to align and assign the labels accordingly.
        if the subword tokens belong to [CLS], [SEP] append the ignore label (-100) to the
        list of labels. If the (2080) subword token (##0) belongs to a token - 2080
        then the labels would be [3, -100] if label_all_tokens is false. Also if the token
        is used only for context (with label NA) it would get the value -100 for its label
        Args:
            labels (Sequence[str]): The list of labels for the input (example)
            word_ids (Sequence[int]): The word_ids after subword tokenization of the input
        Returns:
            label_ids (List[int]): The list of label ids for the input with the ignore label (-100) added
                                       as required.
        """
        label_ids = list()
        previous_word_idx = None
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(self._ignore_label)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                if labels[word_idx] == self._token_ignore_label:
                    label_ids.append(self._ignore_label)
                else:
                    label_ids.append(self._label_to_id[labels[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if labels[word_idx] == self._token_ignore_label:
                    label_ids.append(self._ignore_label)
                else:
                    label_ids.append(
                        self._b_to_i_label[self._label_to_id[labels[word_idx]]]
                        if self._label_all_tokens else self._ignore_label
                    )
            previous_word_idx = word_idx
        return label_ids
