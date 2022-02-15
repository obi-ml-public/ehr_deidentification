from collections import deque
from typing import Deque, List, Sequence, Iterable, Optional, NoReturn, Dict, Mapping, Union, Tuple


class SentenceDataset(object):
    """
    When we mention previous sentence and next sentence, we don't mean exactly one sentence
    but rather a previous chunk and a next chunk. This can include one or more sentences and
    it does not mean that the sentence has to be complete (it can be cutoff in between) - hence a chunk
    This class is used to build a dataset at the sentence
    level. It takes as input all the tokenized sentences in the note. So the input is
    a list of lists where the outer list represents the sentences in the note and the inner list
    is a list of tokens in the sentence. It then returns a dataset where each sentence is 
    concatenated with the previous and a next chunk. This is done so that when we build a model
    we can use the previous and next chunks to add context to the sentence/model. The weights and loss etc
    will be computed and updated based on the current sentence. The previous and next chunks will
    only be used to add context. We could have different sizes of previous and next chunks
    depending on the position of the sentence etc. Essentially we build a sentence level dataset
    where we can also provide context to the sentence by including the previous and next chunks
    """

    def __init__(
            self,
            max_tokens: int,
            max_prev_sentence_token: int,
            max_next_sentence_token: int,
            default_chunk_size: int,
            ignore_label: str
    ) -> NoReturn:
        """
        Set the maximum token length a given training example (sentence level) can have.
        That is the total length of the current sentence + previous chunk + next chunk
        We also set the the maximum length of the previous and next chunks. That is how many
        tokens can be in these chunks. However if the total length exceeds, tokens in the
        previous and next chunks will be dropped to ensure that the total length is < max_tokens
        The default chunk size ensures that the length of the chunks will be a minimum number of
        tokens based on the value passed. For example is default_chunk_size=10, the length
        of the previous chunks and next chunks will be at least 10 tokens.
        Args:
            max_tokens (int): maximum token length a given training example (sentence level) can have
            max_prev_sentence_token (int): The max chunk size for the previous chunks for a given sentence 
                                           (training/prediction example) in the note can have
            max_next_sentence_token (int): The max chunk size for the next chunks for a given sentence 
                                           (training/prediction example) in the note can have
            default_chunk_size (int): the training example will always include a chunk of this length
                                      as part of the previous and next chunks
            ignore_label (str): The label assigned to the previous and next chunks to distinguish
                                from the current sentence
        """
        self._id_num = None
        self._max_tokens = max_tokens
        self._max_prev_sentence_token = max_prev_sentence_token
        self._max_next_sentence_token = max_next_sentence_token
        self._default_chunk_size = default_chunk_size
        self._ignore_label = ignore_label

    @staticmethod
    def chunker(
            seq: Sequence[Mapping[str, Union[str, int]]],
            size: int
    ) -> Iterable[Sequence[Mapping[str, Union[str, int]]]]:
        """
        Return chunks of the sequence. The size of each chunk will be based
        on the value passed to the size argument.
        Args:
            seq (Sequence): maximum token length a given training example (sentence level) can have
            size (int): The max chunk size for the chunks
        Return:
            (Iterable[Sequence[Mapping[str, Union[str, int]]]]): Iterable that iterates through fixed size chunks of
                                                                 the input sequence chunked version of the sequence
                         
        """
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def get_previous_sentences(self, sent_tokens: Sequence[Sequence[Mapping[str, Union[str, int]]]]) -> List[Deque]:
        """
        Go through all the sentences in the medical note and create a list of
        previous sentences. The output of this function will be a list of chunks
        where each index of the list contains the sentences (chunks) - (tokens) present before
        the sentence at that index in the medical note. For example prev_sent[0] will
        be empty since there is no sentence before the first sentence in the note
        prev_sent[1] will be equal to sent[0], that is the previous sentence of the
        second sentence will be the first sentence. We make use of deque, where we
        start to deque elements when it start to exceed max_prev_sentence_token. This
        list of previous sentences will be used to define the previous chunks
        Args:
            sent_tokens (Sequence[str]): Sentences in the note and 
                                         each element of the list contains a
                                         list of tokens in that sentence
        Returns:
            previous_sentences (List[deque]): A list of deque objects where each index contains a 
                                              list (queue) of previous tokens (chunk) with respect 
                                              to the sentence represented by that index in the note
        """
        previous_sentences = list()
        # Create a queue and specify the capacity of the queue
        # Tokens will be popped from the queue when the capacity is exceeded
        prev_sentence = deque(maxlen=self._max_prev_sentence_token)
        # The first previous chunk is empty since the first sentence in the note does not have
        # anything before it
        previous_sentences.append(prev_sentence.copy())
        # As we iterate through the list of sentences in the not, we add the tokens from the previous chunks
        # to the the queue. Since we have a queue, as soon as the capacity is exceeded we pop tokens from
        # the queue
        for sent_token in sent_tokens[:-1]:
            for token in sent_token:
                prev_sentence.append(token)
            # As soon as each sentence in the list is processed
            # We add a copy of the current queue to a list - this list keeps track of the
            # previous chunks for a sentence
            previous_sentences.append(prev_sentence.copy())

        return previous_sentences

    def get_next_sentences(self, sent_tokens: Sequence[Sequence[Mapping[str, Union[str, int]]]]) -> List[Deque]:
        """
        Go through all the sentences in the medical note and create a list of
        next sentences. The output of this function will be a list of lists
        where each index of the list contains the list of sentences present after
        the sentence at that index in the medical note. For example next_sent[-] will
        be empty since there is no sentence after the last sentence in the note
        next_sent[0] will be equal to sent[1:], that is the next sentence of the
        first sentence will be the subsequent sentences. We make use of deque, where we
        start to deque elements when it start to exceed max_next_sentence_token. This
        list of previous sentences will be used to define the previous chunks
        Args:
            sent_tokens (Sequence[str]): Sentences in the note and each 
                                         element of the list contains a
                                         list of tokens in that sentence
        Returns:
            next_sentences (List[deque]): A list of deque objects where each index contains a list (queue) 
                                          of next tokens (chunk) with respect to the sentence represented 
                                          by that index in the note
        """
        # A list of next sentences is first created and reversed
        next_sentences = list()
        # Create a queue and specify the capacity of the queue
        # Tokens will be popped from the queue when the capacity is exceeded
        next_sentence = deque(maxlen=self._max_next_sentence_token)
        # The first (which becomes the last chunk when we reverse this list) next chunk is empty since
        # the last sentence in the note does not have
        # anything after it
        next_sentences.append(next_sentence.copy())
        for sent_token in reversed(sent_tokens[1:]):
            for token in reversed(sent_token):
                next_sentence.appendleft(token)
            next_sentences.append(next_sentence.copy())
        # The list is reversed - since we went through the sentences in the reverse order in
        # the earlier steps
        return [next_sent for next_sent in reversed(next_sentences)]

    def get_sentences(
            self,
            sent_tokens: Sequence[Sequence[Mapping[str, Union[str, int]]]],
            token_text_key: str = 'text',
            label_key: str = 'label',
            start_chunk: Optional[Sequence[Mapping[str, Union[str, int]]]] = None,
            end_chunk: Optional[Sequence[Mapping[str, Union[str, int]]]] = None,
            sub: bool = False
    ) -> Iterable[Tuple[int, Dict[str, Union[List[Dict[str, Union[str, int]]], List[str]]]]]:
        """
        When we mention previous sentence and next sentence, we don't mean exactly one sentence
        but rather a previous chunk and a next chunk. This can include one or more sentences and
        it does not mean that the sentence has to be complete (it can be cutoff in between) - hence a chunk
        We iterate through all the tokenized sentences in the note. So the input is
        a list of lists where the outer list represents the sentences in the note and the inner list
        is a list of tokens in the sentence. It then returns a dataset where each sentence is
        concatenated with the previous and the next sentence. This is done so that when we build a model
        we can use the previous and next sentence to add context to the model. The weights and loss etc
        will be computed and updated based on the current sentence. The previous and next sentence will
        only be used to add context. We could have different sizes of previous and next chunks
        depending on the position of the sentence etc. Since we split a note in several sentences which are
        then used as training data.
        ignore_label is used to differentiate between the current sentence and the previous and next
        chunks. The chunks will have the label NA so that and the current sentence
        will have the label (DATE, AGE etc) so that they can be distinguished. 
        If however we are building a dataset for predictions
        the current sentence will have the default label O, but the next and previous chunks will still
        have the label NA. However if the total length exceeds, tokens in the
        previous and next chunks will be dropped to ensure that the total length is < max_tokens
        The default chunk size ensures that the length of the chunks will be a minimum number of
        tokens based on the value passed. For example is default_chunk_size=10, the length
        of the previous chunks and next chunks will be at least 10 tokens. If the total length > max tokens
        even after decreasing the sizes of the previous and next chunks, then we split this long 
        sentence into sub sentences and repeat the process described above.
        Args:
            sent_tokens (Sequence[Sequence[Mapping[str, Union[str, int]]]]): Sentences in the note and each sentence 
                                                                             contains the tokens (dict) in that sentence
                                                                             the token dict object contains the
                                                                             token text, start, end etc
            token_text_key (str): Each sentence contains a list of tokens where each token is a dict. We use the text
                                  key to extract the text of the token from the dictionary
            label_key (str): Each sentence contains a list of tokens where each token is a dict. We use the label_key
                             key to extract the label of the token from the dictionary. (if it does not have a label
                             the default label will be assigned)
            start_chunk (Optional[Sequence[Mapping[str, Union[str, int]]]]): Prefix the first sentence of with some
                                                                             pre-defined chunk
            end_chunk (Optional[Sequence[Mapping[str, Union[str, int]]]]): Suffix the last sentence of with some
                                                                            pre-defined chunk
            sub (bool): Whether the function is called to process sub-sentences (used when we are splitting 
                        long sentences into smaller sub sentences to keep sentence length < max_tokens
        Returns:
            (Iterable[Tuple[int, Dict[str, Union[List[Dict[str, Union[str, int]]], List[str]]]]]): Iterate through the
                                                                                                   returned sentences,
                                                                                                   where each sentence
                                                                                                   has the previous
                                                                                                   chunks and next
                                                                                                   chunks attached
                                                                                                   to it.
        """
        # Id num keeps track of the id of the sentence - that is the position the sentence occurs in
        # the note. We keep the id of sub sentences the same as the sentence, so that the user
        # knows that these sub sentences are chunked from a longer sentence.
        # <SENT 0> <SENT 1>. Say length of sent 0 with the previous and next chunks is less than max_tokens
        # we return sent 0 with id 0. For sent 1, say the length is longer, we split it into sub
        # sentences - <SUB 1><SUB 2> - we return SUB 1, and SUB 2 with id 1 - so we know that it belongs
        # to <SENT 1> in the note.
        if not sub:
            self._id_num = -1
        # Initialize the object that will take all the sentences in the note and return
        # a dataset where each row represents a sentence in the note. The sentence in each
        # row will also contain a previous chunk and next chunk (tokens) that will act as context
        # when training the mode
        # [ps1, ps 2, ps 3...ps-i], [cs1, cs2, ... cs-j], [ns, ns, ... ns-k] - as you can see the current sentence
        # which is the sentence we train on (or predict on) will be in the middle - the surrounding tokens will
        # provide context to the current sentence
        # Get the previous sentences (chunks) for each sentence in the note
        previous_sentences = self.get_previous_sentences(sent_tokens)
        # Get the next sentences (chunks) for each sentence in the note
        next_sentences = self.get_next_sentences(sent_tokens)
        # For the note we are going to iterate through all the sentences in the note and
        # concatenate each sentence with the previous and next chunks. (This forms the data that
        # will be used for training/predictions) Each sentence with the concatenated chunks will be
        # a training sample. We would do the same thing for getting predictions on a sentence as well
        # The only difference would be the labels that are used. We would use the default label O for
        # prediction and the annotated labels for prediction
        if len(sent_tokens) != len(previous_sentences) or len(sent_tokens) != len(next_sentences):
            raise ValueError('Sentence length mismatch')
        for index, (previous_sent, current_sent, next_sent) in enumerate(
                zip(previous_sentences, sent_tokens, next_sentences)):
            sent_tokens_text = list()
            sent_labels = list()
            sent_toks = list()
            # Get the tokens and labels for the current sentence
            for token in current_sent:
                # We store this, if we need to process sub sentences when a sentence exceeds max_tokens
                sent_toks.append(token)
                sent_tokens_text.append(token[token_text_key])
                sent_labels.append(token[label_key])
            # We check if the number of tokens in teh current sentence + previous chunk
            # + next chunk exceeds max tokens. If it does we start popping tokens from the previous and next chunks
            # until the number of tokens is equal to max tokens
            previous_sent_length = len(previous_sent)
            current_sent_length = len(sent_tokens_text)
            next_sent_length = len(next_sent)
            total_length = previous_sent_length + current_sent_length + next_sent_length
            # If the length of the current sentence plus the length of the previous and next
            # chunks exceeds the max_tokens, start popping tokens from the previous and next
            # chunks until either total length < max_tokens or the number of tokens in the previous and
            # next chunks goes below the default chunk size
            while total_length > self._max_tokens and \
                    (next_sent_length > self._default_chunk_size or previous_sent_length > self._default_chunk_size):
                if next_sent_length >= previous_sent_length:
                    next_sent.pop()
                    next_sent_length -= 1
                    total_length -= 1
                elif previous_sent_length > next_sent_length:
                    previous_sent.popleft()
                    previous_sent_length -= 1
                    total_length -= 1
            # If this is not a sub sentence, increment the ID to
            # indicate the processing of the next sentence of the note
            # If it is a sub sentence, keep the ID the same, to indicate
            # it belongs to a larger sentence
            if not sub:
                self._id_num += 1
            # If total length < max_tokens - process the sentence with the current sentence
            # and add on the previous and next chunks and return
            if total_length <= self._max_tokens:
                # Check if we want to add a pre-defined chunk for the first sentence in the note
                if index == 0 and start_chunk is not None:
                    previous_sent_tokens = [chunk[token_text_key] for chunk in start_chunk] + \
                                           [prev_token[token_text_key] for prev_token in list(previous_sent)]
                else:
                    previous_sent_tokens = [prev_token[token_text_key] for prev_token in list(previous_sent)]
                # Check if we want to add a pre-defined chunk for the last sentence in the note
                if index == len(sent_tokens) - 1 and end_chunk is not None:
                    next_sent_tokens = [next_token[token_text_key] for next_token in list(next_sent)] + \
                                       [chunk[token_text_key] for chunk in end_chunk]
                else:
                    next_sent_tokens = [next_token[token_text_key] for next_token in list(next_sent)]
                previous_sent_length = len(previous_sent_tokens)
                next_sent_length = len(next_sent_tokens)
                # Store information about the current sentence - start and end pos etc
                # this can be used to distinguish from the next and previous chunks
                # current_sent_info = {'token_info':current_sent}
                # Assign an different label (the ignore label) to the chunks - since they are used only for context
                previous_sent_labels = list()
                next_sent_labels = list()
                if self._ignore_label == 'NA':
                    previous_sent_labels = [self._ignore_label] * previous_sent_length
                    next_sent_labels = [self._ignore_label] * next_sent_length
                elif self._ignore_label == 'label':
                    if index == 0 and start_chunk is not None:
                        previous_sent_labels = [chunk[label_key] for chunk in start_chunk] + \
                                               [prev_token[label_key] for prev_token in list(previous_sent)]
                    else:
                        previous_sent_labels = [prev_token[label_key] for prev_token in list(previous_sent)]
                    if index == len(sent_tokens) - 1 and end_chunk is not None:
                        next_sent_labels = [next_token[label_key] for next_token in list(next_sent)] + \
                                           [chunk[label_key] for chunk in end_chunk]
                    else:
                        next_sent_labels = [next_token[label_key] for next_token in list(next_sent)]
                # Concatenate the chunks and the sentence
                # sent_tokens_text.append(token[token_text_key])
                tokens_data = previous_sent_tokens + sent_tokens_text + next_sent_tokens
                labels_data = previous_sent_labels + sent_labels + next_sent_labels
                # Return processed sentences
                yield self._id_num, {'tokens': tokens_data, 'labels': labels_data, 'current_sent_info': current_sent}
            # Process the sub sentences - we take a long sentence
            # and split it into smaller chunks - and we recursively call the function on this list
            # of smaller chunks - as mentioned before the smaller chunks (sub sentences) will have the
            # same ID as the original sentence
            else:
                # Store the smaller chunks - say <SENT1> is too long
                # <PREV CHUNK><SENT1><NEXT CHUNK>
                # We get chunk sent 1 - to <SUB1><SUB2><SUB3> and we pass this [<SUB1><SUB2><SUB3>] to the function
                # as a recursive call. This list is now processed as a smaller note that essentially belongs
                # to a sentence. But as you can see we did not pass <PREV CHUNK> & <NEXT CHUNK>, because
                # these are chunks that are not part of the current sentence, but they still need to be
                # included in the final output - and the work around is mentioned below
                # So that we have a previous chunk for <SUB1> and next chunk for <SUB3>
                # we include the previous_sent_tokens and next_sent_tokens as the start chunk
                # and the next chunk in the function call below
                # <PREV CHUNK><SUB1><NEXT SUB1>, id = x
                # <PREV SUB2><SUB2><NEXT SUB2>, id = x
                # <PREV SUB3><SUB3><NEXT CHUNK>, id = x
                sub_sentences = list()
                # Prefix the first sentence in these smaller chunks
                previous_sent_tokens = list(previous_sent)
                # Suffix the last sentence in these smaller chunks
                next_sent_tokens = list(next_sent)
                # Get chunks
                for chunk in SentenceDataset.chunker(sent_toks, self._max_tokens - (2 * self._default_chunk_size)):
                    sub_sentences.append(chunk)
                # Process list of smaller chunks
                for sub_sent in self.get_sentences(
                        sub_sentences,
                        token_text_key,
                        label_key,
                        start_chunk=previous_sent_tokens,
                        end_chunk=next_sent_tokens,
                        sub=True
                ):
                    yield sub_sent
