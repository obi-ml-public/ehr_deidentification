from collections import defaultdict
from typing import Sequence, Mapping, NoReturn, List, Union


class NoteLevelAggregator(object):
    """
    The input while training the model is at a sentence level. What happens is we
    have a bunch of notes (say 20) which we split into sentences and tokenize, so
    we end up with tokenized sentences (say 400). Each sentence is then used as a
    training example. Now this list of sentences is shuffled and the model is trained.
    For evaluation and prediction however we want to know which sentence belong to
    which note since we want go back from the sentence to the note level. This class
    basically aggregates sentence level information back to the note level. So that we 
    can do evaluation at the note level and get aggregate predictions for the entire note.
    To perform this we keep track of all the note_ids - [ID1, ID2 ...]. We sue this list
    as a reference - so when we return predictions we return a list [[P1], [P2] ..] where
    P1 corresponds to the predictions for the note with note id ID1.
    """

    def __init__(
            self,
            note_ids: Sequence[str],
            note_sent_info: Sequence[Mapping[str, Union[str, int]]]
    ) -> NoReturn:
        """
        Initialize the reference note_ids, this list and the position of the note_id in this list
        is used as reference when aggregating predictions/tokens belonging to a note.
        The note_ids are used as references for the functions below.
        Args:
            note_ids (Sequence[str]): The sequence of note_ids to use as reference
            note_sent_info (Sequence[Mapping[str, Union[str, int]]]): The information for each sentence
                                                                      (training example) it contains which note_id
                                                                      the sentence belongs to and the start and end
                                                                      position of that sentence in the note
        """
        self._note_ids = note_ids
        self._note_index_map = self.__get_note_index_map(note_sent_info)
        check_len = len([index for note_index in self._note_index_map for index in note_index])
        check_len_unique = len(set([index for note_index in self._note_index_map for index in note_index]))
        if len(note_sent_info) != check_len or check_len != check_len_unique:
            raise ValueError('Length mismatch')

    @staticmethod
    def __get_note_aggregate(note_sent_info: Sequence[Mapping[str, Union[str, int]]]) -> defaultdict(list):
        """
        Return a mapping where the key is the note_id and the value is a sequence that
        contain the sentence information. For example 'id1':[{index=8, start:0, end:30},
        {index=80, start:35, end:70}, {index=2, start:71, end:100} ..]
        What this mapping is saying that for this note_id, the first sentence in the note
        is the 8th sentence in the dataset, the second sentence in the note is the 80th
        sentence in the dataset and the third sentence is the 2nd sentence in the dataset.
        This is because the dataset can be shuffled.
        Args:
            note_sent_info (Sequence[Mapping[str, Union[str, int]]]): The information for each sentence
                                                                      (training example) it contains which note_id
                                                                      the sentence belongs to and the start and end
                                                                      position of that sentence in the note
        Returns:
            note_aggregate (defaultdict(list)): Contains the note_id to sentence (train example)
                                                mapping with respect to its position with the dataset
        """
        note_aggregate = defaultdict(list)
        for index, note_sent in enumerate(note_sent_info):
            note_id = note_sent['note_id']
            start = note_sent['start']
            end = note_sent['end']
            note_aggregate[note_id].append({'index': index, 'start': start, 'end': end})
        # Sort the sentences/training example based on its start position in the note
        for note_id, aggregate_info in note_aggregate.items():
            aggregate_info.sort(key=lambda info: info['start'])
        return note_aggregate

    def __get_note_index_map(self, note_sent_info: Sequence[Mapping[str, Union[str, int]]]) -> List[List[int]]:
        """
        Return a sequence that contains a sequence within which contains the sentence position w.r.t to the dataset.
        for that note (the note being note_id_1 for position 1)
        For example we have note_ids=[i1, i2, i3, ...]
        This function will return [[8, 80, 2 ..], [7, 89, 9], [1, 3, 5, ...]
        Where position 1 corresponds to ID - i1 and we say that the 8th, 80th and 2nd
        sentence in the dataset correspond to the sentences in the note i1 (in sorted order).
        For position 2, its ID - i2 and we say that the 7, 89, 9 sentence (training example)
        in the dataset correspond to the sentences in the note i2 (in sorted order).
        Remember the dataset can be shuffled.
        Args:
            note_sent_info (Sequence[Mapping[str, Union[str, int]]]): The information for each sentence
                                                                      (training example) it contains which note_id
                                                                      the sentence belongs to and the start and end
                                                                      position of that sentence in the note
        Returns:
            List[List[int]]: Return a sequence that contains a sequence within which contains
                             the sentence position w.r.t to the dataset for that note
                             (the note being note_id_1 for position 1)
        """
        note_aggregate = NoteLevelAggregator.__get_note_aggregate(note_sent_info)
        return [[note_agg['index'] for note_agg in note_aggregate.get(note_id, None)] for note_id in self._note_ids]

    def get_aggregate_sequences(
            self,
            sequences: Union[Sequence[Sequence[str]], Sequence[Sequence[Mapping[str, Union[str, int]]]]]
    ) -> List[List[str]]:
        """
        Return a sequence that contains a sequence within which contains the tokens or labels.
        for that note (the note being note_id_1 for position 1)
        For example we have note_ids=[i1, i2, i3, ...]
        This function will return [[PREDICTIONS -i1 ...], [PREDICTIONS -i2 ...], [PREDICTIONS -i3 ...]
        Where position 1 corresponds to ID - i1 and it contains the following predictions
        that are present in the note i1 (in sorted order).
        Where position 2 corresponds to ID - i2 and it contains the following predictions
        that are present in the note i2 (in sorted order).
        Return a sequence that contains a sequence within which contains the sentence position w.r.t to the dataset.
        for that note (the note being note_id_1 for position 1)
        For example we have note_ids=[i1, i2, i3, ...]
        This function will return [[8, 80, 2 ..], [7, 89, 9], [1, 3, 5, ...]
        Where position 1 corresponds to ID - i1 and we say that the 8th, 80th and 2nd
        sentence in the dataset correspond to the sentences in the note i1 (in sorted order).
        For position 2, its ID - i2 and we say that the 7, 89, 9 sentence (training example)
        in the dataset correspond to the sentences in the note i2 (in sorted order).
        Remember the dataset can be shuffled.
        Args:
            sequences (Union[Sequence[Sequence[str]], Sequence[Sequence[Mapping[str, Union[str, int]]]]]): The sequence
                                                                                                           of tokens or
                                                                                                           labels
        Returns:
            List[List[int]]: Return a sequence that contains a sequence within which contains
                             the predictions for that note (the note being note_id_1 for position 1)
        """
        return [[sequence for index in note_index for sequence in sequences[index]] for note_index in
                self._note_index_map]
