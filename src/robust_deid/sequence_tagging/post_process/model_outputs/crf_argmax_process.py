from typing import Sequence, NoReturn, List

from .crf_process import CRFProcess


class CRFArgmaxProcess(CRFProcess):

    def __init__(self, label_list: Sequence[str], top_k: int = None) -> NoReturn:
        """
        Initialize a label list where the position corresponds to a particular label. For example
        position 0 will correspond to B-DATE etc. top k will return the top k CRF sequences
        Args:
            label_list (Sequence[str]): The list of NER labels
            top_k (int): The number of top CRF sequences to return
        """
        super().__init__(label_list, top_k)

    def process_sequences(self, sequences: Sequence[Sequence[str]], scores: Sequence[float]) -> List[str]:
        """
        The function will get the top sequence given by the crf layer based on the CRF loss/score.
        Args:
            sequences (Sequence[Sequence[str]]): The list of possible sequences from the CRF layer
            scores (Sequence[float]): The scores for the sequences
        Returns:
            (List[str]): Highest scoring sequence of tags
        """
        return sequences[0]
