from typing import Sequence, NoReturn, Tuple

import numpy as np
from scipy.special import softmax

from .utils import check_consistent_length


class ThresholdProcessSum(object):
    """
    """

    def __init__(self, label_list: Sequence[str], threshold: float) -> NoReturn:
        """
        Initialize a label list where the posiion corresponds to a particular label. For example
        position 0 will correspond to B-DATE etc.
        Args:
            label_list (Sequence[str]): The list of NER labels
        """
        self._label_list = label_list
        self._threshold = threshold
        self._outside_label_index = self._label_list.index('O')
        self._mask = np.zeros((len(self._label_list)), dtype=bool)
        self._mask[self._outside_label_index] = True

    def get_masked_array(self, data):
        return np.ma.MaskedArray(data=data, mask=self._mask)

    def process_prediction(self, prediction):
        softmax_prob = softmax(prediction)
        masked_softmax_prob = self.get_masked_array(data=softmax_prob)
        if masked_softmax_prob.sum() >= self._threshold:
            return masked_softmax_prob.argmax()
        else:
            return self._outside_label_index

    def decode(
            self,
            predictions: Sequence[Sequence[Sequence[float]]],
            labels: Sequence[Sequence[int]]
    ) -> Tuple[Sequence[Sequence[str]], Sequence[Sequence[str]]]:
        """
        Args:
            predictions (Sequence[Sequence[Sequence[float]]]): The logits (scores for each tag) returned by the model
            labels (Sequence[Sequence[int]]): Gold standard labels
        Returns:
            true_predictions (Sequence[Sequence[str]]): The predicted NER tags
            true_labels (Sequence[Sequence[str]]): The gold standard NER tags
        """
        # Remove ignored index (special tokens)
        true_predictions = [
            [self._label_list[self.process_prediction(p)] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self._label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        check_consistent_length(true_predictions, true_labels)
        return true_predictions, true_labels

