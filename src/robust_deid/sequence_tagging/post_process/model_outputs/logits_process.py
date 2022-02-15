from typing import Sequence, NoReturn, Tuple


class LogitsProcess(object):
    """
    Process the output of the model forward pass. The forward pass will return the predictions
    (e.g the logits), labels if present. We process the output and return the processed
    values based on the application. This script we return the final prediction as the
    argmax of the logits.
    """

    def __init__(self, label_list: Sequence[str]) -> NoReturn:
        """
        Initialize a label list where the position corresponds to a particular label. For example
        position 0 will correspond to B-DATE etc.
        Args:
            label_list (Sequence[str]): The list of NER labels
        """
        self._label_list = label_list

    def decode(
            self,
            predictions: Sequence[Sequence[Sequence[float]]],
            labels: Sequence[Sequence[int]]
    ) -> Tuple[Sequence[Sequence[Sequence[float]]], Sequence[Sequence[str]]]:
        """
        Decode the predictions and labels so that the evaluation function and prediction
        functions can use them accordingly. The predictions and labels are numbers (ids)
        of the labels, these will be converted back to the NER tags (B-AGE, I-DATE etc) using
        the label_list. In this function we just take the argmax of the logits (scores) of the predictions
        Also remove the predictions and labels on the subword and context tokens
        Args:
            predictions (Sequence[Sequence[Sequence[float]]]): The logits (scores for each tag) returned by the model
            labels (Sequence[Sequence[int]]): Gold standard labels
        Returns:
            true_predictions (Sequence[Sequence[str]]): The predicted NER tags
            true_labels (Sequence[Sequence[str]]): The gold standard NER tags
        """
        # Remove ignored index (special tokens)
        true_predictions = [
            [[float(value) for value in p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self._label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_predictions, true_labels
