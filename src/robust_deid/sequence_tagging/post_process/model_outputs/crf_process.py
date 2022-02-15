from typing import Sequence, NoReturn

import torch

from .utils import check_consistent_length


class CRFProcess(object):

    def __init__(
            self,
            label_list: Sequence[str],
            top_k: int
    ) -> NoReturn:
        """
        Initialize a label list where the position corresponds to a particular label. For example
        position 0 will correspond to B-DATE etc. top k will return the top k CRF sequences
        Args:
            label_list (Sequence[str]): The list of NER labels
            top_k (int): The number of top CRF sequences to return
        """
        self._label_list = label_list
        self._top_k = top_k
        self._crf = None

    def set_crf(self, crf):
        """
        Store the CRF layer used while training the model
        Args:
            crf (): Set the CRF layer - this contains the CRF weights (NER transition weights)
        """
        self._crf = crf

    def process_sequences(
            self,
            sequences: Sequence[Sequence[str]],
            scores: Sequence[float]
    ) -> NoReturn:
        """
        The function will be implemented by the sub class and will return a sequence of NER
        predictions based on the implemented function
        Args:
            sequences (Sequence[Sequence[str]]): The list of possible sequences from the CRF layer
            scores (Sequence[float]): The scores for the sequences
        """
        raise NotImplementedError

    def decode(
            self,
            predictions: Sequence[Sequence[Sequence[float]]],
            labels: Sequence[Sequence[int]]
    ):
        """
        Decode the predictions and labels so that the evaluation function and prediction
        functions can use them accordingly. The predictions and labels are numbers (ids)
        of the labels, these will be converted back to the NER tags (B-AGE, I-DATE etc) using
        the label_list. In this function we process the CRF sequences and their scores and 
        select the NER sequence based on the implementation of the process_sequences function
        Also remove the predictions and labels on the subword and context tokens
        Args:
            predictions (: Sequence[Sequence[str]]): The logits (scores for each tag) returned by the model
            labels (Sequence[Sequence[str]]): Gold standard labels
        Returns:
            true_predictions (Sequence[Sequence[str]]): The predicted NER tags
            true_labels (Sequence[Sequence[str]]): The gold standard NER tags
        """
        # Check if the CRF layer has been initialized
        if self._crf is None:
            raise ValueError('CRF layer not initialized/set - use the set_crf function to set it')
        # Convert to a torch tensor, since the CRF layer expects a torch tensor
        logits = torch.tensor(predictions)
        labels_tensor = torch.tensor(labels)
        output_tags = list()
        # Get the CRF outputs
        # Process the top K outputs based and store the processed sequence
        # based on process_sequences function
        for seq_logits, seq_labels in zip(logits, labels_tensor):
            seq_mask = seq_labels != -100
            seq_logits_crf = seq_logits[seq_mask].unsqueeze(0)
            tags = self._crf.viterbi_tags(seq_logits_crf, top_k=self._top_k)
            # Unpack "batch" results
            if self._top_k is None:
                sequences = [tag[0] for tag in tags]
                scores = [tag[1] for tag in tags]
            else:
                sequences = [tag[0] for tag in tags[0]]
                scores = [tag[1] for tag in tags[0]]
            output_tags.append(self.process_sequences(sequences, scores))
            # Remove ignored index (special tokens)
        true_predictions = [
            [self._label_list[p] for p in prediction]
            for prediction in output_tags
        ]
        true_labels = [
            [self._label_list[l] for l in label if l != -100]
            for label in labels
        ]
        check_consistent_length(true_predictions, true_labels)
        return true_predictions, true_labels
