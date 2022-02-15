from typing import Sequence

from .argmax_process import ArgmaxProcess
from .crf_argmax_process import CRFArgmaxProcess
from .logits_process import LogitsProcess
from .threshold_process_max import ThresholdProcessMax
from .threshold_process_sum import ThresholdProcessSum


class PostProcessPicker(object):
    """
    This class is used to pick the post process layer that processed the model
    logits. The class provides functions that returns the desired post processor objects
    For example we can pick the argamx of the logits, where we just choose the highest scoring
    tag as the prediction for a token or we can use a crf layer to pick the highest 
    scoring sequence of tags
    """
    def __init__(self, label_list):
        """
        Initialize the NER label list
        Args:
            label_list (Sequence[str]): The NER labels. e.g B-DATE, I-DATE, B-AGE ...
        """
        self._label_list = label_list
        
    def get_argmax(self) -> ArgmaxProcess:
        """
        Return a post processor that uses argmax to process the model logits for obtaining the predictions
        Chooses the highest scoring tag.
        Returns:
            (ArgmaxProcess): Return argmax post processor
        """
        return ArgmaxProcess(self._label_list)
    
    def get_crf(self) -> CRFArgmaxProcess:
        """
        Return a post processor that uses CRF layer to process the model logits for obtaining the predictions
        Chooses the highest scoring sequence of tags based on the CRF layer
        Returns:
            (CRFArgmaxProcess): Return CRF layer post processor
        """
        return CRFArgmaxProcess(self._label_list)
    
    def get_logits(self) -> LogitsProcess:
        """
        Return a post processor that returns the model logits
        Returns:
            (LogitsProcess): Return Logits layer post processor
        """
        return LogitsProcess(self._label_list)
    
    def get_threshold_max(self, threshold) -> ThresholdProcessMax:
        """
        Return a post processor that uses a threshold (max) to process and return the model logits
        Returns:
            (ThresholdProcessMax): Return Threshold Max post processor
        """
        return ThresholdProcessMax(self._label_list, threshold=threshold)
    
    def get_threshold_sum(self, threshold) -> ThresholdProcessSum:
        """
        Return a post processor that uses a threshold (sum) to process and return the model logits
        Returns:
            (ThresholdProcessMax): Return Threshold Sum post processor
        """
        return ThresholdProcessSum(self._label_list, threshold=threshold)
