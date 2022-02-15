from typing import List


def check_consistent_length(y_true: List[List[str]], y_pred: List[List[str]]):
    """
    Check that all arrays have consistent first and second dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Args:
        y_true : 2d array.
        y_pred : 2d array.
    """
    len_true = list(map(len, y_true))
    len_pred = list(map(len, y_pred))
    is_list = set(map(type, y_true)) | set(map(type, y_pred))

    if len(y_true) != len(y_pred) or len_true != len_pred:
        message = 'Found input variables with inconsistent numbers of samples:\n{}\n{}'.format(len_true, len_pred)
        raise ValueError(message)
