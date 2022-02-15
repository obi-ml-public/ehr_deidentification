from typing import Mapping, Union, NoReturn


class NERPredictTokenLabels(object):
    """
    Assign a default label while creating the dataset for prediction.
    This is done since the sequence tagging code expects the input
    file to contain a labels field, hence we assign a default label
    to meet this requirement
    """

    def __init__(self, default_label: str) -> NoReturn:
        """
        Initialize the default label
        Args:
            default_label (str): Default label that will be used
        """
        # Keeps track of all the spans (list) in the text (note)
        self._default_label = default_label

    def get_labels(self, token: Mapping[str, Union[str, int]]) -> str:
        """
        Given a token, return the default label.
        Args:
            token (Mapping[str, Union[str, int]]): Contains the token text, start and end position of the token
                                                   in the text
        Returns:
            default_label (str): default label
        """
        return self._default_label
