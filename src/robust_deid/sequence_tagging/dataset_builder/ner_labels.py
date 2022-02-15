from typing import Sequence, List, NoReturn, Dict


class NERLabels(object):
    """
    Prepare the labels that will be used by the model. Parse the NER types
    and prepare the NER labels. For example - NER Types: [AGE, DATE],
    it will create a list like this (for BIO notation) [B-AGE, I-AGE, B-DATE, I-DATE, O]
    These are the labels that will be assigned to the tokens based on the PHI type.
    Say we had the following NER types: NAME, AGE, HOSP
    The NER labels in the BIO notation would be B-AGE, B-HOSP, B-NAME, I-AGE, I-HOSP, I-NAME, O
    This script creates a list of the NER labels ([B-AGE, B-HOSP, B-NAME, I-AGE, I-HOSP, I-NAME, O])
    based on the NER types (NAME, AGE, HOSP) that have been defined. Labels have been sorted.
    The script also returns the number of labels, the label_to_id mapping, the id_to_label mapping
    Label_id_mapping: {B-AGE:0, B-HOSP:1, B-NAME:2, I-AGE:3, I-HOSP:4, I-NAME:5, O:6}
    This information will be used  during training, evaluation and prediction.
    """

    def __init__(self, notation: str, ner_types: Sequence[str]) -> NoReturn:
        """
        Initialize the notation that we are using for the NER task
        Args:
            notation (str): The notation that will be used for the NER labels
            ner_types (Sequence[str]): The list of NER categories
        """
        self._notation = notation
        self._ner_types = ner_types

    def get_label_list(self) -> List[str]:
        """
        Given the NER types return the NER labels.
        NER Types: [AGE, DATE] -> return a list like this (for BIO notation) [B-AGE, I-AGE, B-DATE, I-DATE, O]
        Returns:
            ner_labels (List[str]): The list of NER labels based on the NER notation (e.g BIO)
        """
        # Add the 'O' (Outside - Non-phi) label to the list
        if 'O' not in self._ner_types:
            ner_labels = ['O']
        else:
            ner_labels = list()
        # Go through each label and prefix it based on the notation (e.g - B, I etc)
        for ner_type in self._ner_types:
            for ner_tag in list(self._notation):
                if ner_tag != 'O':
                    ner_labels.append(ner_tag + '-' + ner_type)
        ner_labels.sort()
        return ner_labels

    def get_label_to_id(self) -> Dict[str, int]:
        """
        Return a label to id mapping
        Returns:
            label_to_id (Dict[str, int]): label to id mapping
        """
        labels = self.get_label_list()
        label_to_id = {label: index_id for index_id, label in enumerate(labels)}
        return label_to_id

    def get_id_to_label(self) -> Dict[int, str]:
        """
        Return a id to label mapping
        Returns:
            id_to_label (Dict[int, str]): id to label mapping
        """
        labels = self.get_label_list()
        id_to_label = {index_id: label for index_id, label in enumerate(labels)}
        return id_to_label
