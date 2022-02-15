from typing import List, Sequence, Mapping, Optional, NoReturn, Dict, Union
from .ner_labels import NERLabels


class LabelMapper(object):
    """
    This class is used to map one set of NER labels to another set of NER labels
    For example we might want to map all NER labels to Binary HIPAA labels.
    E.g:
    We change the token labels - [B-AGE, O, O, U-LOC, B-DATE, L-DATE, O, B-STAFF, I-STAFF, L-STAFF] to
    [B-HIPAA, O, O, U-HIPAA, B-HIPAA, I-HIPAA, O, O, O, O] or if we wanted binary I2B2 labels we map it to
    [B-I2B2, O, O, U-I2B2, B-I2B2, -I2B2, O, B-I2B2, I-I2B2, L-I2B2]
    We do this mapping at the token and the span level. That is we have a span from says start=9, end=15
    labelled as LOC, we map this label to HIPAA or I2B2. This class maps an exisitng set of labels to
    another set of labels
    """

    def __init__(
            self,
            notation,
            ner_types: Sequence[str],
            ner_types_maps: Sequence[str],
            description: str
    ) -> NoReturn:
        """
        Initialize the variables that will be used to map the NER labels and spans
        The ner_map and spans_map should correspond to each other and contain the same NER types
        Args:
        """
        self._description = description
        if('O' in ner_types_maps):
            self._types = list(set(ner_types_maps) - set('O'))
        else:
            self._types = list(set(ner_types_maps))
        self._types.sort()
        self._spans_map = {ner_type: ner_type_map for ner_type, ner_type_map in zip(ner_types, ner_types_maps)}
        ner_labels = NERLabels(notation=notation, ner_types=ner_types)
        self._ner_map = dict()
        for label in ner_labels.get_label_list():
            if label == 'O' or self._spans_map[label[2:]] == 'O':
                self._ner_map[label] = 'O'
            else:
                self._ner_map[label] = label[0:2] + self._spans_map[label[2:]]

    def map_sequence(self, tag_sequence: Sequence[str]) -> List[str]:
        """
        Mapping a sequence of NER labels to another set of NER labels.
        E.g: If we use a binary HIPAA mapping
        This sequence [B-AGE, O, O, U-LOC, B-DATE, L-DATE, O, B-STAFF, I-STAFF, L-STAFF] will be mapped to
        [B-HIPAA, O, O, U-HIPAA, B-HIPAA, I-HIPAA, O, O, O, O]
        Return the original sequence if no mapping is used (i.e the maps are == None)
        Args:
            tag_sequence (Sequence[str]): A sequence of NER labels
        Returns:
            (List[str]): A mapped sequence of NER labels
        """
        # Return the original sequence if no mapping is used
        return [self._ner_map[tag] for tag in tag_sequence]

    def map_spans(self, spans: Sequence[Mapping[str, Union[str, int]]]) -> Sequence[Dict[str, Union[str, int]]]:
        """
        Mapping a sequence of NER spans to another set of NER spans.
        E.g: If we use a binary HIPAA mapping
        The spans: [{start:0, end:5, label: DATE}, {start:17, end:25, label: STAFF}, {start:43, end:54, label: PATIENT}]
        will be mapped to: [{start:0, end:5, label: HIPAA}, {start:17, end:25, label: O}, {start:43, end:54, label: HIPAA}]
        Return the original list of spans if no mapping is used (i.e the maps are == None)
        Args:
            spans (Sequence[Mapping[str, Union[str, int]]]): A sequence of NER spans
        Returns:
            (Sequence[Mapping[str, Union[str, int]]]): A mapped sequence of NER spans
        """
        return [{'start': span['start'], 'end': span['end'], 'label': self._spans_map[span['label']]} \
                for span in spans]

    def get_ner_description(self) -> str:
        """
        Get the description of the ner labels and span maps used
        Returns:
            (str): A description of the label/span maps used
        """
        return self._description

    def get_ner_types(self) -> List[str]:
        """
        Get the PHI types back from the list of NER labels
        [B-AGE, I-AGE, B-DATE, I-DATE ..] ---> [AGE, DATE, ...]
        Returns:
            ner_types (List[str]): The list of unique NER types
        """
        return self._types
