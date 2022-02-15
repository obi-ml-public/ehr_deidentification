from collections import Counter, defaultdict
from typing import Sequence, Mapping, NoReturn


class NERDistribution(object):
    """
    Store the distribution of ner types based on some key.
    That is we store the NER type distribution for some given key value and we update
    the distribution when spans related to that key is passed
    """

    def __init__(self) -> NoReturn:
        """
        Initialize the NER type - count mapping
        """
        # Counter the captures the ner types and counts per patient/note_id in the dataset
        # Depending on what we set the group_key as. Basically gather counts with respect
        # to some grouping of the notes
        # E.g - {{PATIENT 1: {AGE: 99, DATE: 55, ...}, {PATIENT 2: {AGE: 5, DATE: 9, ...} ... }
        self._ner_distribution = defaultdict(Counter)

    def update_distribution(self, spans: Sequence[Mapping[str, str]], key: str) -> NoReturn:
        """
        Update the distribution of ner types for the given key
        Args:
            spans (Sequence[Mapping[str, str]]): The list of spans in the note
            key (str): The note id or patient id of the note (some grouping)
        """
        # Go through the spans in the note and compute the ner distribution
        # Compute both the overall ner distribution and ner distribution per
        # patient (i.e the ner types in all the notes associated with the patient)
        if not self._ner_distribution.get(key, False):
            self._ner_distribution[key] = Counter()
        for span in spans:
            self._ner_distribution[key][span['label']] += 1

    def get_ner_distribution(self) -> defaultdict:
        """
        Return overall ner distribution. The NER type distribution for every key.
        Returns:
            ner_distribution (defaultdict(Counter)): Overall NER type distribution for all keys
        """
        return self._ner_distribution

    def get_group_distribution(self, key: str) -> Counter:
        """
        Return the NER type distribution for the given key
        Returns:
            (Counter): ner distribution w.r.t some grouping (key)
        """
        if key in self._ner_distribution.keys():
            return self._ner_distribution[key]
        else:
            raise ValueError('Key not found')
