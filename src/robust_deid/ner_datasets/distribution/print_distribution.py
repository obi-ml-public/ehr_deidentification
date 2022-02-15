from collections import Counter
from typing import Sequence, NoReturn

from .ner_distribution import NERDistribution


class PrintDistribution(object):
    """
    This class is used to print the distribution of NER types
    """

    def __init__(self, ner_distribution: NERDistribution, key_counts: Counter) -> NoReturn:
        """
        Initialize
        Args:
            ner_distribution (NERDistribution): NERDistribution object that keeps track of the NER type distributions
            key_counts (Counter): Number of keys/groups (e.g note_ids, patient ids etc)
        """
        self._ner_distribution = ner_distribution
        self._key_counts = key_counts

    def split_distribution(self, split: str, split_info: Sequence[str]) -> NoReturn:
        """
        Print NER type distribution
        Args:
            split (str): The dataset split
            split_info (Sequence[str]): The keys belonging to that split
        """
        split_distribution = Counter()
        number_of_notes = 0
        for key in split_info:
            number_of_notes += self._key_counts[key]
            split_distribution.update(self._ner_distribution.get_group_distribution(key))
        total_ner = sum(split_distribution.values())
        percentages = {ner_type: float(count) / total_ner * 100 if total_ner else 0
                       for ner_type, count in split_distribution.items()}
        print('{:^70}'.format('============ ' + split.upper() + ' NER Distribution ============='))
        print('{:<20}{:<10}'.format('Number of Notes: ', number_of_notes))
        print('{:<20}{:<10}\n'.format('Number of Groups: ', len(split_info)))
        for ner_type, count in split_distribution.most_common():
            print('{:<10}{:<10}{:<5}{:<10}{:<5}{:<10}'.format(
                'NER Type: ', ner_type,
                'Count: ', count,
                'Percentage: ', '{:0.2f}'.format(percentages[ner_type]))
            )
        print('{:<10}{:<10}{:<5}{:<10}{:<5}{:<10}'.format(
            'NER Type:', 'TOTALS', 'Count: ', total_ner, 'Percentage: ', '{:0.2f}'.format(100))
        )
        print('\n')
