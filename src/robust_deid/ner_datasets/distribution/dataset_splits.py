import random
from collections import Counter
from typing import NoReturn

from .ner_distribution import NERDistribution

random.seed(41)


class DatasetSplits(object):
    """
    Prepare dataset splits - training, validation & testing splits
    Compute ner distributions in the dataset. Based on this we assign
    notes to different splits and at the same time, we keep the distribution of
    NER types in each split similar. .
    Keep track of the split information - which notes are present in which split.
    The label distribution in each split, the number of notes in each split.
    """

    def __init__(
            self,
            ner_distribution: NERDistribution,
            train_proportion: int,
            validation_proportion: int,
            test_proportion: int,
            margin: float
    ) -> NoReturn:
        """
        Maintain split information. Assign notes based on the proportion of
        the splits, while keeping the label distribution in each split similar.
        Keep track of the split information - which notes are present in which split.
        The label distribution in each split, the number of notes in each split.
        Keep track of the dataset splits and the counts in each split etc.
        These will be used to assign the different notes to different
        splits while keeping the proportion of ner similar in each split.
        Get the maximum number of ner that can be present in the train,
        validation and test split. The total count will be used to
        calculate the current proportion of ner in the split. This can be used
        to keep the proportion of ner types consistent among different splits
        Args:
            ner_distribution (NERDistribution): The NER distribution in the dataset
            train_proportion (int): Ratio of train dataset
            validation_proportion (int): Ratio of validation dataset
            test_proportion (int): Ratio of test dataset
            margin (float): Margin by which the label distribution can be exceeded in the split
        """
        self._ner_distribution = ner_distribution
        # Compute the counts of NER types in the entire dataset
        total_distribution = Counter()
        for key, counts in ner_distribution.get_ner_distribution().items():
            for label, count in counts.items():
                total_distribution[label] += count
        # Compute the percentages of NER types in the entire dataset
        self._total_ner = sum(total_distribution.values())
        self._label_dist_percentages = {
            ner_type: float(count) / self._total_ner * 100 if self._total_ner else 0
            for ner_type, count in total_distribution.items()
        }
        self._margin = margin
        # The three splits
        self._splits = ['train', 'validation', 'test']
        self._split_weights = None
        self._splits_info = None
        # Keep track of the patient_ids that have been processed.
        # Since a patient can have multiple notes and we already know the
        # ner distribution for this patient across all the notes (i.e the ner types
        # and count that appear in all the notes associated with this patient)
        # We also keep all the notes associated with a patient in the same split
        # So we check if adding all the notes associated with this patient will
        # disturb the ner distribution (proportions) as mentioned before.
        self._processed_keys = dict()
        # Based on these proportions we compute train_ner_count, validation_ner_count, test_ner_count
        # Say the proportion are 85, 10, 5
        # The train split will have a maximum of 85% of the overall ner, validation will have 10 and test will 5
        # That is if there are total count of all ner is 100, on splitting the datasets
        # the train split will have a total of 85 ner, validation split will have a total of 10 ner and the
        # test split will have a total of 5 ner
        train_ner_count = int(train_proportion * self._total_ner / 100)
        validation_ner_count = int(validation_proportion * self._total_ner / 100)
        test_ner_count = int(test_proportion * self._total_ner / 100)
        # So based on this, we check if adding a note keeps the balance in proportion or not
        # If it does not, we check the splits given in the "remain" field of the dict (which is
        # the 2 other splits
        self._split_weights = [train_proportion, validation_proportion, test_proportion]
        # Based on the split proportions, ner counts and ner distribution
        # we need to split our dataset into train, validation and test split
        # For each split we try and maintain the same distribution (proportions) between ner types
        # that we computed from the entire dataset (given by - ner_distribution)
        # If the entire dataset had AGE:50%, DATE:30%, LOC:20%, we want the same proportions
        # in each of the train, validation and test splits
        # So based on this, we check if adding a note keeps the balance in proportion or not
        # If it does not, we check the splits given in the "remain" field of the dict (which is
        # the 2 other splits
        self._splits_info = {'train': {'remain': ['validation', 'test'],
                                       'total': train_ner_count,
                                       'remain_weights': [validation_proportion, test_proportion],
                                       'groups': list(), 'number_of_notes': 0, 'label_dist': Counter()},
                             'validation': {'remain': ['train', 'test'],
                                            'total': validation_ner_count,
                                            'remain_weights': [train_proportion, test_proportion],
                                            'groups': list(), 'number_of_notes': 0, 'label_dist': Counter()},
                             'test': {'remain': ['validation', 'train'],
                                      'total': test_ner_count,
                                      'remain_weights': [validation_proportion, train_proportion],
                                      'groups': list(), 'number_of_notes': 0, 'label_dist': Counter()}}

    def __set_split(self, split: str) -> NoReturn:
        """
        Set the split that you are currently checking/processing. 
        Based on the split you can perform certain checks and 
        computation for that split.
        Args:
            split (str): The split - train, validation or test
        """
        self._split = split

    def __update_label_dist(self, distribution: Counter) -> NoReturn:
        """
        Once we have determined that a note can be added to the split we need to 
        update the current count of the ner types in the split. So we pass the ner counts 
        in the note that will be updated and update the counts of the ner types in the split.
        Args:
            distribution (Counter): Contains the ner type and it's counts (distribution)
        """
        self._splits_info[self._split]['label_dist'].update(distribution)

    def __update_groups(self, note_group_key: str) -> NoReturn:
        """
        Once we have determined that a note can be added to the split, we append
        to a list some distinct element of the note (e.g note_id). This list will
        contain the note_ids of the notes that belong to this split.
        Args:
            note_group_key (str): Contains the note metadata - e.g note_id, institute etc
        """
        self._processed_keys[note_group_key] = self._split
        self._splits_info[self._split]['groups'].append(note_group_key)

    def __check_split(self, distribution: Counter) -> bool:
        """
        This function is used to check the resulting ner distribution in the split on adding this
        note to the split. We check how the proportion of ner changes if this note is added to
        the split. If the proportion exceeds the desired proportion then we return false
        to indicate that adding this note will upset the ner distribution across splits, so we should
        instead check adding this note to another split. If it does not update the balance then we return
        True, which means we can add this note to this split. The desired proportion of ner is passed
        in the percentages argument - where we have the desired proportion for each ner type.
        Args:
            distribution (Counter): Contains the mapping between ner type and count
        Returns:
            (bool): True if the note can be added to the split, false otherwise
        """
        # Get the current ner types and counts in the split
        split_label_dist = self._splits_info[self._split]['label_dist']
        # Get the max ner count that can be present in the split
        # This will be used to compute the ner proportions in the split
        split_total = self._splits_info[self._split]['total']
        # Check if the proportion of the split picked in zero
        # and return False because we cant add any note to this split
        if split_total == 0:
            return False
        for ner_type, count in distribution.items():
            percentage = (split_label_dist.get(ner_type, 0) + count) / split_total * 100
            # Check if the proportion on adding this note exceeds the desired proportion
            # within the margin of error
            # If it does return false
            if percentage > self._label_dist_percentages[ner_type] + self._margin:
                return False
        return True

    def get_split(self, key: str) -> str:
        """
        Assign a split to the note - based on the distribution of ner types in the note
        and the distribution of ner types in the split. Essentially assign a note to a split
        such that the distribution of ner types in each split is similar, once all notes have
        been assigned to their respective splits.
        Args:
            key (str): The note id or patient id of the note (some grouping key)
        Returns:
            (str): The split
        """
        current_splits = self._splits
        current_weights = self._split_weights
        distribution = self._ner_distribution.get_group_distribution(key=key)
        if self._processed_keys.get(key, False):
            return self._processed_keys[key]
        while True:
            # Pick and set the split
            check_split = random.choices(current_splits, current_weights)[0]
            self.__set_split(check_split)
            # Get the ner distribution for this particular patient (across all the notes associated
            # with this patient) and check if the notes can be added to this split.
            # The margin of error for the ner proportions. As we said above we try and keep the proportions
            # across the splits the same, but we allow for some flexibility, so we can go +- the amount
            # given by margin.
            include = self.__check_split(distribution=distribution)
            if include:
                self.__update_groups(key)
                self.__update_label_dist(distribution=distribution)
                return check_split
            else:
                # Check the two other possible splits
                if len(current_splits) == 3:
                    current_splits = self._splits_info[check_split]['remain']
                    current_weights = self._splits_info[check_split]['remain_weights']
                    # Check the one other possible split (when the one of the above two other split check returns false)
                elif len(current_splits) == 2 and current_weights[1 - current_splits.index(check_split)] != 0:
                    index = current_splits.index(check_split)
                    current_splits = [current_splits[1 - index]]
                    current_weights = [100]
                # If it can't be added to any split - choose a split randomly
                else:
                    current_splits = self._splits
                    current_weights = self._split_weights
                    check_split = random.choices(current_splits, current_weights)[0]
                    self.__set_split(check_split)
                    self.__update_groups(key)
                    self.__update_label_dist(distribution=distribution)
                    return check_split
