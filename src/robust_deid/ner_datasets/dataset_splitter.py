import json
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Counter
from typing import NoReturn, List

from .distribution import NERDistribution, DatasetSplits, PrintDistribution

random.seed(41)


class DatasetSplitter(object):
    """
    Prepare dataset splits - training, validation & testing splits
    Compute ner distributions in our dataset. Compute ner distributions
    based on which we create and store a dictionary which will contain
    information about which notes (in a dataset) belong to which split.
    Based on this distribution and whether we want to keep certain notes
    grouped (e.g by patient) we assign notes to a split, such that the
    final ner type distribution in each split is similar.
    """

    def __init__(
            self,
            train_proportion: int = 70,
            validation_proportion: int = 15,
            test_proportion: int = 15
    ) -> NoReturn:
        """
        Initialize the proportions of the splits.
        Args:
            train_proportion (int): Ratio of train dataset
            validation_proportion (int): Ratio of validation dataset
            test_proportion (int): Ratio of test dataset
        """
        self._train_proportion = train_proportion
        self._validation_proportion = validation_proportion
        self._test_proportion = test_proportion
        self._split = None
        self._lookup_split = dict()

    def get_split(self, split: str) -> List[str]:
        return [key for key in self._lookup_split[split].keys()]

    def set_split(self, split: str) -> NoReturn:
        """
        Set the split that you are currently checking/processing.
        Based on the split you can perform certain checks and
        computation. Once the split is set, read the information
        present in the split_info_path. Extract only the information
        belonging to the split. Create a hash map where we have
        the keys as the note_ids/patient ids that belong to the split. This hashmap
        can then be used to check if a particular note belongs to this
        split.
        Args:
            split (str): The split - train, test etc (depends on how you named it)
        """
        if split not in ['train', 'validation', 'test']:
            raise ValueError('Invalid split')
        self._split = split

    def __update_split(self, key: str) -> NoReturn:
        """
        Update the hash map where we have
        the keys (e.g note_id) that belong to the split. This hashmap
        can then be used to check if a particular note belongs to this
        split.
        Args:
            key (str): The key that identify the note belonging to the split
        """
        self._lookup_split[self._split][key] = 1

    def check_note(self, key: str) -> bool:
        """
        Use the hash map created in the __get_i2b2_filter_map function
        to check if the note (note_info) belongs to this split (train,
        val, test etc). If it does, return true, else false
        Args:
            key (str): The key that identify the note belonging to the split
        Returns:
            (bool): True if the note belongs to the split, false otherwise
        """
        if self._split is None:
            raise ValueError('Split not set')
        if self._lookup_split[self._split].get(key, False):
            return True
        else:
            return False

    def assign_splits(
            self,
            input_file: str,
            spans_key: str = 'spans',
            metadata_key: str = 'meta',
            group_key: str = 'note_id',
            margin: float = 0.3
    ) -> NoReturn:
        """
        Get the dataset splits - training, validation & testing splits
        Based on the NER distribution and whether we want to keep certain
        notes grouped (e.g by patient). Return an iterable that contains
        a tuple that contains the note_id and the split. This can be used
        to filter notes based on the splits.
        Args:
            input_file (str): The input file
            spans_key (str): The key where the note spans are present
            metadata_key (str): The key where the note metadata is present
            group_key (str): The key where the note group (e.g note_id or patient id etc) is present.
                             This field is what the notes will be grouped by, and all notes belonging
                             to this grouping will be in the same split
            margin (float): Margin of error when maintaining proportions in the splits
        """
        # Compute the distribution of NER types in the grouped notes.
        # For example the distribution of NER types in all notes belonging to a
        # particular patient
        self._lookup_split = {
            'train': dict(),
            'validation': dict(),
            'test': dict()
        }
        ner_distribution = NERDistribution()
        for line in open(input_file, 'r'):
            note = json.loads(line)
            key = note[metadata_key][group_key]
            ner_distribution.update_distribution(spans=note[spans_key], key=key)
        # Initialize the dataset splits object
        dataset_splits = DatasetSplits(
            ner_distribution=ner_distribution,
            train_proportion=self._train_proportion,
            validation_proportion=self._validation_proportion,
            test_proportion=self._test_proportion,
            margin=margin
        )
        # Check the note and assign it to a split
        for line in open(input_file, 'r'):
            note = json.loads(line)
            key = note[metadata_key][group_key]
            split = dataset_splits.get_split(key=key)
            self.set_split(split)
            self.__update_split(key)
        return None


def main() -> NoReturn:
    """
    Prepare dataset splits - training, validation & testing splits
    Compute ner distributions in our dataset. Based on this distribution
    and whether we want to keep certain notes grouped (e.g by patient)
    we assign notes to a split, such that the final ner type distribution 
    in each split is similar.
    """
    # Compute the distribution of NER types in the grouped notes.
    # For example the distribution of NER types in all notes belonging to a
    # particular patient
    # The following code sets up the arguments to be passed via CLI or via a JSON file
    cli_parser = ArgumentParser(
        description='configuration arguments provided at run time from the CLI',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    cli_parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='the the jsonl file that contains the notes'
    )
    cli_parser.add_argument(
        '--spans_key',
        type=str,
        default='spans',
        help='the key where the note spans is present in the json object'
    )
    cli_parser.add_argument(
        '--metadata_key',
        type=str,
        default='meta',
        help='the key where the note metadata is present in the json object'
    )
    cli_parser.add_argument(
        '--group_key',
        type=str,
        default='note_id',
        help='the key to group notes by in the json object'
    )
    cli_parser.add_argument(
        '--train_proportion',
        type=int,
        default=70,
        help='ratio of train dataset'
    )
    cli_parser.add_argument(
        '--train_file',
        type=str,
        default=None,
        help='The file to store the train data'
    )
    cli_parser.add_argument(
        '--validation_proportion',
        type=int,
        default=15,
        help='ratio of validation dataset'
    )
    cli_parser.add_argument(
        '--validation_file',
        type=str,
        default=None,
        help='The file to store the validation data'
    )
    cli_parser.add_argument(
        '--test_proportion',
        type=int,
        default=15,
        help='ratio of test dataset'
    )
    cli_parser.add_argument(
        '--test_file',
        type=str,
        default=None,
        help='The file to store the test data'
    )
    cli_parser.add_argument(
        '--margin',
        type=float,
        default=0.3,
        help='margin of error when maintaining proportions in the splits'
    )
    cli_parser.add_argument(
        '--print_dist',
        action='store_true',
        help='whether to print the label distribution in the splits'
    )
    args = cli_parser.parse_args()
    dataset_splitter = DatasetSplitter(
        train_proportion=args.train_proportion,
        validation_proportion=args.validation_proportion,
        test_proportion=args.test_proportion
    )
    dataset_splitter.assign_splits(
        input_file=args.input_file,
        spans_key=args.spans_key,
        metadata_key=args.metadata_key,
        group_key=args.group_key,
        margin=args.margin
    )

    if args.train_proportion > 0:
        with open(args.train_file, 'w') as file:
            for line in open(args.input_file, 'r'):
                note = json.loads(line)
                key = note[args.metadata_key][args.group_key]
                dataset_splitter.set_split('train')
                if dataset_splitter.check_note(key):
                    file.write(json.dumps(note) + '\n')

    if args.validation_proportion > 0:
        with open(args.validation_file, 'w') as file:
            for line in open(args.input_file, 'r'):
                note = json.loads(line)
                key = note[args.metadata_key][args.group_key]
                dataset_splitter.set_split('validation')
                if dataset_splitter.check_note(key):
                    file.write(json.dumps(note) + '\n')

    if args.test_proportion > 0:
        with open(args.test_file, 'w') as file:
            for line in open(args.input_file, 'r'):
                note = json.loads(line)
                key = note[args.metadata_key][args.group_key]
                dataset_splitter.set_split('test')
                if dataset_splitter.check_note(key):
                    file.write(json.dumps(note) + '\n')

    if args.print_dist:
        # Read the dataset splits file and compute the NER type distribution
        key_counts = Counter()
        ner_distribution = NERDistribution()
        for line in open(args.input_file, 'r'):
            note = json.loads(line)
            key = note[args.metadata_key][args.group_key]
            key_counts[key] += 1
            ner_distribution.update_distribution(spans=note[args.spans_key], key=key)
        print_distribution = PrintDistribution(ner_distribution=ner_distribution, key_counts=key_counts)
        train_splits = dataset_splitter.get_split('train')
        validation_splits = dataset_splitter.get_split('validation')
        test_splits = dataset_splitter.get_split('test')
        all_splits = train_splits + validation_splits + test_splits
        # Print distribution for each split
        print_distribution.split_distribution(split='total', split_info=all_splits)
        print_distribution.split_distribution(split='train', split_info=train_splits)
        print_distribution.split_distribution(split='validation', split_info=validation_splits)
        print_distribution.split_distribution(split='test', split_info=test_splits)


if __name__ == "__main__":
    main()
