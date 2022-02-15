from typing import Sequence, Optional, NoReturn

from datasets import load_dataset, Dataset


class NERDataset(object):
    """
    This class is a wrapper around the huggingface datasets library
    It maintains the train, validation and test datasets based on the
    train, validation and test files passed by loading the dataset object
    from the file and provides a get function to access each of the datasets. 
    """

    def __init__(
            self,
            train_file: Optional[Sequence[str]] = None,
            validation_file: Optional[Sequence[str]] = None,
            test_file: Optional[Sequence[str]] = None,
            extension: str = 'json',
            shuffle: bool = True,
            seed: int = 41
    ) -> NoReturn:
        """
        Load the train, validation and test datasets from the files passed. Read the files and convert
        it into a huggingface dataset.
        Args:
            train_file (Optional[Sequence[str]]): The list of files that contain train data
            validation_file (Optional[Sequence[str]]): The list of files that contain validation data
            test_file (Optional[Sequence[str]]): The list of files that contain test data
            shuffle (bool): Whether to shuffle the dataset
            seed (int): Shuffle seed
       
        """
        self._datasets = NERDataset.__prepare_data(
            train_file,
            validation_file,
            test_file,
            extension,
            shuffle,
            seed
        )

    @staticmethod
    def __prepare_data(
            train_file: Optional[Sequence[str]],
            validation_file: Optional[Sequence[str]],
            test_file: Optional[Sequence[str]],
            extension: str,
            shuffle: bool,
            seed: int
    ) -> Dataset:
        """
        Get the train, validation and test datasets from the files passed. Read the files and convert
        it into a huggingface dataset.
        Args:
            train_file (Optional[Sequence[str]]): The list of files that contain train data
            validation_file (Optional[Sequence[str]]): The list of files that contain validation data
            test_file (Optional[Sequence[str]]): The list of files that contain test data
            shuffle (bool): Whether to shuffle the dataset
            seed (int): Shuffle seed
        Returns:
            (Dataset): The huggingface dataset with train, validation, test splits (if included)
        """
        # Read the datasets (train, validation, test etc).
        data_files = {}
        if train_file is not None:
            data_files['train'] = train_file
        if validation_file is not None:
            data_files['validation'] = validation_file
        if test_file is not None:
            data_files['test'] = test_file
        # Shuffle the dataset
        if shuffle:
            datasets = load_dataset(extension, data_files=data_files).shuffle(seed=seed)
        else:
            # Don't shuffle the dataset
            datasets = load_dataset(extension, data_files=data_files)
        return datasets

    def get_train_dataset(self) -> Dataset:
        """
        Return the train dataset
        Returns:
            (Dataset): The huggingface dataset - train split
        """
        return self._datasets['train']

    def get_validation_dataset(self) -> Dataset:
        """
        Return the validation dataset
        Returns:
            (Dataset): The huggingface dataset - validation split
        """
        return self._datasets['validation']

    def get_test_dataset(self) -> Dataset:
        """
        Return the test dataset
        Returns:
            (Dataset): The huggingface dataset - test split
        """
        return self._datasets['test']
