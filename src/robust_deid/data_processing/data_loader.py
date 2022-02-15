# Convert parquet file to jsonl file. Most of the scripts written in this
# project require jsonl file as input. This script helps when we need to
# convert a parquet file to a jsonl file.
import json
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Sequence, Dict, Iterable, Union, NoReturn, Optional


class DataLoader(object):
    """
    Convert parquet file to jsonl file. While some of the columns in the parquet file will
    be directly used as keys in the json object, some of the columns will be stored as metadata.
    The parquet_columns columns specify the columns from the parquet file that will be stored in the
    json object. The json_columns specify which columns will be stored directly as keys in
    the json object and metadata_columns columns specify which columns will be stored as metadata.
    The ordering in these lists need to match, because we do the above operations based on index
    positions.
    E.g - parquet_columns = ['NoteText', 'NoteID', 'PatientID', 'institution']
        - json_columns = ['text']
        - metadata_columns = ['note_id', 'patient_id', 'institute']
    NoteTXT corresponds to text, NoteID -> note_id, PatientID -> patient_id, institution -> institute
    As you can see we match based on positions. Once we process converted_columns, we process the
    metadata columns (i.e index is used for mapping parquet columns to jsonl keys).
    Hence it is important that the columns are specified in the right order.
    JSON Object: {'text': medical text, 'meta':{'note_id':12345, 'patient_id':54321, 'institute':PP}}
    """

    def __init__(
            self,
            parquet_columns: Optional[Sequence[str]] = None,
            json_columns: Optional[Sequence[str]] = None,
            metadata_columns: Optional[Sequence[str]] = None
    ) -> NoReturn:
        """
        Initialize the parquet column names and json object key names
        Args:
            parquet_columns (Optional[Sequence[str]]): Columns to extract from parquet file.
                                                       If not given - will assign ['NoteText', 'NoteID']
            json_columns (Optional[Sequence[str]]): Fields that will be stored directly in json object.
                                                    If not given - will assign ['text']
            metadata_columns (Optional[Sequence[str]]): Fields that will be stored as metadata in json object.
                                                        If not given - will assign ['note_id']
        """
        if metadata_columns is None:
            metadata_columns = ['note_id']
        if json_columns is None:
            json_columns = ['text']
        if parquet_columns is None:
            parquet_columns = ['NoteText', 'NoteID']
        self._parquet_columns = parquet_columns
        self._json_columns = json_columns
        self._metadata_columns = metadata_columns

    def load(self, input_file: str) -> Iterable[Dict[str, Union[str, Dict[str, str]]]]:
        """
        Read a parquet file, extract the relevant columns and create a json object for each row
        of the parquet file. This function will return an iterable that can be used to iterate
        through each of these json objects. The data and structure in the json objects will depend
        on how this class has been initialized (parquet_columns, json_columns, metadata_columns)
        JSON Object: {'text': medical text, 'meta':{'note_id':12345, 'patient_id':54321, 'institute':PP}}
        Args:
            input_file (str): Input parquet file
        Returns:
            (Iterable[Dict[str, Union[str, Dict[str, str]]]]): An iterable that iterates through the json objects
        """
        data = pd.read_parquet(input_file)
        data = data[self._parquet_columns]
        for data_load in data.itertuples():
            data_load_dict = {}
            index = 0
            for index, column in enumerate(self._json_columns):
                data_load_dict[column] = data_load[index + 1]
            index += 2
            data_load_dict['meta'] = {metadata_column: data_load[meta_index + index]
                                      for meta_index, metadata_column in enumerate(self._metadata_columns)}
            yield data_load_dict


def main() -> NoReturn:
    """
    Convert parquet file to jsonl file. While some of the columns in the parquet file will
    be directly used as keys in the json object, some of the column will be stored as metadata.
    The relevant columns specify the columns from the parquet file that will be stored in the
    json object. The converted_columns specify which columns will be stored directly as keys in
    the json object and metadata_columns columns specify which columns will be stored as metadata.
    The ordering in these lists need to match, because we do the above operations based on index
    positions. 
    E.g - relevant_columns = ['NoteTXT', 'NoteID', 'PatientID', 'institution']
        - converted_columns = ['text']
        - metadata_columns = ['note_id', 'patient_id', 'institute']
    NoteTXT corresponds to text, NoteID -> note_id, PatientID -> patient_id, institution -> institute
    As you can see we match based on positions. Once we process converted_columns, we process the 
    metadata columns (i.e index is used for mapping parquet columns to jsonl keys). 
    Hence it is important that the columns are specified in the right order.
    JSON Object: {'text': medical text, 'meta':{'note_id':12345, 'patient_id':54321, 'institute':PP}}
    """
    cli_parser = ArgumentParser(
        description='configuration arguments provided at run time from the CLI',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    # Take the first argument as a list instead of a file
    cli_parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='The input parquet file'
    )
    cli_parser.add_argument(
        '--parquet_columns',
        nargs="+",
        default=['NoteText', 'NoteID'],
        help='Columns to extract from parquet file. If not given - will assign [NoteText, NoteID]'
    )
    cli_parser.add_argument(
        '--json_columns',
        nargs="+",
        default=['text'],
        help='fields that will be stored directly in json object'
    )
    cli_parser.add_argument(
        '--metadata_columns',
        nargs="+",
        default=['note_id'],
        help='fields that will be stored as the metadata field in json object'
    )
    cli_parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='where to write the jsonl output'
    )
    args = cli_parser.parse_args()
    data_loader = DataLoader(
        parquet_columns=args.parquet_columns,
        json_columns=args.json_columns,
        metadata_columns=args.metadata_columns
    )
    notes = data_loader.load(input_file=args.input_file)
    # Write the jsonl output to the specified location
    with open(args.output_file, 'w') as file:
        for note in notes:
            if 'spans' not in note.keys():
                note['spans'] = []
            file.write(json.dumps(note) + '\n')
    return None


if __name__ == "__main__":
    main()
