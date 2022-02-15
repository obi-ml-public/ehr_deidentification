import json
import random
from argparse import ArgumentParser
from typing import Union, NoReturn, Iterable, Dict, List

random.seed(41)


class SpanValidation(object):
    """
    This class is used to build  a mapping between the note id
    and the annotated spans in that note. This will be used during the
    evaluation of the models. This is required to perform span level
    evaluation.
    """
    @staticmethod
    def get_spans(
            input_file: str,
            metadata_key: str = 'meta',
            note_id_key: str = 'note_id',
            spans_key: str = 'spans'
    ):
        """
        Get a mapping between the note id
        and the annotated spans in that note. This will mainly be used during the
        evaluation of the models.
        Args:
            input_file (str): The input file
            metadata_key (str): The key where the note metadata is present
            note_id_key (str): The key where the note id is present
            spans_key (str): The key that contains the annotated spans for a note dictionary
        Returns:
            (Iterable[Dict[str, Union[str, List[Dict[str, str]]]]]): An iterable that iterates through each note
                                                                     and contains the note id and annotated spans
                                                                     for that note
        """
        # Read the input files (data source)
        for line in open(input_file, 'r'):
            note = json.loads(line)
            note_id = note[metadata_key][note_id_key]
            # Store the note_id and the annotated spans
            note[spans_key].sort(key=lambda _span: (_span['start'], _span['end']))
            yield {'note_id': note_id, 'note_spans': note[spans_key]}


def main() -> NoReturn:
    cli_parser = ArgumentParser(description='configuration arguments provided at run time from the CLI')
    cli_parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='the the jsonl file that contains the notes'
    )
    cli_parser.add_argument(
        '--metadata_key',
        type=str,
        default='meta',
        help='the key where the note metadata is present in the json object'
    )
    cli_parser.add_argument(
        '--note_id_key',
        type=str,
        default='note_id',
        help='the key where the note id is present in the json object'
    )
    cli_parser.add_argument(
        '--spans_key',
        type=str,
        default='spans',
        help='the key where the annotated spans for the notes are present in the json object'
    )
    cli_parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='the file where the note id and the corresponding spans for that note are to be saved'
    )
    args = cli_parser.parse_args()

    # Write the dataset to the output file
    with open(args.output_file, 'w') as file:
        for span_info in SpanValidation.get_spans(
                input_file=args.input_file,
                metadata_key=args.metadata_key,
                note_id_key=args.note_id_key,
                spans_key=args.spans_key):
            file.write(json.dumps(span_info) + '\n')


if __name__ == "__main__":
    main()

