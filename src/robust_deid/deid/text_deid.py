import json
import re
from argparse import ArgumentParser
from typing import Sequence, List, Tuple, Mapping, Union, Any, Type

import regex
from seqeval.scheme import IOB1, IOB2, IOBES, BILOU, Entities

from .utils import remove, replace_tag_type, replace_informative


class TextDeid(object):

    def __init__(self, notation, span_constraint):
        self._span_constraint = span_constraint
        if self._span_constraint == 'strict':
            self._scheme = TextDeid.__get_scheme('IO')
        elif self._span_constraint == 'super_strict':
            self._scheme = TextDeid.__get_scheme('IO')
        else:
            self._scheme = TextDeid.__get_scheme(notation)

    def decode(self, tokens, predictions):
        if self._span_constraint == 'exact':
            return predictions
        elif self._span_constraint == 'strict':
            return TextDeid.__get_relaxed_predictions(predictions)
        elif self._span_constraint == 'super_strict':
            return TextDeid.__get_super_relaxed_predictions(tokens, predictions)

    def get_predicted_entities_positions(
            self,
            tokens: Sequence[Mapping[str, Union[str, int]]],
            predictions: List[str],
            suffix: bool
    ) -> List[List[Union[Tuple[Union[str, int], Union[str, int]], Any]]]:
        """
        Use the seqeval get_entities method, which goes through the predictions and returns
        where the span starts and ends. - [O, O, B-AGE, I-AGE, O, O] this will return
        spans starts at token 2 and ends at token 3 - with type AGE. We then extract the
        position of the token in the note (character position) - so we return that
        this span starts at 32 and ends at 37. The function then returns a nested list
        that contains a tuple of tag type and tag position (character positions).
        Example: [[(3, 9), LOC], [(34, 41), PATIENT], ...]]
        Args:
            tokens (Sequence[Mapping[str, Union[str, int]]]): The list of tokens in the note
            predictions (Sequence[str]): The list of predictions for the note
            suffix (str): Whether the B, I etc is in the prefix or the suffix
        Returns:
            positions_info (List[Tuple[Tuple[int, int], str]])): List containing tuples of tag positions and tag type
        """
        positions_info = list()
        entities = Entities(sequences=[predictions], scheme=self._scheme, suffix=suffix)
        for entity_list in entities.entities:
            for entity in entity_list:
                position = (tokens[entity.start]['start'], tokens[entity.end - 1]['end'])
                positions_info.append([position, entity.tag])
        return positions_info

    def run_deid(
            self,
            input_file,
            predictions_file,
            deid_strategy,
            keep_age: bool = False,
            metadata_key: str = 'meta',
            note_id_key: str = 'note_id',
            tokens_key: str = 'tokens',
            predictions_key: str = 'predictions',
            text_key: str = 'text'
    ):
        # Store note_id to note mapping
        note_map = dict()
        for line in open(input_file, 'r'):
            note = json.loads(line)
            note_id = note[metadata_key][note_id_key]
            note_map[note_id] = note
        # Go through note predictions and de identify the note accordingly
        for line in open(predictions_file, 'r'):
            note = json.loads(line)
            # Get the text using the note_id for this note from the note_map dict
            note_id = note[note_id_key]
            # Get the note from the note_map dict
            deid_note = note_map[note_id]
            # Get predictions
            predictions = self.decode(tokens=note[tokens_key], predictions=note[predictions_key])
            # Get entities and their positions
            entity_positions = self.get_predicted_entities_positions(
                tokens=note[tokens_key],
                predictions=predictions,
                suffix=False
            )
            yield TextDeid.__get_deid_text(
                deid_note=deid_note,
                entity_positions=entity_positions,
                deid_strategy=deid_strategy,
                keep_age=keep_age,
                text_key=text_key
            )

    @staticmethod
    def __get_deid_text(
            deid_note,
            entity_positions,
            deid_strategy,
            keep_age: bool = False,
            text_key: str = 'text'
    ):
        tag_mapping = TextDeid.__get_tag_mapping(deid_strategy=deid_strategy)
        age_pattern = '((?<!\d+)([1-7]\d?)(?!\d+))|((?<!\d+)(8[0-8]?)(?!\d+))'
        # Sort positions - store the last occurring tag first - i.e in descending order
        # of start positions.
        entity_positions.sort(key=lambda info: info[0][0], reverse=True)
        # Get text and de identify it
        note_text = deid_note[text_key]
        deid_text = deid_note[text_key]
        # Go through the entities and their positions and de identify the text
        # Since we have the positions in sorted order (descending by start positions)
        # we de identify the text from the end to the start - i.e back to front
        for positions, tag in entity_positions:
            start_pos, end_pos = positions
            deid_tag = tag_mapping[tag]
            age_unchanged = False
            if tag == 'AGE' and keep_age:
                span_text = note_text[start_pos:end_pos]
                if regex.search(age_pattern, span_text, flags=regex.IGNORECASE):
                    deid_tag = span_text
                    age_unchanged = True
                else:
                    deid_tag = deid_tag
            if deid_strategy == 'replace_informative' and not age_unchanged:
                deid_text = deid_text[:start_pos] + deid_tag.format(note_text[start_pos:end_pos]) + deid_text[end_pos:]
            else:
                deid_text = deid_text[:start_pos] + deid_tag + deid_text[end_pos:]
        deid_note['deid_text'] = regex.sub('[\n]+', '\n', regex.sub('[ \t\r\f\v]+', ' ', deid_text)).strip()
        return deid_note

    @staticmethod
    def __get_tag_mapping(deid_strategy):
        if deid_strategy == 'remove':
            return remove()
        elif deid_strategy == 'replace_tag_type':
            return replace_tag_type()
        elif deid_strategy == 'replace_informative':
            return replace_informative()

    @staticmethod
    def __get_relaxed_predictions(predictions):
        return ['I-' + prediction[2:] if '-' in prediction else prediction for prediction in predictions]

    @staticmethod
    def __get_super_relaxed_predictions(tokens, predictions):
        # Super relaxed
        # 360 Longwood Ave, OBI, Boston
        # Tokens: ['360', 'Longwood', 'Ave', ',', 'OBI', ',', Boston[
        # Predictions: [B-LOC, I-LOC, L-LOC, O, U-LOC, O, U-LOC]
        # Relaxed: [I-LOC, I-LOC, I-LOC, O, I-LOC, O, I-LOC]
        # Super relaxed: [I-LOC, I-LOC, I-LOC, I-LOC, I-LOC, I-LOC, I-LOC]
        relaxed_predictions = TextDeid.__get_relaxed_predictions(predictions)
        prev_type = None
        replace_indexes = list()
        super_relaxed_predictions = list()
        for index, (token, prediction) in enumerate(zip(tokens, relaxed_predictions)):
            super_relaxed_predictions.append(prediction)
            # Check special characters that appear after a prediction
            # we can assign the prediction label to this sequence of special characters
            if prediction == 'O' and prev_type is not None:
                # [a-zA-Z0-9]
                if re.search('^(\W|_)+$', token['text'], flags=re.IGNORECASE | re.DOTALL):
                    replace_indexes.append(index)
                else:
                    prev_type = None
                    replace_indexes = list()
            # Replace all the tokens identified above with the NER prediction type
            # This is done only ig the current prediction type matches the previous type
            elif prediction != 'O':
                if prediction[2:] == prev_type and replace_indexes != []:
                    for replace_index in replace_indexes:
                        super_relaxed_predictions[replace_index] = 'I-' + prev_type
                # Reset list and previous type
                replace_indexes = list()
                prev_type = prediction[2:]
            else:
                prev_type = None
        return super_relaxed_predictions

    @staticmethod
    def __get_scheme(notation: str) -> Union[Type[IOB2], Type[IOBES], Type[BILOU], Type[IOB1]]:
        """
        Get the seqeval scheme based on the notation
        Args:
            notation (str): The NER notation
        Returns:
            (Union[IOB2, IOBES, BILOU, IOB1]): The seqeval scheme
        """
        if notation == 'BIO':
            return IOB2
        elif notation == 'BIOES':
            return IOBES
        elif notation == 'BILOU':
            return BILOU
        elif notation == 'IO':
            return IOB1
        else:
            raise ValueError('Invalid Notation')


def main():
    # The following code sets up the arguments to be passed via CLI or via a JSON file
    cli_parser = ArgumentParser(description='configuration arguments provided at run time from the CLI')
    cli_parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='the the jsonl file that contains the notes'
    )
    cli_parser.add_argument(
        '--predictions_file',
        type=str,
        required=True,
        help='the location where the predictions are'
    )
    cli_parser.add_argument(
        '--span_constraint',
        type=str,
        required=True,
        choices=['exact', 'strict', 'super_strict'],
        help='whether we want to modify the predictions, make the process of removing phi more struct etc'
    )
    cli_parser.add_argument(
        '--notation',
        type=str,

        required=True,
        help='the NER notation in the predictions'
    )
    cli_parser.add_argument(
        '--deid_strategy',
        type=str,
        required=True,
        choices=['remove', 'replace_tag_type', 'replace_informative'],
        help='The strategy '
    )
    cli_parser.add_argument(
        '--keep_age',
        action='store_true',
        help='whether to keep ages below 89'
    )
    cli_parser.add_argument(
        '--text_key',
        type=str,
        default='text',
        help='the key where the note text is present in the json object'
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
        '--tokens_key',
        type=str,
        default='tokens',
        help='the key where the tokens for the notes are present in the json object'
    )
    cli_parser.add_argument(
        '--predictions_key',
        type=str,
        default='predictions',
        help='the key where the note predictions is present in the json object'
    )
    cli_parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='the location we would write the deid notes'
    )
    # Parse args
    args = cli_parser.parse_args()
    text_deid = TextDeid(notation=args.notation, span_constraint=args.span_constraint)
    deid_notes = text_deid.run_deid(
            input_file=args.input_file,
            predictions_file=args.predictions_file,
            deid_strategy=args.deid_strategy,
            keep_age=args.keep_age,
            metadata_key=args.metadata_key,
            note_id_key=args.note_id_key,
            tokens_key=args.tokens_key,
            predictions_key=args.predictions_key,
            text_key=args.text_key
    )
    # Write the dataset to the output file
    with open(args.output_file, 'w') as file:
        for deid_note in deid_notes:
            file.write(json.dumps(deid_note) + '\n')


if __name__ == "__main__":
    # Get deid notes
    main()
