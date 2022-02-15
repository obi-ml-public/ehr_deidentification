import re
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Iterable, Dict, List, Sequence, Union, Mapping, Tuple, NoReturn

from .preprocessing import PreprocessingLoader


class SpanFixer(object):
    """
    The tokens and spans may not align depending on the tokenizer used.
    This class either expands the span to cover the tokens, so we don't have a mismatch.
    A mismatch is when a span_start will not coincide with some token_start or the span_end
    will not coincide with some token_end. This class changes the span_start and span_end
    so that the span_start will coincide with some token_start and the span_end
    will coincide with some token_end - and we don't get any position mismatch errors while
    building our dataset. This entire process involves updating span positions which can lead to duplicate
    or overlapping spans, which then need to be removed.
    E.g we have text: The patient is 75yo man
    AGE Span: 75
    Token: 75yo
    As you can see the span is smaller than the token, which will lead to an error when
    building the NER dataset.
    To ensure this does not happen, we correct the span. We change the span from
    75 to 75yo -> So now AGE Span is 75yo instead of 75. This script essentially changes
    the annotated spans to match the tokens. In an ideal case we wouldn't need this script
    but since medical notes have many typos, this script becomes necessary to deal with
    issues and changes that arise from different tokenizers.
    Also sort the spans and convert the start and end keys of the spans to integers
    """

    def __init__(
            self,
            sentencizer: str,
            tokenizer: str,
            ner_priorities: Mapping[str, int],
            verbose: bool = True
    ) -> NoReturn:
        """
        Initialize the sentencizer and tokenizer
        Args:
            sentencizer (str): The sentencizer to use for splitting text into sentences
            tokenizer (str): The tokenizer to use for splitting text into tokens
            ner_priorities (Mapping[str, int]): The priority when choosing which duplicates to remove.
                                                Mapping that represents a priority for each NER type
            verbose (bool): To print out warnings etc
        """
        self._sentencizer = PreprocessingLoader.get_sentencizer(sentencizer)
        self._tokenizer = PreprocessingLoader.get_tokenizer(tokenizer)
        self._ner_priorities = ner_priorities
        self._verbose = verbose

    def __get_token_positions(self, text: str) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Get the start and end positions of all the tokens in the note.
        Args:
            text (str): The text present in the note
        Returns:
            token_start_positions (Mapping[int, int]): The start positions of all the tokens in the note
            token_end_positions (Mapping[int, int]): The end positions of all the tokens in the note
        """
        token_start_positions = dict()
        token_end_positions = dict()
        for sentence in self._sentencizer.get_sentences(text):
            offset = sentence['start']
            for token in self._tokenizer.get_tokens(sentence['text']):
                start = token['start'] + offset
                end = token['end'] + offset
                token_start_positions[start] = 1
                token_end_positions[end] = 1
        return token_start_positions, token_end_positions

    def get_duplicates(
            self,
            spans: List[Dict[str, Union[str, int]]],
    ) -> List[int]:
        """
        Return the indexes where there are duplicate/overlapping spans. A duplicate or
        span is one where the same token can have two labels. 
        E.g:
        Token: BWH^Bruce
        This is a single token where BWH is the hospital label and Bruce is the Patient label
        The fix_alignment function assigns this entre token the hospital label but it also
        assigns this entire token the patient label. Since we have two labels for the same
        token, we need to remove one of them.
        We assign this entire token one label - either hospital label or the patient label
        In this case we assign patient because of higher priority. So now we need to remove
        the hospital label from the dataset (since it is essentially a duplicate label). This
        script handles this case.
        There are cases when two different labels match the same token partially
        E.g
        Text: JT/781-815-9090
        Spans: JT - hospital, 781-815-9090 - Phone
        Tokens: (Jt/781) & (- 815 - 9090)
        As you can see the token JT/781 will be assigned the label in the fix_alignment function
        but 781-815-9090 is also phone and the 781 portion is overlapped, and we need to resolve this.
        In this script, we resolve it by treating JT/781 as one span (hospital) and 
        -815-9090 as another span (phone).
        Args:
            spans ([List[Dict[str, Union[str, int]]]): The NER spans in the note
        Returns:
            remove_spans (Sequence[int]): A list of indexes of the spans to remove
        """
        remove_spans = list()
        prev_start = -1
        prev_end = -1
        prev_label = None
        prev_index = None
        spans.sort(key=lambda _span: (_span['start'], _span['end']))
        for index, span in enumerate(spans):
            current_start = span['start']
            current_end = span['end']
            current_label = span['label']
            if type(current_start) != int or type(current_end) != int:
                raise ValueError('The start and end keys of the span must be of type int')
            # Check if the current span matches another span
            # that is if this span covers the same tokens as the
            # previous spans (but has a different label)
            # Based on the priority, treat the span with the low
            # priority label as a duplicate label and add it to the
            # list of spans that need to be removed
            if current_start == prev_start and current_end == prev_end:
                if self._ner_priorities[current_label] > self._ner_priorities[prev_label]:
                    # Store index of the previous span if it has lower priority
                    remove_spans.append(prev_index)
                    # Reset span details
                    prev_start = current_start
                    prev_end = current_end
                    prev_index = index
                    prev_label = current_label
                    if self._verbose:
                        print('DUPLICATE: ', span)
                        print('REMOVED: ', spans[remove_spans[-1]])
                elif self._ner_priorities[current_label] <= self._ner_priorities[prev_label]:
                    # Store current index of span if it has lower priority
                    remove_spans.append(index)
                    if self._verbose:
                        print('DUPLICATE: ', spans[prev_index])
                        print('REMOVED: ', spans[remove_spans[-1]])
            # Check for overlapping span
            elif current_start < prev_end:
                # If the current span end matches the overlapping span end
                # Remove the current span, since it is smaller
                if current_end <= prev_end:
                    remove_spans.append(index)
                    if self._verbose:
                        print('DUPLICATE: ', spans[prev_index])
                        print('REMOVED: ', spans[remove_spans[-1]])
                # If the current end is greater than the prev_end
                # then we split it into tow spans. We treat the previous span
                # as one span and the end of the previous span to the end of the current span
                # as another span.
                elif current_end > prev_end:
                    # Create the new span - start=previous_span_end, end=current_span_end
                    overlap_length = spans[prev_index]['end'] - current_start
                    new_text = span['text'][overlap_length:]
                    # Remove extra spaces that may arise during this span separation
                    new_text = re.sub('^(\s+)', '', new_text, flags=re.DOTALL)
                    span['start'] = current_end - len(new_text)
                    span['text'] = new_text
                    if self._verbose:
                        print('OVERLAP: ', spans[prev_index])
                        print('UPDATED: ', span)
                    # Reset span details
                    prev_start = current_start
                    prev_end = current_end
                    prev_label = current_label
                    prev_index = index
            # Reset span details
            else:
                prev_start = current_start
                prev_end = current_end
                prev_label = current_label
                prev_index = index
        return remove_spans

    def fix_alignment(
            self,
            text: str,
            spans: Sequence[Dict[str, Union[str, int]]]
    ) -> Iterable[Dict[str, Union[str, int]]]:
        """
        Align the span and tokens. When the tokens and spans don't align, we change the 
        start and end positions of the spans so that they align with the tokens. This is
        needed when a different tokenizer is used and the spans which are defined against
        a different tokenizer don't line up with the new tokenizer. Also remove spaces present
        at the start or end of the span.
        E.g:
        Token: BWH^Bruce
        This is a single token where BWH is the hospital label and Bruce is the Patient label
        The fix_alignment function assigns this entre token the hospital label but it also
        assigns this entire token the patient label. This function basically expands the span
        so that it matches the start and end positions of some token. By doing this it may create
        overlapping and duplicate spans. As you can see it expands the patient label to match the
        start of the token and it expands the hospital label to match the end of the token.
        function.
        Args:
            text (str): The text present in the note
            spans ([Sequence[Dict[str, Union[str, int]]]): The NER spans in the note
        Returns:
            (Iterable[Dict[str, Union[str, int]]]): Iterable through the modified spans
        """
        # Get token start and end positions so that we can check if a span
        # coincides with the start and end position of some token.
        token_start_positions, token_end_positions = self.__get_token_positions(text)
        for span in spans:
            start = span['start']
            end = span['end']
            if type(start) != int or type(end) != int:
                raise ValueError('The start and end keys of the span must be of type int')
            if re.search('^\s', text[start:end]):
                if self._verbose:
                    print('WARNING - space present in the start of the span')
                start = start + 1
            if re.search('(\s+)$', text[start:end], flags=re.DOTALL):
                new_text = re.sub('(\s+)$', '', text[start:end], flags=re.DOTALL)
                end = start + len(new_text)
            # When a span does not coincide with the start and end position of some token
            # it means there will be an error when building the ner dataset, we try and avoid
            # that error by updating the spans itself, that is we expand the start/end positions
            # of the spans so that it is aligned with the tokens.
            while token_start_positions.get(start, False) is False:
                start -= 1
            while token_end_positions.get(end, False) is False:
                end += 1
            # Print what the old span was and what the new expanded span will look like
            if self._verbose and (int(span['start']) != start or int(span['end']) != end):
                print('OLD SPAN: ', text[int(span['start']):int(span['end'])])
                print('NEW SPAN: ', text[start:end])
            # Update the span with its new start and end positions
            span['start'] = start
            span['end'] = end
            span['text'] = text[start:end]
            yield span

    def fix_note(
            self,
            text: str,
            spans: Sequence[Dict[str, Union[str, int]]],
    ) -> Iterable[Dict[str, Union[str, int]]]:
        """
        This function changes the span_start and span_end
        so that the span_start will coincide with some token_start and the span_end
        will coincide with some token_end and also removes duplicate/overlapping spans
        that may arise when we change the span start and end positions. The resulting
        spans from this function will always coincide with some token start and token
        end, and hence will not have any token and span mismatch errors when building the
        NER dataset. For more details and examples check the documentation of the 
        fix_alignment and get_duplicates functions.
        Args:
            text (str): The text present in the note
            spans ([Sequence[Mapping[str, Union[str, int]]]): The NER spans in the note
        Returns:
            (Iterable[Mapping[str, Union[str, int]]]): Iterable through the fixed spans
        """
        # Fix span position alignment
        spans = [span for span in self.fix_alignment(text=text, spans=spans)]
        # Check for duplicate/overlapping spans
        remove_spans = self.get_duplicates(spans=spans)
        for index, span in enumerate(spans):
            # Remove the duplicate/overlapping spans
            if index not in remove_spans:
                yield span

    def fix(
            self,
            input_file: str,
            text_key: str = 'text',
            spans_key: str = 'spans'
    ) -> Iterable[Dict[str, Union[str, Dict[str, str], List[Dict[str, str]]]]]:
        """
        This function changes the span_start and span_end
        so that the span_start will coincide with some token_start and the span_end
        will coincide with some token_end and also removes duplicate/overlapping spans
        that may arise when we change the span start and end positions. The resulting
        spans from this function will always coincide with some token start and token
        end, and hence will not have any token and span mismatch errors when building the
        NER dataset. For more details and examples check the documentation of the
        fix_alignment and get_duplicates functions. Fix spans that arise due to bad typos,
        which are not fixed during tokenization. This essentially updates the spans so that
        they line up with the start and end positions of tokens - so that there is no error
        when we assign labels to tokens based on these spans
        Args:
            input_file (str): The file that contains the notes that we want to fix the token issues in
            text_key (str) the key where the note & token text is present in the json object
            spans_key (str): The key where the note spans are present in the json object
        Returns:
            (Iterable[Dict[str, Union[str, Dict[str, str], List[Dict[str, str]]]]]): Iterable through the fixed
                                                                                     notes
        """
        for line in open(input_file, 'r'):
            note = json.loads(line)
            note[spans_key] = [span for span in self.fix_note(text=note[text_key], spans=note[spans_key])]
            yield note


def main():
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
        '--sentencizer',
        type=str,
        required=True,
        help='the sentencizer to use for splitting notes into sentences'
    )
    cli_parser.add_argument(
        '--tokenizer',
        type=str,
        required=True,
        help='the tokenizer to use for splitting text into tokens'
    )
    cli_parser.add_argument(
        '--ner_types',
        nargs="+",
        required=True,
        help='the NER types'
    )
    cli_parser.add_argument(
        '--ner_priorities',
        nargs="+",
        required=True,
        help='the priorities for the NER types - the priority when choosing which duplicates to remove'
    )
    cli_parser.add_argument(
        '--verbose',
        action='store_true',
        help='whether to print the information aout span fixes'
    )
    cli_parser.add_argument(
        '--text_key',
        type=str,
        default='text',
        help='the key where the note & token text is present in the json object'
    )
    cli_parser.add_argument(
        '--spans_key',
        type=str,
        default='spans',
        help='the key where the note spans is present in the json object'
    )
    cli_parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='the output json file that will contain the new fixed spans'
    )
    args = cli_parser.parse_args()
    # Mapping that represents a priority for each PHI type
    # For example, the PATIENT type will have a higher priority as
    # compared to STAFF.
    if len(args.ner_types) == len(args.ner_priorities):
        ner_priorities = {ner_type: priority for ner_type, priority in zip(args.ner_types, args.ner_priorities)}
    else:
        raise ValueError('Length of ner_types and ner_priorities must be the same')
    span_fixer = SpanFixer(
        tokenizer=args.tokenizer,
        sentencizer=args.sentencizer,
        ner_priorities=ner_priorities,
        verbose=args.verbose
    )
    with open(args.output_file, 'w') as file:
        for note in span_fixer.fix(
                input_file=args.input_file,
                text_key=args.text_key,
                spans_key=args.spans_key
        ):
            file.write(json.dumps(note) + '\n')


if __name__ == '__main__':
    main()
