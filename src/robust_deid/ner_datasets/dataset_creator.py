import json
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Iterable, Dict, List, Union, Optional, Sequence, NoReturn

from .dataset_builder import Dataset, SentenceDataset
from .preprocessing import PreprocessingLoader

random.seed(41)


class DatasetCreator(object):
    """
    Build a NER token classification dataset
    For training we will build the dataset using the annotated spans (e.g from prodigy)
    For predictions we will assign default labels.
    The dataset is on a sentence level, i.e each note is split into sentences and the de-id
    task is run on a sentence level. Even the predictions are run on a sentence level
    The dataset would be something like:
    Tokens: [[tok1, tok2, ... tok-n], [tok ...], ..., [tok ...]]
    Labels: [[lab1, lab2, ... lab-n], [lab ...], ..., [lab ...]]
    Where the inner list represents the sentences - the tokens in the sentence and the respective
    labels for each token. The labels depend on the notation
    This script can also be used for predictions, the Labels will be filled with some
    default value. This is done so that we can use the same script for building a dataset to train a model
    and a dataset to obtain predictions using a model
    Example:
    Note: Bruce Wayne is a 60yo man. He lives in Gotham
    Sentences: [Bruce Wayne Jr is a 60yo man., He lives in Gotham]
    Tokens: [[Bruce, Wayne, Jr, is, a, 60, yo, man, .], [He, lives, in, Gotham]]
    Labels (BIO notation): [[B-Name, I-Name, I-Name, O, O, O, O, O, O], [O, O, O, B-LOC]]
    Labels (BILOU notation): [[B-Name, I-Name, L-Name, O, O, O, O, O, O], [O, O, O, U-LOC]]
    We also can create sentences that uses previous/next chunks as context - in this case the dataset would
    look something like this. (Assume we limit the size of the chunks to 3 tokens)
    Sentences: [Bruce Wayne Jr is a 60yo man., He lives in Gotham]
    Tokens: [[Bruce, Wayne, Jr, is, a, 60, yo, man, ., He, lives, in], [yo, man, ., He, lives, in, Gotham]]
    Labels (BIO notation): [[B-Name, I-Name, I-Name, O, O, O, O, O, O, NA, NA, NA], [NA, NA, NA, O, O, O, B-LOC]]
    Labels (BILOU notation): [[B-Name, I-Name, L-Name, O, O, O, O, O, O, NA, NA, NA], [NA, NA, NA, O, O, O, U-LOC]]
    NA represents the token is used for context
    """

    def __init__(
            self,
            sentencizer: str,
            tokenizer: str,
            max_tokens: int = 128,
            max_prev_sentence_token: int = 32,
            max_next_sentence_token: int = 32,
            default_chunk_size: int = 32,
            ignore_label: str = 'NA'
    ) -> NoReturn:
        """
        Initialize the sentencizer and tokenizer
        Args:
            sentencizer (str): Specify which sentencizer you want to use
            tokenizer (str): Specify which tokenizer you want to use
            max_tokens (int): The maximum number of tokens allowed in a sentence/training example,
                              truncate if it exceeds.
            max_prev_sentence_token (int): The maximum number of previous chunk tokens allowed in a
                                           sentence/training example
            max_next_sentence_token (int): The maximum number of next chunk tokens allowed in a
                                           sentence/training example.
            ignore_label (str): The label assigned to the previous and next chunks to distinguish
                                from the current sentence
        """
        self._sentencizer = PreprocessingLoader.get_sentencizer(sentencizer=sentencizer)
        self._tokenizer = PreprocessingLoader.get_tokenizer(tokenizer=tokenizer)
        # Initialize the object that will be used to get the tokens and the sentences
        self._dataset = Dataset(sentencizer=self._sentencizer, tokenizer=self._tokenizer)
        # Initialize the object that will take all the sentences in the note and return
        # a dataset where each row represents a sentence in the note. The sentence in each
        # row will also contain a previous chunk and next chunk (tokens) that will act as context
        # when training the mode
        # [ps1, ps 2, ps 3...ps-i], [cs1, cs2, ... cs-j], [ns, ns, ... ns-k] - as you can see the current sentence
        # which is the sentence we train on (or predict on) will be in the middle - the surrounding tokens will
        # provide context to the current sentence
        self._sentence_dataset = SentenceDataset(
            max_tokens=max_tokens,
            max_prev_sentence_token=max_prev_sentence_token,
            max_next_sentence_token=max_next_sentence_token,
            default_chunk_size=default_chunk_size,
            ignore_label=ignore_label
        )

    def create(
            self,
            input_file: str,
            mode: str = 'predict',
            notation: str = 'BIO',
            token_text_key: str = 'text',
            metadata_key: str = 'meta',
            note_id_key: str = 'note_id',
            label_key: str = 'labels',
            span_text_key: str = 'spans'
    ) -> Iterable[Dict[str, Union[List[Dict[str, Union[str, int]]], List[str]]]]:
        """
        This function is used to get the sentences that will be part of the NER dataset.
        We check whether the note belongs to the desired dataset split. If it does,
        we fix any spans that can cause token-span alignment errors. Then we extract
        all the sentences in the notes, the tokens in each sentence. Finally we
        add some context tokens to the sentence if required. This function returns
        an iterable that iterated through each of the processed sentences
        Args:
            input_file (str): Input jsonl file. Make sure the spans are in ascending order (based on start position)
            mode (str): Dataset being built for train or predict.
            notation (str): The NER labelling notation
            token_text_key (str): The key where the note text and token text is present in the json object
            metadata_key (str): The key where the note metadata is present in the json object
            note_id_key (str): The key where the note id is present in the json object
            label_key (str): The key where the token label will be stored in the json object
            span_text_key (str): The key where the note spans is present in the json object
        Returns:
            (Iterable[Dict[str, Union[List[Dict[str, Union[str, int]]], List[str]]]]): Iterate through the processed
                                                                                       sentences/training examples
        """
        # Go through the notes
        for line in open(input_file, 'r'):
            note = json.loads(line)
            note_text = note[token_text_key]
            note_id = note[metadata_key][note_id_key]
            
            # Skip to next note if empty string
            if not note_text:
                continue
            
            if mode == 'train':
                note_spans = note[span_text_key]
            # No spans in predict mode
            elif mode == 'predict':
                note_spans = None
            else:
                raise ValueError("Invalid mode - can only be train/predict")
            # Store the list of tokens in the sentence
            # Eventually this list will contain all the tokens in the note (split on the sentence level)
            # Store the start and end positions of the sentence in the note. This can
            # be used later to reconstruct the note from the sentences
            # we also store the note_id for each sentence so that we can map it back
            # to the note and therefore have all the sentences mapped back to the notes they belong to.
            sent_tokens = [sent_tok for sent_tok in self._dataset.get_tokens(
                text=note_text,
                spans=note_spans,
                notation=notation
            )]
            # The following loop goes through each sentence in the note and returns
            # the current sentence and previous and next chunks that will be used for context
            # The chunks will have a default label (e.g NA) to distinguish from the current sentence
            # and so that we can ignore these chunks when calculating loss and updating weights
            # during training
            for ner_sent_index, ner_sentence in self._sentence_dataset.get_sentences(
                    sent_tokens=sent_tokens,
                    token_text_key=token_text_key,
                    label_key=label_key
            ):
                # Return the processed sentence. This sentence will then be used
                # by the model
                current_sent_info = ner_sentence['current_sent_info']
                note_sent_info_store = {'start': current_sent_info[0]['start'],
                                        'end': current_sent_info[-1]['end'], 'note_id': note_id}
                ner_sentence['note_sent_info'] = note_sent_info_store
                yield ner_sentence


def main():
    cli_parser = ArgumentParser(
        description='configuration arguments provided at run time from the CLI',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    cli_parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='the the jsonl file that contains the notes. spans need to be sorted in ascending order (based on start '
             'position) '
    )
    cli_parser.add_argument(
        '--notation',
        type=str,
        default='BIO',
        help='the notation we will be using for the label scheme'
    )
    cli_parser.add_argument(
        '--max_tokens',
        type=int,
        default=128,
        help='The max tokens that a given sentence (training/prediction example) in the note can have'
    )
    cli_parser.add_argument(
        '--default_chunk_size',
        type=int,
        default=32,
        help='the default chunk size for the previous and next chunks for a given sentence (training/prediction '
             'example) in the note can have '
    )
    cli_parser.add_argument(
        '--max_prev_sentence_token',
        type=int,
        default=32,
        help='the max chunk size for the previous chunks for a given sentence (training/prediction example) in the '
             'note can have '
    )
    cli_parser.add_argument(
        '--max_next_sentence_token',
        type=int,
        default=32,
        help='the max chunk size for the next chunks for a given sentence (training/prediction example) in the note '
             'can have '
    )
    cli_parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict'],
        required=True,
        help='whether we are building the dataset for training or prediction'
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
        '--ignore_label',
        type=str,
        default='NA',
        help='whether to use the ignore label or not'
    )
    cli_parser.add_argument(
        '--token_text_key',
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
        help='the key where the note metadata is present in the json object'
    )
    cli_parser.add_argument(
        '--label_key',
        type=str,
        default='label',
        help='the key where the note label for each token is present in the json object'
    )
    cli_parser.add_argument(
        '--span_text_key',
        type=str,
        default='spans',
        help='the key where the note annotates spans are present in the json object'
    )
    cli_parser.add_argument(
        '--format',
        type=str,
        default='jsonl',
        help='format to store the dataset in: jsonl or conll'
    )
    cli_parser.add_argument(
        '--output_file',
        type=str,
        help='The file where the NER dataset will be stored'
    )
    args = cli_parser.parse_args()
    dataset_creator = DatasetCreator(
        sentencizer=args.sentencizer,
        tokenizer=args.tokenizer,
        max_tokens=args.max_tokens,
        max_prev_sentence_token=args.max_prev_sentence_token,
        max_next_sentence_token=args.max_next_sentence_token,
        default_chunk_size=args.default_chunk_size,
        ignore_label=args.ignore_label)
    ner_notes = dataset_creator.create(
        input_file=args.input_file,
        mode=args.mode,
        notation=args.notation,
        token_text_key=args.token_text_key,
        metadata_key=args.metadata_key,
        note_id_key=args.note_id_key,
        label_key=args.label_key,
        span_text_key=args.span_text_key
    )
    # Store the NER dataset in the desired format
    if args.format == 'jsonl':
        # Write the dataset to the output file
        with open(args.output_file, 'w') as file:
            for ner_sentence in ner_notes:
                file.write(json.dumps(ner_sentence) + '\n')
    elif args.format == 'conll':
        with open(args.output_file, 'w') as file:
            for ner_sentence in ner_notes:
                tokens = ner_sentence['tokens']
                labels = ner_sentence['labels']
                current_sent_info = ner_sentence['current_sent_info']
                note_id = ner_sentence['note_sent_info']['note_id']
                if len(tokens) != len(labels) or len(labels) != len(current_sent_info):
                    raise ValueError('Length mismatch')
                for token, label, sent_info in zip(tokens, labels, current_sent_info):
                    sent_info['note_id'] = note_id
                    data = token + ' ' + label + ' ' + json.dumps(sent_info) + '\n'
                    file.write(data)
                file.write('\n')


if __name__ == '__main__':

    main()
