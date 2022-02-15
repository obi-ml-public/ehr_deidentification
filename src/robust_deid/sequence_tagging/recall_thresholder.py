import json
import numpy as np
from scipy.special import softmax
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import NoReturn
from sklearn.metrics import precision_recall_curve

from .dataset_builder import NERLabels

class RecallThresholder(object):

    def __init__(self, ner_types, notation):
        ner_labels = NERLabels(notation=notation, ner_types=ner_types)
        label_list = ner_labels.get_label_list()
        self._mask = np.zeros((len(label_list)), dtype=bool)
        self._mask[label_list.index('O')] = True

    def get_precision_recall_threshold(self, logits_file, recall_cutoff, threshold_mode='max', predictions_key='predictions', labels_key='labels'):
        if(threshold_mode == 'max'):
            y_true, y_pred = self.__convert_binary_max(
                logits_file=logits_file,
                predictions_key=predictions_key,
                labels_key=labels_key
            )
        elif(threshold_mode == 'sum'):
            y_true, y_pred = self.__convert_binary_sum(
                logits_file=logits_file,
                predictions_key=predictions_key,
                labels_key=labels_key
            )
        precision, recall, threshold = self.__get_precision_recall_threshold(y_true=y_true, y_pred=y_pred, recall_cutoff=recall_cutoff)
        return precision[-1], recall[-1], threshold[-1]

    def __convert_binary_max(self, logits_file, predictions_key='predictions', labels_key='labels'):
        y_true = list()
        y_pred = list()
        for line in open(logits_file, 'r'):
            note = json.loads(line)
            for prediction, label in zip(note[predictions_key], note[labels_key]):
                logits = softmax(prediction)
                masked_logits = np.ma.MaskedArray(data=logits, mask=self._mask)
                y_true.append(0 if label == 'O' else 1)
                y_pred.append(masked_logits.max())
        return y_true, y_pred

    def __convert_binary_sum(self, logits_file, predictions_key='predictions', labels_key='labels'):
        y_true = list()
        y_pred = list()
        for line in open(logits_file, 'r'):
            note = json.loads(line)
            for prediction, label in zip(note[predictions_key], note[labels_key]):
                logits = softmax(prediction)
                masked_logits = np.ma.MaskedArray(data=logits, mask=self._mask)
                y_true.append(0 if label == 'O' else 1)
                y_pred.append(masked_logits.sum())
        return y_true, y_pred

    def __get_precision_recall_threshold(self, y_true, y_pred, recall_cutoff):
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred, pos_label=1)
        thresholds = np.append(thresholds, thresholds[-1])
        precision_filter = precision[recall > recall_cutoff]
        recall_filter = recall[recall > recall_cutoff]
        thresholds_filter = thresholds[recall > recall_cutoff]
        return precision_filter, recall_filter, thresholds_filter


def main() -> NoReturn:
    cli_parser = ArgumentParser(
        description='configuration arguments provided at run time from the CLI',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    cli_parser.add_argument(
        '--logits_file',
        type=str,
        required=True,
        help='the jsonl file that contains the logit predictions at each token position'
    )
    cli_parser.add_argument(
        '--ner_types',
        nargs="+",
        required=True,
        help='the NER types'
    )
    cli_parser.add_argument(
        '--notation',
        type=str,
        default='BIO',
        help='the notation we will be using for the label scheme'
    )
    cli_parser.add_argument(
        '--threshold_mode',
        type=str,
        choices=['max', 'sum'],
        required=True,
        help='whether we want to use the summation approach or max approach for thresholding. will need to call the right approach with the sequence tagger as well'
    )
    cli_parser.add_argument(
        '--recall_cutoff',
        type=float,
        required=True,
        help='the recall value you are trying to achieve'
    )
    cli_parser.add_argument(
        '--predictions_key',
        type=str,
        default='predictions',
        help='the key where the note predictions (logits) for each token is present in the json object'
    )
    cli_parser.add_argument(
        '--labels_key',
        type=str,
        default='labels',
        help='the key where the note labels for each token is present in the json object'
    )
    args = cli_parser.parse_args()

    recall_thresholder = RecallThresholder(ner_types=args.ner_types, notation=args.notation)
    precision, recall, threshold = recall_thresholder.get_precision_recall_threshold(
        logits_file=args.logits_file,
        recall_cutoff=args.recall_cutoff/100,
        threshold_mode=args.threshold_mode,
        predictions_key=args.predictions_key,
        labels_key=args.labels_key
    )
    print('Threshold Mode: ' + args.threshold_mode.upper())
    print('At threshold: ', threshold)
    print('Precision is: ', precision * 100)
    print('Recall is: ', recall * 100)

if __name__ == "__main__":
    main()
