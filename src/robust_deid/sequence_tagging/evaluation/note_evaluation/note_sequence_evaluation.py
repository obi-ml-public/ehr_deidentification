# Script to evaluate at a spans level
# Sequence evaluation - code is based on the seqeval library/package
# While seqeval evaluates at the token position level, we evalaute at a
# character position level. Since most of the code is the same, refer
# to the library/github repo of the seqeval package fore more details.
import warnings
from collections import defaultdict
from typing import Sequence, List, Optional, Tuple, Type, Union, Mapping

import numpy as np
from seqeval.metrics.sequence_labeling import get_entities
from seqeval.reporters import DictReporter, StringReporter
from seqeval.scheme import Entities, Token
from sklearn.exceptions import UndefinedMetricWarning

PER_CLASS_SCORES = Tuple[List[float], List[float], List[float], List[int]]
AVERAGE_SCORES = Tuple[float, float, float, int]
SCORES = Union[PER_CLASS_SCORES, AVERAGE_SCORES]


class NoteSequenceEvaluation(object):
    """
    There already exists a package (seqeval) that can do the sequence evaluation.
    The reason we have this class is that the package seqeval looks at it from a
    token level classification perspective. So it evaluates if the spans formed by
    the token classification (predictions) matches/not matches spans formed by the labels.
    But in medical notes, there are many cases where the token and label dont align
    E.g inboston - is one token, but the LOC span is in[boston], it does not cover the 
    entire token. Since seqeval evaluates at a token level, it makes it hard to evaluate models
    or penalize models that dont handle tokenization issues. This evaluation script is used
    to evaluate the model at a character level. We essentially see if the character positions
    line up, as opposed to token positions, in which case we can handle evaluation of cases
    like inboston. We borrow most of the code and intuition from seqeval and make changes
    where necessary to suit our needs.    
    """

    @staticmethod
    def extract_predicted_spans_default(
            tokens: Sequence[Mapping[str, Union[str, int]]],
            predictions: Sequence[str],
            suffix: str
    ) -> defaultdict(set):
        """
        Use the seqeval get_entities method, which goes through the predictions and returns
        where the span starts and ends. - [O, O, B-AGE, I-AGE, O, O] this will return
        spans starts at token 2 and ends at token 3 - with type AGE. We then extract the
        position of the token in the note (character position) - so we return that
        this span starts at 32 and ends at 37. The function then returns a dict
        where the keys are the NER types and the values are the list of different
        positions these types occur within the note.
        Args:
            tokens (Sequence[Mapping[str, Union[str, int]]]): The list of tokens in the note
            predictions (Sequence[str]): The list of predictions for the note
            suffix (str): Whether the B, I etc is in the prefix or the suffix
        Returns:
            entities_pred (defaultdict(set)): Keys are the NER types and the value is a set that 
                                              contains the positions of these types
        """
        entities_pred = defaultdict(set)
        for type_name, start, end in get_entities(predictions, suffix=suffix):
            entities_pred[type_name].add((tokens[start]['start'], tokens[end]['end']))
        return entities_pred

    @staticmethod
    def extract_predicted_spans_strict(
            tokens: Sequence[Mapping[str, Union[str, int]]],
            predictions: Sequence[str],
            ner_types: Sequence[str],
            scheme: Type[Token],
            suffix: str
    ) -> defaultdict(set):
        """
        Use the seqeval get_entities method, which goes through the predictions and returns
        where the span starts and ends. - [O, O, B-AGE, I-AGE, O, O] this will return
        spans starts at token 2 and ends at token 3 - with type AGE. We then extract the
        position of the token in the note (character position) - so we return that
        this span starts at 32 and ends at 37. The function then returns a dict
        where the keys are the NER types and the values are the set of different
        positions these types occur within the note. The difference with the
        extract_predicted_spans_default function is this is more strict in that
        the spans needs to start with B tag and other constraints depending on the scheme
        Args:
            tokens (Sequence[Mapping[str, Union[str, int]]]): The list of tokens in the note
            predictions (Sequence[str]): The list of predictions for the note
            scheme (Type[Token]): The NER labelling scheme
            suffix (str): Whether the B, I etc is in the prefix or the suffix
        Returns:
            entities_pred (defaultdict(set)): Keys are the NER types and the value is a set that 
                                              contains the positions of these types
        """
        entities_pred = defaultdict(set)
        entities = Entities(sequences=[predictions], scheme=scheme, suffix=suffix)
        for tag in ner_types:
            for entity in entities.filter(tag):
                entities_pred[entity.tag].add((tokens[entity.start]['start'], tokens[entity.end - 1]['end']))
        return entities_pred

    @staticmethod
    def extract_true_spans(note_spans: Sequence[Mapping[str, Union[str, int]]]) -> defaultdict(set):
        """
        Go through the list of annotated spans and create a mapping like we do for the other
        functions - where the mapping contains keys which are the NER types and the values are the set of different
        positions (start, end) these NER types/spans occur within the note.
        Args:
            note_spans (Sequence[Mapping[str, Union[str, int]]]): The list of spans in the note
        Returns:
            entities_true (defaultdict(set)): Keys are the NER types and the value is a set that 
                                              contains the positions of these types
        """
        entities_true = defaultdict(set)
        for span in note_spans:
            entities_true[span['label']].add((int(span['start']), int(span['end'])))
        return entities_true

    @staticmethod
    def extract_tp_actual_correct(
            note_predictions: Sequence[Sequence[str]],
            note_tokens: Sequence[Sequence[Mapping[str, Union[str, int]]]],
            note_spans: Sequence[Sequence[Mapping[str, Union[str, int]]]],
            ner_types: Sequence[str],
            scheme: Type[Token],
            mode: str,
            suffix: str
    ) -> Tuple[List, List, List]:
        """
        Extract the the number of gold spans per NER types, the number of predicted spans per
        NER type, the number of spans where the gold standard and predicted spans match for all 
        the notes in the evaluation dataset. This is mainly done by comparing the gold standard span
        positions and the predicted span positions using the extract_predicted_spans_default,
        extract_predicted_spans_strict, extract_true_spans functions
        The annotated spans is a list that contains a list of spans for each note. This list of spans contain
        the span label and the position (start, end) of the span in the note (character positions).
        We use this as our true labels. The reason we do this, is because for medical notes it's better
        to have character level positions, because it makes it easier to evaluate typos.
        Note tokens is a list that in turn contains a list of tokens present in the note. For each token
        we have it start and end position (character positions) in the note. For evaluation of the model
        predictions, the note_spans and note_tokens, remain constant and hence we initialize it here.
        We use note tokens to map the predictions of the model to the character positions and then
        compare it with the character positions of the annotated spans.
        Args:
            note_predictions (Sequence[Sequence[str]]): The list of predictions in the evaluation dataset
            note_spans (Sequence[Sequence[Mapping[str, Union[str, int]]]]): The list of annotated spans for the notes 
                                                                            in the evaluation dataset 
            note_tokens (Sequence[Sequence[Mapping[str, Union[str, int]]]]): The list of tokens for the notes 
                                                                             in the evaluation dataset
            ner_types (Sequence[str]): The list of NER types e.g AGE, DATE etc
            scheme (Type[Token]): The NER labelling scheme
            mode (str): Whether to use default or strict evaluation
            suffix (str): Whether the B, I etc is in the prefix or the suffix
        Returns:
            pred_sum (np.array): The number of predicted spans
            tp_sum (np.array): The number of predicted spans that match gold standard spans
            true_sum (np.array): The number of gold standard spans
        """
        # Initialize the arrays that will store the number of predicted spans per NER type
        # the gold standard number of spans per NER type and the number of spans that match between
        # the predicted and actual (true positives)
        tp_sum = np.zeros(len(ner_types), dtype=np.int32)
        pred_sum = np.zeros(len(ner_types), dtype=np.int32)
        true_sum = np.zeros(len(ner_types), dtype=np.int32)
        # Calculate the number of true positives, predicted and actual number of spans per NER type
        # for each note and sum up the results
        for spans, tokens, predictions in zip(note_spans, note_tokens, note_predictions):
            # Get all the gold standard spans
            entities_true = NoteSequenceEvaluation.extract_true_spans(note_spans=spans)
            # Get all the predicted spans
            if mode == 'default':
                entities_pred = NoteSequenceEvaluation.extract_predicted_spans_default(
                    tokens=tokens,
                    predictions=predictions,
                    suffix=suffix
                )
            elif mode == 'strict':
                entities_pred = NoteSequenceEvaluation.extract_predicted_spans_strict(
                    tokens=tokens,
                    predictions=predictions,
                    ner_types=ner_types,
                    scheme=scheme,
                    suffix=suffix
                )
            else:
                raise ValueError('Invalid Mode')
                # Calculate and store the number of the gold standard spans, predicted spans and true positives
            # for each NER type
            for ner_index, ner_type in enumerate(ner_types):
                entities_true_type = entities_true.get(ner_type, set())
                entities_pred_type = entities_pred.get(ner_type, set())
                tp_sum[ner_index] += len(entities_true_type & entities_pred_type)
                pred_sum[ner_index] += len(entities_pred_type)
                true_sum[ner_index] += len(entities_true_type)
        return pred_sum, tp_sum, true_sum

    @staticmethod
    def precision_recall_fscore(
            note_predictions: Sequence[Sequence[str]],
            note_tokens: Sequence[Sequence[Mapping[str, Union[str, int]]]],
            note_spans: Sequence[Sequence[Mapping[str, Union[str, int]]]],
            ner_types: Sequence[str],
            scheme: Type[Token],
            mode: str,
            *,
            average: Optional[str] = None,
            warn_for=('precision', 'recall', 'f-score'),
            beta: float = 1.0,
            sample_weight: Optional[List[int]] = None,
            zero_division: str = 'warn',
            suffix: bool = False
    ) -> SCORES:
        """
        Extract the precision, recall and F score based on the number of predicted spans per
        NER type, the number of spans where the gold standard and predicted spans match for all 
        the notes in the evaluation dataset.
        Return the precision, recall, f1 scores for each NER type and averaged scores (micro, macro etc)
        Args:
            note_predictions (Sequence[Sequence[str]]): The list of predictions in the evaluation dataset
            note_spans (Sequence[Sequence[Mapping[str, Union[str, int]]]]): The list of annotated spans for the notes 
                                                                            in the evaluation dataset 
            note_tokens (Sequence[Sequence[Mapping[str, Union[str, int]]]]): The list of tokens for the notes 
                                                                             in the evaluation dataset
            ner_types (Sequence[str]): The list of NER types e.g AGE, DATE etc
            scheme (Type[Token]): The NER labelling scheme
            mode (str): Whether to use default or strict evaluation
            suffix (str): Whether the B, I etc is in the prefix or the suffix
            sample_weight : array-like of shape (n_samples,), default=None
                Sample weights.
            zero_division : "warn", 0 or 1, default="warn"
                Sets the value to return when there is a zero division:
                   - recall: when there are no positive labels
                   - precision: when there are no positive predictions
                   - f-score: both
                If set to "warn", this acts as 0, but warnings are also raised.
        Returns:
            (SCORES): Precision, recall, f1 scores for each NER type - and averaged scores (micro, macro etc)
        """
        if beta < 0:
            raise ValueError('beta should be >=0 in the F-beta score')

        average_options = (None, 'micro', 'macro', 'weighted')
        if average not in average_options:
            raise ValueError('average has to be one of {}'.format(average_options))
        # Calculate and store the number of the gold standard spans, predicted spans and true positives
        # for each NER type - this will be used to calculate the precision, recall and f1 scores
        pred_sum, tp_sum, true_sum = NoteSequenceEvaluation.extract_tp_actual_correct(
            note_predictions=note_predictions,
            note_tokens=note_tokens,
            note_spans=note_spans,
            ner_types=ner_types,
            scheme=scheme,
            mode=mode,
            suffix=suffix
        )

        if average == 'micro':
            tp_sum = np.array([tp_sum.sum()])
            pred_sum = np.array([pred_sum.sum()])
            true_sum = np.array([true_sum.sum()])

        # Finally, we have all our sufficient statistics. Divide! #
        beta2 = beta ** 2

        # Divide, and on zero-division, set scores and/or warn according to
        # zero_division:
        precision = NoteSequenceEvaluation._prf_divide(
            numerator=tp_sum,
            denominator=pred_sum,
            metric='precision',
            modifier='predicted',
            average=average,
            warn_for=warn_for,
            zero_division=zero_division
        )
        recall = NoteSequenceEvaluation._prf_divide(
            numerator=tp_sum,
            denominator=true_sum,
            metric='recall',
            modifier='true',
            average=average,
            warn_for=warn_for,
            zero_division=zero_division
        )

        # warn for f-score only if zero_division is warn, it is in warn_for
        # and BOTH precision and recall are ill-defined
        if zero_division == 'warn' and ('f-score',) == warn_for:
            if (pred_sum[true_sum == 0] == 0).any():
                NoteSequenceEvaluation._warn_prf(
                    average, 'true nor predicted', 'F-score is', len(true_sum)
                )

        # if tp == 0 F will be 1 only if all predictions are zero, all labels are
        # zero, and zero_division=1. In all other case, 0
        if np.isposinf(beta):
            f_score = recall
        else:
            denom = beta2 * precision + recall

            denom[denom == 0.] = 1  # avoid division by 0
            f_score = (1 + beta2) * precision * recall / denom

        # Average the results
        if average == 'weighted':
            weights = true_sum
            if weights.sum() == 0:
                zero_division_value = 0.0 if zero_division in ['warn', 0] else 1.0
                # precision is zero_division if there are no positive predictions
                # recall is zero_division if there are no positive labels
                # fscore is zero_division if all labels AND predictions are
                # negative
                return (
                    zero_division_value if pred_sum.sum() == 0 else 0.0,
                    zero_division_value,
                    zero_division_value if pred_sum.sum() == 0 else 0.0,
                    sum(true_sum)
                )

        elif average == 'samples':
            weights = sample_weight
        else:
            weights = None

        if average is not None:
            precision = np.average(precision, weights=weights)
            recall = np.average(recall, weights=weights)
            f_score = np.average(f_score, weights=weights)
            true_sum = sum(true_sum)

        return precision, recall, f_score, true_sum

    @staticmethod
    def classification_report(
            note_predictions,
            note_tokens,
            note_spans,
            ner_types: Sequence[str],
            scheme: Type[Token],
            mode: str,
            *,
            sample_weight: Optional[List[int]] = None,
            digits: int = 2,
            output_dict: bool = False,
            zero_division: str = 'warn',
            suffix: bool = False
    ) -> Union[str, dict]:
        """
        Build a text report showing the main tagging metrics.
        Args:
            note_predictions (Sequence[Sequence[str]]): The list of preditions in the evaluation dataset
            note_spans (Sequence[Sequence[Mapping[str, Union[str, int]]]]): The list of annotated spans for the notes 
                                                                            in the evaluation dataset 
            note_tokens (Sequence[Sequence[Mapping[str, Union[str, int]]]]): The list of tokens for the notes 
                                                                             in the evaluation dataset
            ner_types (Sequence[str]): The list of NER types e.g AGE, DATE etc
            scheme (Type[Token]): The NER labelling scheme
            mode (str): Whether to use default or strict evaluation
            suffix (str): Whether the B, I etc is in the prefix or the suffix
            sample_weight : array-like of shape (n_samples,), default=None
                Sample weights.
            digits (int): Number of digits for formatting output floating point values.
            output_dict (bool(default=False)): If True, return output as dict else str.
            zero_division : "warn", 0 or 1, default="warn"
                Sets the value to return when there is a zero division:
                   - recall: when there are no positive labels
                   - precision: when there are no positive predictions
                   - f-score: both
                If set to "warn", this acts as 0, but warnings are also raised.
        Returns:
            report : string/dict. Summary of the precision, recall, F1 score for each class.
        Examples:
            >>> from seqeval.metrics.v1 import classification_report
            >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> print(classification_report(y_true, y_pred))
                         precision    recall  f1-score   support
            <BLANKLINE>
                   MISC       0.00      0.00      0.00         1
                    PER       1.00      1.00      1.00         1
            <BLANKLINE>
              micro avg       0.50      0.50      0.50         2
              macro avg       0.50      0.50      0.50         2
           weighted avg       0.50      0.50      0.50         2
            <BLANKLINE>
        """
        NoteSequenceEvaluation.check_consistent_length(note_tokens, note_predictions)
        if len(note_spans) != len(note_tokens):
            raise ValueError('Number of spans and number of notes mismatch')

        if output_dict:
            reporter = DictReporter()
        else:
            name_width = max(map(len, ner_types))
            avg_width = len('weighted avg')
            width = max(name_width, avg_width, digits)
            reporter = StringReporter(width=width, digits=digits)

        # compute per-class scores.
        p, r, f1, s = NoteSequenceEvaluation.precision_recall_fscore(
            note_predictions=note_predictions,
            note_tokens=note_tokens,
            note_spans=note_spans,
            ner_types=ner_types,
            scheme=scheme,
            mode=mode,
            average=None,
            sample_weight=sample_weight,
            zero_division=zero_division,
            suffix=suffix
        )
        for row in zip(ner_types, p, r, f1, s):
            reporter.write(*row)
        reporter.write_blank()

        # compute average scores.
        average_options = ('micro', 'macro', 'weighted')
        for average in average_options:
            avg_p, avg_r, avg_f1, support = NoteSequenceEvaluation.precision_recall_fscore(
                note_predictions=note_predictions,
                note_tokens=note_tokens,
                note_spans=note_spans,
                ner_types=ner_types,
                scheme=scheme,
                mode=mode,
                average=average,
                sample_weight=sample_weight,
                zero_division=zero_division,
                suffix=suffix)
            reporter.write('{} avg'.format(average), avg_p, avg_r, avg_f1, support)
        reporter.write_blank()

        return reporter.report()

    @staticmethod
    def _prf_divide(
            numerator,
            denominator,
            metric,
            modifier,
            average,
            warn_for,
            zero_division='warn'
    ):
        """
        Performs division and handles divide-by-zero.
        On zero-division, sets the corresponding result elements equal to
        0 or 1 (according to ``zero_division``). Plus, if
        ``zero_division != "warn"`` raises a warning.
        The metric, modifier and average arguments are used only for determining
        an appropriate warning.
        """
        mask = denominator == 0.0
        denominator = denominator.copy()
        denominator[mask] = 1  # avoid infs/nans
        result = numerator / denominator

        if not np.any(mask):
            return result

        # if ``zero_division=1``, set those with denominator == 0 equal to 1
        result[mask] = 0.0 if zero_division in ['warn', 0] else 1.0

        # the user will be removing warnings if zero_division is set to something
        # different than its default value. If we are computing only f-score
        # the warning will be raised only if precision and recall are ill-defined
        if zero_division != 'warn' or metric not in warn_for:
            return result

        # build appropriate warning
        # E.g. "Precision and F-score are ill-defined and being set to 0.0 in
        # labels with no predicted samples. Use ``zero_division`` parameter to
        # control this behavior."

        if metric in warn_for and 'f-score' in warn_for:
            msg_start = '{0} and F-score are'.format(metric.title())
        elif metric in warn_for:
            msg_start = '{0} is'.format(metric.title())
        elif 'f-score' in warn_for:
            msg_start = 'F-score is'
        else:
            return result

        NoteSequenceEvaluation._warn_prf(average, modifier, msg_start, len(result))

        return result

    @staticmethod
    def _warn_prf(average, modifier, msg_start, result_size):
        axis0, axis1 = 'sample', 'label'
        if average == 'samples':
            axis0, axis1 = axis1, axis0
        msg = ('{0} ill-defined and being set to 0.0 {{0}} '
               'no {1} {2}s. Use `zero_division` parameter to control'
               ' this behavior.'.format(msg_start, modifier, axis0))
        if result_size == 1:
            msg = msg.format('due to')
        else:
            msg = msg.format('in {0}s with'.format(axis1))
        warnings.warn(msg, UndefinedMetricWarning, stacklevel=2)

    @staticmethod
    def check_consistent_length(
            note_tokens: Sequence[Sequence[Mapping[str, Union[str, int]]]],
            note_predictions: Sequence[Sequence[str]]
    ):
        """
        Check that all arrays have consistent first and second dimensions.
        Checks whether all objects in arrays have the same shape or length.
        Args:
            y_true : 2d array.
            y_pred : 2d array.
        """
        len_tokens = list(map(len, note_tokens))
        len_predictions = list(map(len, note_predictions))

        if len(note_tokens) != len(note_predictions) or len_tokens != len_predictions:
            message = 'Found input variables with inconsistent numbers of samples:\n{}\n{}'.format(len_tokens,
                                                                                                   len_predictions)
            raise ValueError(message)
