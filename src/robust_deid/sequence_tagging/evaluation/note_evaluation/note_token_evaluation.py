from collections import Counter
from typing import Sequence, List, Tuple, Union, Type, Optional

from seqeval.reporters import DictReporter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class NoteTokenEvaluation(object):
    """
    This class is used to evaluate token level precision, recall and F1 scores.
    Script to evaluate at a token level. Calculate precision, recall, and f1 metrics
    at the token level rather than the span level.
    """

    @staticmethod
    def unpack_nested_list(nested_list: Sequence[Sequence[str]]) -> List[str]:
        """
        Use this function to unpack a nested list and also for token level evaluation we dont
        need to consider the B, I prefixes (depending on the NER notation, so remove that as well.
        Args:
            nested_list (Sequence[Sequence[str]]): A nested list of predictions/labels
        Returns:
            (List[str]): Unpacked nested list of predictions/labels
        """
        return [inner if inner == 'O' else inner[2:] for nested in nested_list for inner in nested]

    @staticmethod
    def get_counts(sequence: Sequence[str], ner_types: Sequence[str]) -> List[int]:
        """
        Use this function to get the counts for each NER type
        Args:
            ner_list (Sequence[str]): A list of the NER labels/predicitons
        Returns:
            (List[int]): Position 0 contains the counts for the NER type that corresponds to position 0
        """
        counts = Counter()
        counts.update(sequence)
        return [counts[ner_type] for ner_type in ner_types]

    @staticmethod
    def precision_recall_fscore(
            labels: Sequence[str],
            predictions: Sequence[str],
            ner_types: Sequence[str],
            average: Optional[str] = None
    ) -> Tuple[Union[float, List[float]], Union[float, List[float]], Union[float, List[float]], Union[int, List[int]]]:
        """
        Use this function to get the token level precision, recall and fscore. Internally we use the
        sklearn precision_score, recall_score and f1 score functions. Also return the count of each
        NER type.
        Args:
            labels (Sequence[str]): A list of the gold standard NER labels
            predictions (Sequence[str]): A list of the predicted NER labels
            average (Optional[str]): None for per NER types scores, or pass an appropriate average value
        Returns:
            eval_precision (Union[float, List[float]]): precision score (averaged or per ner type)
            eval_precision (Union[float, List[float]]): recall score (averaged or per ner type)
            eval_precision (Union[float, List[float]]): F1 score (averaged or per ner type)
            counts (Union[int, List[int]]): Counts (total or per ner type)
        """
        eval_precision = precision_score(y_true=labels, y_pred=predictions, labels=ner_types, average=average)
        eval_recall = recall_score(y_true=labels, y_pred=predictions, labels=ner_types, average=average)
        eval_f1 = f1_score(y_true=labels, y_pred=predictions, labels=ner_types, average=average)
        counts = NoteTokenEvaluation.get_counts(sequence=labels, ner_types=ner_types)
        if (average == None):
            eval_precision = list(eval_precision)
            eval_recall = list(eval_recall)
            eval_f1 = list(eval_f1)
        else:
            counts = sum(counts)
        return eval_precision, eval_recall, eval_f1, counts

    @staticmethod
    def get_confusion_matrix(labels: Sequence[str], predictions: Sequence[str], ner_types: Sequence[str]):
        """
        Use this function to get the token level precision, recall and fscore per NER type
        and also the micro, macro and weighted averaged precision, recall and f scores.
        Essentially we return a classification report
        Args:
            labels (Sequence[str]): A list of the gold standard NER labels
            predictions (Sequence[str]): A list of the predicted NER labels
        Returns:
            (Type[DictReporter]): Classification report
        """
        labels = NoteTokenEvaluation.unpack_nested_list(labels)
        predictions = NoteTokenEvaluation.unpack_nested_list(predictions)
        return confusion_matrix(y_true=labels, y_pred=predictions, labels=ner_types + ['O', ])

    @staticmethod
    def classification_report(
            labels: Sequence[Sequence[str]],
            predictions: Sequence[Sequence[str]],
            ner_types: Sequence[str]
    ) -> Type[DictReporter]:
        """
        Use this function to get the token level precision, recall and fscore per NER type
        and also the micro, macro and weighted averaged precision, recall and f scores.
        Essentially we return a classification report which contains all this information
        Args:
            labels (Sequence[Sequence[str]]): A list of the gold standard NER labels for each note
            predictions (Sequence[Sequence[str]]): A list of the predicted NER labels for each note
        Returns:
            (Type[DictReporter]): Classification report that contains the token level metric scores
        """
        # Unpack the nested lists (labels and predictions) before running the evaluation metrics
        labels = NoteTokenEvaluation.unpack_nested_list(nested_list=labels)
        predictions = NoteTokenEvaluation.unpack_nested_list(nested_list=predictions)
        # Store results in this and return this object
        reporter = DictReporter()
        # Calculate precision, recall and f1 for each NER type
        eval_precision, eval_recall, eval_f1, counts = NoteTokenEvaluation.precision_recall_fscore(
            labels=labels,
            predictions=predictions,
            ner_types=ner_types,
            average=None
        )
        # Store the results
        for row in zip(ner_types, eval_precision, eval_recall, eval_f1, counts):
            reporter.write(*row)
        reporter.write_blank()
        # Calculate the overall precision, recall and f1 - based on the defined averages
        average_options = ('micro', 'macro', 'weighted')
        for average in average_options:
            eval_precision, eval_recall, eval_f1, counts = NoteTokenEvaluation.precision_recall_fscore(
                labels=labels,
                predictions=predictions,
                ner_types=ner_types,
                average=average
            )
            # Store the results
            reporter.write('{} avg'.format(average), eval_precision, eval_recall, eval_f1, counts)
        reporter.write_blank()
        # Return the token level results
        return reporter.report()
