""" modified seqeval metric. """
from typing import Sequence, List, Optional, Type, Union, Mapping, Dict

# This script uses the two other scripts note_sequence_evaluation.py
# and note_token_evalaution.py to gather the span level and token
# level metrics during the evaluation phase in the huggingface
# training process. More information on how this script works
# can be found in - https://github.com/huggingface/datasets/tree/master/metrics/seqeval
# The code is borrowed from there and minor changes are made - to include token
# level metrics and evaluating spans at the character level as opposed to the
# token level
import datasets

from .note_sequence_evaluation import NoteSequenceEvaluation
from .note_token_evaluation import NoteTokenEvaluation
from .violations import Violations

_CITATION = """\
@inproceedings{ramshaw-marcus-1995-text,
    title = "Text Chunking using Transformation-Based Learning",
    author = "Ramshaw, Lance  and
      Marcus, Mitch",
    booktitle = "Third Workshop on Very Large Corpora",
    year = "1995",
    url = "https://www.aclweb.org/anthology/W95-0107",
}
@misc{seqeval,
  title={{seqeval}: A Python framework for sequence labeling evaluation},
  url={https://github.com/chakki-works/seqeval},
  note={Software available from https://github.com/chakki-works/seqeval},
  author={Hiroki Nakayama},
  year={2018},
}
"""

_DESCRIPTION = """seqeval is a Python framework for sequence labeling evaluation. seqeval can evaluate the 
performance of chunking tasks such as named-entity recognition, part-of-speech tagging, semantic role labeling and so 
on. This is well-tested by using the Perl script conlleval, which can be used for measuring the performance of a 
system that has processed the CoNLL-2000 shared task data. seqeval supports following formats: IOB1 IOB2 IOE1 IOE2 
IOBES See the [README.md] file at https://github.com/chakki-works/seqeval for more information. """

_KWARGS_DESCRIPTION = """
Produces labelling scores along with its sufficient statistics
from a source against one or more references.
Args:
    predictions: List of List of predicted labels (Estimated targets as returned by a tagger)
    references: List of List of reference labels (Ground truth (correct) target values)
    suffix: True if the IOB prefix is after type, False otherwise. default: False
Returns:
    'scores': dict. Summary of the scores for overall and per type
        Overall:
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': F1 score, also known as balanced F-score or F-measure,
        Per type:
            'precision': precision,
            'recall': recall,
            'f1': F1 score, also known as balanced F-score or F-measure
Examples:
    >>> predictions = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    >>> references = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    >>> seqeval = datasets.load_metric("seqeval")
    >>> results = seqeval.compute(predictions=predictions, references=references)
    >>> print(list(results.keys()))
    ['MISC', 'PER', 'overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']
    >>> print(results["overall_f1"])
    0.5
    >>> print(results["PER"]["f1"])
    1.0
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NoteEvaluation(datasets.Metric):

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            features=datasets.Features(
                {
                    "references": datasets.Sequence(datasets.Value("string", id="label"), id="sequence"),
                    "predictions": datasets.Sequence(datasets.Value("string", id="label"), id="sequence"),
                }
            ),
            inputs_description=_KWARGS_DESCRIPTION
        )

    def _compute(
            self,
            references: Sequence[Sequence[str]],
            predictions: Sequence[Sequence[str]],
            note_tokens: Sequence[Sequence[Mapping[str, Union[str, int]]]],
            note_spans: Sequence[Sequence[Mapping[str, Union[str, int]]]],
            ner_types: Sequence[str],
            ner_description: str,
            notation: str,
            scheme: str,
            mode: str,
            confusion_matrix: bool = False,
            suffix: bool = False,
            sample_weight: Optional[List[int]] = None,
            zero_division: Union[str, int] = "warn",
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Use the NoteSequenceEvaluation and NoteTokenEvaluation classes to extract the
        token and span level precision, recall and f1 scores. Also return the micro averaged
        precision recall and f1 scores
        Args:
            references (Sequence[Sequence[str]]): The list of annotated labels in the evaluation dataset
            predictions (Sequence[Sequence[str]]): The list of predictions in the evaluation dataset
            note_tokens (Sequence[Sequence[Mapping[str, Union[str, int]]]]): The list of tokens for the notes 
                                                                             in the evaluation dataset
            note_spans (Sequence[Sequence[Mapping[str, Union[str, int]]]]): The list of annotated spans for the notes 
                                                                            in the evaluation dataset 
            ner_types (Sequence[str]): The list of NER types e.g AGE, DATE etc
            ner_description (str): A prefix added to the evaluation result keys
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
            (Dict[str, Dict[str, Union[int, float]]]): The token and span level metric scores
        """
        # Span level metrics scores
        report = NoteSequenceEvaluation.classification_report(
            note_predictions=predictions,
            note_tokens=note_tokens,
            note_spans=note_spans,
            ner_types=ner_types,
            scheme=scheme,
            mode=mode,
            suffix=suffix,
            output_dict=True,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        # Token level metric scores
        token_report = NoteTokenEvaluation.classification_report(
            labels=references,
            predictions=predictions,
            ner_types=ner_types
        )
        violation_count = sum([Violations.get_violations(tag_sequence=prediction, notation=notation)
                               for prediction in predictions])
        # Remove the macro and weighted average results
        macro_score = report.pop("macro avg")
        report.pop("weighted avg")
        macro_token_score = token_report.pop("macro avg")
        token_report.pop("weighted avg")
        overall_score = report.pop("micro avg")
        token_overall_score = token_report.pop("micro avg")
        # Extract span level scores for each NER type
        scores = {
            type_name: {
                "precision": score["precision"],
                "recall": score["recall"],
                "f1": score["f1-score"],
                "number": score["support"],
            }
            for type_name, score in report.items()
        }
        # Extract token level scores for each NER type
        token_scores = {
            type_name + '-TOKEN': {
                "precision": score["precision"],
                "recall": score["recall"],
                "f1": score["f1-score"],
                "number": score["support"],
            }
            for type_name, score in token_report.items()
        }
        # Extract micro averaged span level score
        overall = {'overall' + ner_description:
                       {"precision": overall_score["precision"],
                        "recall": overall_score["recall"],
                        "f1": overall_score["f1-score"],
                        }
                   }
        # Extract micro averaged token level score
        token_overall = {'token-overall' + ner_description:
                             {"precision": token_overall_score["precision"],
                              "recall": token_overall_score["recall"],
                              "f1": token_overall_score["f1-score"],
                              }
                         }
        # Extract macro averaged token level score
        macro_overall = {'macro-overall' + ner_description:
                             {"precision": macro_score["precision"],
                              "recall": macro_score["recall"],
                              "f1": macro_score["f1-score"],
                              }
                         }
        # Extract macro averaged token level score
        macro_token_overall = {'macro-token-overall' + ner_description:
                                   {"precision": macro_token_score["precision"],
                                    "recall": macro_token_score["recall"],
                                    "f1": macro_token_score["f1-score"],
                                    }
                               }
        # Store number of NER violations
        violation_count = {'violations' + ner_description: {'count': violation_count}}
        # Return the results
        if confusion_matrix:
            confusion_matrix = {'confusion' + ner_description:
                {'matrix': NoteTokenEvaluation.get_confusion_matrix(
                    labels=references,
                    predictions=predictions,
                    ner_types=ner_types
                )}}
            return {**scores, **overall, **token_scores, **token_overall, **macro_overall, **macro_token_overall,
                    **violation_count, **confusion_matrix}
        else:
            return {**scores, **overall, **token_scores, **token_overall, **macro_overall, **macro_token_overall,
                    **violation_count}
