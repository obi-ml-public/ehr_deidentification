from typing import Optional
from dataclasses import dataclass, field

@dataclass
class EvaluationArguments:
    """
    Arguments pertaining to the evaluation process.
    """
    model_eval_script: Optional[str] = field(
        default=None,
        metadata={"help": "The script that is used for evaluation"},
    )
    evaluation_mode: Optional[str] = field(
        default=None,
        metadata={"help": "Strict or default mode for sequence evaluation"},
    )
    validation_spans_file: Optional[str] = field(
        default=None,
        metadata={"help": "A span evaluation data file to evaluate on span level (json file). This will contain a "
                          "mapping between the note_ids and note spans"},
    )
    ner_type_maps: Optional[str] = field(
        default=None,
        metadata={"help": "List that contains the mappings between the original NER types to another set of NER "
                          "types. Used mainly for evaluation. to map ner token labels to another set of ner token"},
    )