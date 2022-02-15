from typing import Sequence, Tuple, Dict, NoReturn, Mapping, Union, Type

from seqeval.scheme import IOB1, IOB2, IOBES, BILOU


class MetricsCompute(object):
    """
    This is the evaluation script which is passed to the huggingface
    trainer - specifically the compute_metrics function. The trainer uses
    this function to run the evaluation on the validation dataset and log/save
    the metrics. This script is used to evaluate the token and span level metrics
    on the validation dataset by the huggingface trainer. The evaluation is also run
    on the NER labels and spans produced by the different label mapper
    objects. For example we might run the evaluation on the original list of NER labels/spans
    and we also run the evaluation on binary HIPAA labels/spans. This is done by mapping the
    NER labels & spans using the list of label_mapper object present in label_mapper_list
    The same evaluation script and metrics are first run on the original ner types/labels/spans
    e.g: 
    [AGE, STAFF, DATE], [B-AGE, O, O, U-LOC, B-DATE, L-DATE, O, B-STAFF, I-STAFF, L-STAFF], 
    [{start:0, end:5, label: AGE}, {start:17, end:25, label: LOC}, {start:43, end:54, label: DATE}, 
    {start:77, end:84, label: STAFF}] 
    and we also run on some mapped version of these ner types/labels/spans shown below
    [HIPAA], [B-HIPAA, O, O, U-HIPAA, B-HIPAA, I-HIPAA, O, O, O, O], [{start:0, end:5, label: HIPAA},
    {start:17, end:25, label: HIPAA}, {start:43, end:54, label: HIPAA}, {start:77, end:84, label: O}] 
    The context and subword tokens are excluded from the evaluation process
    The results are returned - which are saved and logged
    """

    def __init__(
            self,
            metric,
            note_tokens: Sequence[Sequence[Mapping[str, Union[str, int]]]],
            note_spans: Sequence[Sequence[Mapping[str, Union[str, int]]]],
            label_mapper_list: Sequence,
            post_processor,
            note_level_aggregator,
            notation: str,
            mode: str,
            confusion_matrix: bool = False,
            format_results: bool = True
    ) -> NoReturn:
        """
        Initialize the variables used ot perform evaluation. The evaluation object.
        How the model predictions are decoded (e.g argmax, crf). The post processor object
        also handles excluding context and subword tokens are excluded from the evaluation process
        The notation, evaluation mode label maps. The note_tokens is used in the span level evaluation 
        process to check the character position of each token - and check if they match with the character 
        position of the spans. The note_spans are also used in the span level evaluation process, they contain 
        the position and labels of the spans.
        Args:
            metric (): The huggingface metric object, which contains the span and token level evaluation code
            note_tokens (Sequence[Sequence[Mapping[str, Union[str, int]]]]): The list of tokens in the entire dataset
            note_spans (Sequence[Sequence[Mapping[str, Union[str, int]]]]): The list of note spans in the entire dataset
            post_processor (): Post processing the predictions (logits) - argmax, or crf etc. The prediction logits are
                               passed to this object, which is then processed using the argmax of the logits or a
                               crf function to return the sequence of NER labels
            note_level_aggregator (): Aggregate sentence level predictions back to note level for evaluation
                                      using this object
            label_mapper_list (Sequence): The list of label mapper object that are used to map ner spans and 
                                          labels for evaluation 
            notation (str): The NER notation
            mode (str): The span level eval mode - strict or default
            format_results (bool): Format the results - return either a single dict (true) or a dict of dicts (false)
        """
        self._metric = metric
        self._note_tokens = note_tokens
        self._note_spans = note_spans
        self._label_mapper_list = label_mapper_list
        self._note_level_aggregator = note_level_aggregator
        self._notation = notation
        self._scheme = MetricsCompute.get_scheme(self._notation)
        self._mode = mode
        self._post_processor = post_processor
        self._confusion_matrix = confusion_matrix
        self._format_results = format_results

    @staticmethod
    def get_scheme(notation: str) -> Union[Type[IOB2], Type[IOBES], Type[BILOU], Type[IOB1]]:
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

    def run_metrics(
            self,
            note_labels: Sequence[Sequence[str]],
            note_predictions: Sequence[Sequence[str]]
    ) -> Union[Dict[str, Union[int, float]], Dict[str, Dict[str, Union[int, float]]]]:
        """
        Run the evaluation metrics and return the span and token level results.
        The metrics are run for each mapping of ner labels - based on the object in the
        label_mapper_list. The evaluation is also run on the NER labels and spans produced by the different 
        label mapper objects. For example we might run the evaluation on the original list of NER labels/spans
        and we also run the evaluation on binary HIPAA labels/spans. This is done by mapping the
        NER labels & spans using the list of label_mapper object present in label_mapper_list
        The same evaluation script and metrics are first run on the original ner types/labels/spans
        e.g: 
        [AGE, STAFF, DATE], [B-AGE, O, O, U-LOC, B-DATE, L-DATE, O, B-STAFF, I-STAFF, L-STAFF], 
        [{start:0, end:5, label: AGE}, {start:17, end:25, label: LOC}, {start:43, end:54, label: DATE}, 
        {start:77, end:84, label: STAFF}] 
        and we also run on some mapped version of these ner types/labels/spans shown below
        [HIPAA], [B-HIPAA, O, O, U-HIPAA, B-HIPAA, I-HIPAA, O, O, O, O], [{start:0, end:5, label: HIPAA},
        {start:17, end:25, label: HIPAA}, {start:43, end:54, label: HIPAA}, {start:77, end:84, label: O}]
        Args:
            note_labels (Sequence[Sequence[str]]): The list of NER labels for each note
            note_predictions (Sequence[Sequence[str]]): The list of NER predictions for each notes
        Returns:
            final_results (Union[Dict[str, Union[int, float]], Dict[str, Dict[str, Union[int, float]]]]): Span and token
                                                                                                          level
                                                                                                          metric results
        """
        final_results = {}
        # Go through the list of different mapping (e.g HIPAA/I2B2)
        for label_mapper in self._label_mapper_list:
            # Get the NER information
            ner_types = label_mapper.get_ner_types()
            ner_description = label_mapper.get_ner_description()
            # Map the NER labels and spans
            predictions = [label_mapper.map_sequence(prediction) for prediction in note_predictions]
            labels = [label_mapper.map_sequence(label) for label in note_labels]
            spans = [label_mapper.map_spans(span) for span in self._note_spans]
            # Run the span level and token level evaluation metrics
            results = self._metric.compute(
                predictions=predictions,
                references=labels,
                note_tokens=self._note_tokens,
                note_spans=spans,
                ner_types=ner_types,
                ner_description=ner_description,
                notation=self._notation,
                scheme=self._scheme,
                mode=self._mode,
                confusion_matrix=self._confusion_matrix
            )
            # Return the results as a single mapping or a nested mapping
            if not self._format_results:
                for key, value in results.items():
                    final_results[key] = value
            else:
                for key, value in results.items():
                    if isinstance(value, dict):
                        for n, v in value.items():
                            final_results[f"{key}_{n}"] = v
                    else:
                        final_results[key] = value
        # Return the results
        return final_results

    def compute_metrics(
            self,
            p: Tuple[Sequence[Sequence[str]], Sequence[Sequence[str]]]
    ) -> Union[Dict[str, Union[int, float]], Dict[str, Dict[str, Union[int, float]]]]:
        """
        This script is used to compute the token and span level metrics when
        the predictions and labels are passed. The first step is to convert the
        model logits into the sequence of NER predictions using the post_processor
        object (argmax, crf etc) and also exclude any context and subword tokens from the
        evaluation process. Once we have the NER labels and predictions we run
        the span and token level evaluation.
        The evaluation is also run on the NER labels and spans produced by the different label mapper
        objects. For example we might run the evaluation on the original list of NER labels/spans
        and we also run the evaluation on binary HIPAA labels/spans. This is done by mapping the
        NER labels & spans using the list of label_mapper object present in label_mapper_list
        The same evaluation script and metrics are first run on the original ner types/labels/spans
        e.g: 
        [AGE, STAFF, DATE], [B-AGE, O, O, U-LOC, B-DATE, L-DATE, O, B-STAFF, I-STAFF, L-STAFF], 
        [{start:0, end:5, label: AGE}, {start:17, end:25, label: LOC}, {start:43, end:54, label: DATE}, 
        {start:77, end:84, label: STAFF}] 
        and we also run on some mapped version of these ner types/labels/spans shown below
        [HIPAA], [B-HIPAA, O, O, U-HIPAA, B-HIPAA, I-HIPAA, O, O, O, O], [{start:0, end:5, label: HIPAA},
        {start:17, end:25, label: HIPAA}, {start:43, end:54, label: HIPAA}, {start:77, end:84, label: O}]
        Run the evaluation metrics and return the span and token level results.
        The metrics are run for each mapping of ner labels - based on the object in the
        label_mapper_list
        Args:
            p (Tuple[Sequence[Sequence[str]], Sequence[Sequence[str]]]): Tuple of model logits and labels
        Returns:
            final_results (Union[Dict[str, Union[int, float]], Dict[str, Dict[str, Union[int, float]]]]): Span and token
                                                                                                          level
                                                                                                          metric results
        """
        predictions, labels = p
        # Convert the logits (scores) to predictions
        true_predictions, true_labels = self._post_processor.decode(predictions, labels)
        # Aggregate sentence level labels and predictions to note level for evaluation
        note_predictions = self._note_level_aggregator.get_aggregate_sequences(true_predictions)
        note_labels = self._note_level_aggregator.get_aggregate_sequences(true_labels)
        # Return results
        return self.run_metrics(note_labels, note_predictions)
