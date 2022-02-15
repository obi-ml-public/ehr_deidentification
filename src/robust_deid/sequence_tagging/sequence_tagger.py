# Train a model using the huggingface library
# The datasets have been built using the scripts in the ner_datasets folder
# these datasets will be used as input to the model.
import os
import sys
import json
import logging
from typing import Optional, Sequence

import datasets
import transformers
from datasets import load_dataset, load_metric
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.13.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

from .models.hf import ModelPicker
from .evaluation import MetricsCompute
from .note_aggregate import NoteLevelAggregator
from .post_process.model_outputs import PostProcessPicker
from .dataset_builder import DatasetTokenizer, LabelMapper, NERDataset, NERLabels
from .arguments import ModelArguments, DataTrainingArguments, EvaluationArguments


class SequenceTagger(object):

    def __init__(
            self,
            task_name,
            notation,
            ner_types,
            model_name_or_path,
            config_name: Optional[str] = None,
            tokenizer_name: Optional[str] = None,
            post_process: str = 'argmax',
            cache_dir: Optional[str] = None,
            model_revision: str = 'main',
            use_auth_token: bool = False,
            threshold: Optional[float] = None,
            do_lower_case=False,
            fp16: bool = False,
            seed: int = 41,
            local_rank: int = - 1
    ):
        self._task_name = task_name
        self._notation = notation
        self._ner_types = ner_types
        self._model_name_or_path = model_name_or_path
        self._config_name = config_name if config_name else self._model_name_or_path
        self._tokenizer_name = tokenizer_name if tokenizer_name else self._model_name_or_path
        self._post_process = post_process
        self._cache_dir = cache_dir
        self._model_revision = model_revision
        self._use_auth_token = use_auth_token
        ner_labels = NERLabels(notation=self._notation, ner_types=self._ner_types)
        self._label_list = ner_labels.get_label_list()
        self._label_to_id = ner_labels.get_label_to_id()
        self._id_to_label = ner_labels.get_id_to_label()
        self._config = self.__get_config()
        self._tokenizer = self.__get_tokenizer(do_lower_case=do_lower_case)
        self._model, self._post_processor = self.__get_model(threshold=threshold)
        self._dataset_tokenizer = None
        # Data collator
        self._data_collator = DataCollatorForTokenClassification(
            self._tokenizer,
            pad_to_multiple_of=8 if fp16 else None
        )
        self._metrics_compute = None
        self._train_dataset = None
        self._eval_dataset = None
        self._test_dataset = None
        self._trainer = None
        # Setup logging
        self._logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        log_level = logging.INFO if is_main_process(local_rank) else logging.WARN
        self._logger.setLevel(log_level)
        # Set the verbosity to info of the Transformers logger (on main process only):
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        # Set seed before initializing model.
        self._seed = seed
        set_seed(self._seed)

    def load(
            self,
            text_column_name: str = 'tokens',
            label_column_name: str = 'labels',
            pad_to_max_length: bool = False,
            truncation: bool = True,
            max_length: int = 512,
            is_split_into_words: bool = True,
            label_all_tokens: bool = False,
            token_ignore_label: str = 'NA'
    ):
        # This following two lines of code is the one that is used to read the input dataset
        # Run the subword tokenization on the pre-split tokens and then
        # as mentioned above align the subtokens and labels and add the ignore
        # label. This will read the input - say [60, year, old, in, 2080]
        # and will return the subtokens - [60, year, old, in, 208, ##0]
        # some other information like token_type_ids etc
        # and the labels [0, 20, 20, 20, 3, -100] (0 - corresponds to B-AGE, 20 corresponds to O
        # and 3 corresponds to B-DATE. This returned input serves as input for training the model
        # or for gathering predictions from a trained model.
        # Another important thing to note is that we have mentioned before that
        # we add chunks of tokens that appear before and after the current chunk for context. We would
        # also need to assign the label -100 (ignore_label) to these chunks, since we are using them
        # only to provide context. For example the input would be something
        # like tokens: [James, Doe, 60, year, old, in, 2080, BWH, tomorrow, only],
        # labels: [NA, NA, B-DATE, O, O, O, B-DATE, NA, NA, NA]. NA represents the tokens used for context
        # This function would return some tokenizer info (e.g attention mask etc), along with
        # the information that maps the tokens to the subtokens -
        # [James, Doe, 60, year, old, in, 208, ##0, BW, ##h, tomorrow, only]
        # and the labels - [-100, -100, 0, 20, 20, 20, 3, -100, -100, -100]
        # (if label_all_tokens was true, we would return [-100, -100, 0, 20, 20, 20, 3, 3, -100, -100]).
        # Create an object that has the tokenize_and_align_labels function to perform
        # the operation described above
        # Map that sends B-Xxx label to its I-Xxx counterpart
        b_to_i_label = []
        if label_all_tokens:
            if self._notation != 'BIO':
                raise ValueError('Label all tokens works only with BIO notation!')
            b_to_i_label = []
            for idx, label in enumerate(self._label_list):
                if label.startswith("B-") and label.replace("B-", "I-") in self._label_list:
                    b_to_i_label.append(self._label_list.index(label.replace("B-", "I-")))
                else:
                    b_to_i_label.append(idx)
        # Padding strategy
        padding = "max_length" if pad_to_max_length else False
        self._dataset_tokenizer = DatasetTokenizer(
            tokenizer=self._tokenizer,
            token_column=text_column_name,
            label_column=label_column_name,
            label_to_id=self._label_to_id,
            b_to_i_label=b_to_i_label,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            is_split_into_words=is_split_into_words,
            label_all_tokens=label_all_tokens,
            token_ignore_label=token_ignore_label
        )

    def set_train(
            self,
            train_file: str,
            max_train_samples: Optional[int] = None,
            preprocessing_num_workers: Optional[int] = None,
            overwrite_cache: bool = False,
            file_extension: str = 'json',
            shuffle: bool = True,
    ):
        if shuffle:
            train_dataset = load_dataset(
                file_extension,
                data_files={'train': train_file},
                cache_dir=self._cache_dir
            ).shuffle(seed=self._seed)
        else:
            train_dataset = load_dataset(
                file_extension,
                data_files={'train': train_file},
                cache_dir=self._cache_dir
            )
        train_dataset = train_dataset['train']
        # Run the tokenizer (subword), tokenize and align the labels as mentioned above on
        # every example (row) of the dataset - (map function). This tokenized_datasets will be the
        # input to the model (either for training or predictions
        if max_train_samples is not None:
            train_dataset = train_dataset.select(range(max_train_samples))
        self._train_dataset = train_dataset.map(
            self._dataset_tokenizer.tokenize_and_align_labels,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
        )

    def set_eval(
            self,
            validation_file: str,
            max_val_samples: Optional[int] = None,
            preprocessing_num_workers: Optional[int] = None,
            overwrite_cache: bool = False,
            file_extension: str = 'json',
            shuffle: bool = True,
    ):
        if shuffle:
            eval_dataset = load_dataset(
                file_extension,
                data_files={'eval': validation_file},
                cache_dir=self._cache_dir
            ).shuffle(seed=self._seed)
        else:
            eval_dataset = load_dataset(
                file_extension,
                data_files={'eval': validation_file},
                cache_dir=self._cache_dir
            )
        eval_dataset = eval_dataset['eval']
        # Eval
        if max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(max_val_samples))
        self._eval_dataset = eval_dataset.map(
            self._dataset_tokenizer.tokenize_and_align_labels,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
        )

    def set_predict(
            self,
            test_file: str,
            max_test_samples: Optional[int] = None,
            preprocessing_num_workers: Optional[int] = None,
            overwrite_cache: bool = False,
            file_extension: str = 'json',
            shuffle: bool = False,
    ):
        if shuffle:
            test_dataset = load_dataset(
                file_extension,
                data_files={'test': test_file},
                cache_dir=self._cache_dir
            ).shuffle(seed=self._seed)
        else:
            test_dataset = load_dataset(
                file_extension,
                data_files={'test': test_file},
                cache_dir=self._cache_dir
            )
        test_dataset = test_dataset['test']
        # Eval
        if max_test_samples is not None:
            test_dataset = test_dataset.select(range(max_test_samples))
        self._test_dataset = test_dataset.map(
            self._dataset_tokenizer.tokenize_and_align_labels,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
        )

    def set_eval_metrics(
            self,
            validation_spans_file: str,
            model_eval_script: str = './evaluation/note_evaluation.py',
            ner_types_maps: Optional[Sequence[Sequence[str]]] = None,
            evaluation_mode: str = 'strict'
    ):

        if self._eval_dataset is None:
            raise ValueError("Validation data not present")

        validation_ids = [json.loads(line)['note_id'] for line in open(validation_spans_file, 'r')]
        validation_spans = [json.loads(line)['note_spans'] for line in open(validation_spans_file, 'r')]
        descriptions = ['']
        type_maps = [self._ner_types]
        if ner_types_maps is not None:
            descriptions += [''.join(list(set(ner_types_map) - set('O'))) for ner_types_map in ner_types_maps]
            type_maps += ner_types_maps
        label_mapper_list = [LabelMapper(
            notation=self._notation,
            ner_types=self._ner_types,
            ner_types_maps=ner_types_map,
            description=description
        ) for ner_types_map, description in zip(type_maps, descriptions)]
        # Use this to aggregate sentences back to notes for validation
        note_level_aggregator = NoteLevelAggregator(
            note_ids=validation_ids,
            note_sent_info=self._eval_dataset['note_sent_info']
        )
        note_tokens = note_level_aggregator.get_aggregate_sequences(
            sequences=self._eval_dataset['current_sent_info']
        )
        self._metrics_compute = MetricsCompute(
            metric=load_metric(model_eval_script),
            note_tokens=note_tokens,
            note_spans=validation_spans,
            label_mapper_list=label_mapper_list,
            post_processor=self._post_processor,
            note_level_aggregator=note_level_aggregator,
            notation=self._notation,
            mode=evaluation_mode,
            confusion_matrix=False,
            format_results=True
        )

    def setup_trainer(self, training_args):
        # Log on each process the small summary:
        self._logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        self._logger.info(f"Training/evaluation parameters {training_args}")
        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(
                training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                self._logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        # Initialize our Trainer
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=self._train_dataset,
            eval_dataset=self._eval_dataset,
            tokenizer=self._tokenizer,
            data_collator=self._data_collator,
            compute_metrics=None if self._metrics_compute is None else self._metrics_compute.compute_metrics,
        )
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        return checkpoint

    def train(self, checkpoint: None):
        if self._train_dataset is not None and self._trainer is not None:
            train_result = self._trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            self._trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics["train_samples"] = len(self._train_dataset)

            self._trainer.log_metrics("train", metrics)
            self._trainer.save_metrics("train", metrics)
            self._trainer.save_state()
        else:
            if self._trainer is None:
                raise ValueError('Trainer not setup - Run setup_trainer')
            else:
                raise ValueError('Train data not setup - Run set_train')
        return metrics

    def evaluate(self):
        if self._eval_dataset is not None and self._trainer is not None:
            # Evaluation
            self._logger.info("*** Evaluate ***")
            metrics = self._trainer.evaluate()
            metrics["eval_samples"] = len(self._eval_dataset)
            self._trainer.log_metrics("eval", metrics)
            self._trainer.save_metrics("eval", metrics)
        else:
            if self._trainer is None:
                raise ValueError('Trainer not setup - Run setup_trainer')
            else:
                raise ValueError('Evaluation data not setup - Run set_eval')
        return metrics

    def predict(self, output_predictions_file: Optional[str] = None):
        if self._test_dataset is not None and self._trainer is not None:
            self._logger.info("*** Predict ***")
            predictions, labels, metrics = self._trainer.predict(self._test_dataset, metric_key_prefix="predict")
            unique_note_ids = set()
            for note_sent_info in self._test_dataset['note_sent_info']:
                note_id = note_sent_info['note_id']
                unique_note_ids = unique_note_ids | {note_id}
            note_ids = list(unique_note_ids)
            note_level_aggregator = NoteLevelAggregator(
                note_ids=note_ids,
                note_sent_info=self._test_dataset['note_sent_info']
            )
            note_tokens = note_level_aggregator.get_aggregate_sequences(
                sequences=self._test_dataset['current_sent_info']
            )
            true_predictions, true_labels = self._post_processor.decode(predictions, labels)
            note_predictions = note_level_aggregator.get_aggregate_sequences(sequences=true_predictions)
            note_labels = note_level_aggregator.get_aggregate_sequences(sequences=true_labels)
            self._trainer.log_metrics("test", metrics)
            self._trainer.save_metrics("test", metrics)
            if output_predictions_file is None:
                return SequenceTagger.__get_predictions(
                    note_ids,
                    note_tokens,
                    note_labels,
                    note_predictions
                )
            else:
                SequenceTagger.__write_predictions(
                    output_predictions_file,
                    note_ids,
                    note_tokens,
                    note_labels,
                    note_predictions
                )
        else:
            if self._trainer is None:
                raise ValueError('Trainer not setup - Run setup_trainer')
            else:
                raise ValueError('Test data not setup - Run set_predict')

    def __get_config(self):
        return AutoConfig.from_pretrained(
            self._config_name,
            num_labels=len(self._label_to_id.keys()),
            label2id=self._label_to_id,
            id2label=self._id_to_label,
            finetuning_task=self._task_name,
            cache_dir=self._cache_dir,
            revision=self._model_revision,
            use_auth_token=self._use_auth_token,
        )

    def __get_tokenizer(self, do_lower_case=False):
        if self._config is None:
            raise ValueError('Model config not initialized')
        if self._config.model_type in {"gpt2", "roberta"}:
            tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name,
                cache_dir=self._cache_dir,
                use_fast=True,
                do_lower_case=do_lower_case,
                revision=self._model_revision,
                use_auth_token=self._use_auth_token,
                add_prefix_space=True,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name,
                cache_dir=self._cache_dir,
                use_fast=True,
                do_lower_case=do_lower_case,
                revision=self._model_revision,
                use_auth_token=self._use_auth_token,
            )
        # Tokenizer check: this script requires a fast tokenizer.
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise ValueError(
                "This example script only works for models that have a fast tokenizer. "
                "Checkout the big table of models ""at https://huggingface.co/transformers/index.html "
                "#bigtable to find the model types that meet this requirement")
        return tokenizer

    def __get_model(self, threshold: Optional[float] = None):
        # Get the model
        post_process_picker = PostProcessPicker(label_list=self._label_list)
        model_picker = ModelPicker(
            model_name_or_path=self._model_name_or_path,
            config=self._config,
            cache_dir=self._cache_dir,
            model_revision=self._model_revision,
            use_auth_token=self._use_auth_token
        )
        if self._post_process == 'argmax':
            model = model_picker.get_argmax_bert_model()
            post_processor = post_process_picker.get_argmax()
        elif self._post_process == 'threshold_max':
            model = model_picker.get_argmax_bert_model()
            post_processor = post_process_picker.get_threshold_max(threshold=threshold)
        elif self._post_process == 'threshold_sum':
            model = model_picker.get_argmax_bert_model()
            post_processor = post_process_picker.get_threshold_sum(threshold=threshold)
        elif self._post_process == 'logits':
            model = model_picker.get_argmax_bert_model()
            post_processor = post_process_picker.get_logits()
        elif self._post_process == 'crf':
            model = model_picker.get_crf_bert_model(notation=self._notation, id_to_label=self._id_to_label)
            post_processor = post_process_picker.get_crf()
            post_processor.set_crf(model.crf)
        else:
            raise ValueError('Invalid post_process argument')
        return model, post_processor

    @staticmethod
    def __write_predictions(output_predictions_file, note_ids, note_tokens, note_labels, note_predictions):
        # Save predictions
        with open(output_predictions_file, "w") as file:
            for note_id, note_token, note_label, note_prediction in zip(
                    note_ids,
                    note_tokens,
                    note_labels,
                    note_predictions
            ):
                prediction_info = {
                    'note_id': note_id,
                    'tokens': note_token,
                    'labels': note_label,
                    'predictions': note_prediction
                }
                file.write(json.dumps(prediction_info) + '\n')

    @staticmethod
    def __get_predictions(note_ids, note_tokens, note_labels, note_predictions):
        # Return predictions
        for note_id, note_token, note_label, note_prediction in zip(
                note_ids,
                note_tokens,
                note_labels,
                note_predictions
        ):
            prediction_info = {
                'note_id': note_id,
                'tokens': note_token,
                'labels': note_label,
                'predictions': note_prediction
            }
            yield prediction_info


def main():
    parser = HfArgumentParser((
        ModelArguments,
        DataTrainingArguments,
        EvaluationArguments,
        TrainingArguments
    ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, evaluation_args, training_args = \
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, evaluation_args, training_args = \
            parser.parse_args_into_dataclasses()

    sequence_tagger = SequenceTagger(
        task_name=data_args.task_name,
        notation=data_args.notation,
        ner_types=data_args.ner_types,
        model_name_or_path=model_args.model_name_or_path,
        config_name=model_args.config_name,
        tokenizer_name=model_args.tokenizer_name,
        post_process=model_args.post_process,
        cache_dir=model_args.cache_dir,
        model_revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
        threshold=model_args.threshold,
        do_lower_case=data_args.do_lower_case,
        fp16=training_args.fp16,
        seed=training_args.seed,
        local_rank=training_args.local_rank
    )
    sequence_tagger.load()
    if training_args.do_train:
        sequence_tagger.set_train(
            train_file=data_args.train_file,
            max_train_samples=data_args.max_train_samples,
            preprocessing_num_workers=data_args.preprocessing_num_workers,
            overwrite_cache=data_args.overwrite_cache
        )
    if training_args.do_eval:
        sequence_tagger.set_eval(
            validation_file=data_args.validation_file,
            max_val_samples=data_args.max_eval_samples,
            preprocessing_num_workers=data_args.preprocessing_num_workers,
            overwrite_cache=data_args.overwrite_cache
        )
        sequence_tagger.set_eval_metrics(
            validation_spans_file=evaluation_args.validation_spans_file,
            model_eval_script=evaluation_args.model_eval_script,
            ner_types_maps=evaluation_args.ner_type_maps,
            evaluation_mode=evaluation_args.evaluation_mode
        )
    if training_args.do_predict:
        sequence_tagger.set_predict(
            test_file=data_args.test_file,
            max_test_samples=data_args.max_predict_samples,
            preprocessing_num_workers=data_args.preprocessing_num_workers,
            overwrite_cache=data_args.overwrite_cache
        )
    sequence_tagger.setup_trainer(training_args=training_args)
    if training_args.do_train:
        sequence_tagger.train(checkpoint=training_args.resume_from_checkpoint)
    if training_args.do_eval:
        metrics = sequence_tagger.evaluate()
        with open(training_args.output_dir + 'model_eval_results.json', 'w') as file:
            file.write(json.dumps(metrics) + '\n')
    if training_args.do_predict:
        sequence_tagger.predict(output_predictions_file=data_args.output_predictions_file)
        # for i in sequence_tagger.predict(output_predictions_file=None):
        #    print(i)


if __name__ == '__main__':
    main()
