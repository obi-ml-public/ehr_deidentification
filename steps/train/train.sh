# STEPS

# We go through the 6 steps that are required to train a model using the transformer architectures from huggingface.

## STEP 1: INITIALIZE
# Initialize the path where the dataset is located (input_file).
INPUT_FILE=/home/pk621/projects/data/ehr_deidentification/i2b2/i2b2.jsonl
# Initialize the location where we will store the train data
TRAIN_FILE_RAW=/home/pk621/projects/data/ehr_deidentification/i2b2/train_unfixed.jsonl
# Initialize the location where we will store the validation data
VALIDATION_FILE_RAW=/home/pk621/projects/data/ehr_deidentification/i2b2/validation_unfixed.jsonl
# Initialize the location where we will store the train data after fixing the spans
TRAIN_FILE=/home/pk621/projects/data/ehr_deidentification/i2b2/train.jsonl
# Initialize the location where we will store the validation data after fixing the spans
VALIDATION_FILE=/home/pk621/projects/data/ehr_deidentification/i2b2/validation.jsonl
# Initialize the location where the spans for hte validation data are stored
VALIDATION_SPANS_FILE=/home/pk621/projects/data/ehr_deidentification/i2b2/validation_spans.jsonl
# Initialize the location where we will store the sentencized and tokenized train dataset (train_file)
NER_TRAIN_FILE=/home/pk621/projects/data/ehr_deidentification/ner_datasets/i2b2_train/train.jsonl
# Initialize the location where we will store the sentencized and tokenized validation dataset (validation_file)
NER_VALIDATION_FILE=/home/pk621/projects/data/ehr_deidentification/ner_datasets/i2b2_train/validation.jsonl
# Initialize the model config. This config file contains the various parameters of the model.
MODEL_CONFIG=./run/i2b2/train_i2b2.json
# Initialize the sentencizer and tokenizer
SENTENCIZER=en_core_sci_sm
TOKENIZER=clinical

# STEP 2: DATASET SPLITS
## Read the input dataset and split it into train and validation splits

split_data --input_file $INPUT_FILE --train_proportion 87 --validation_proportion 13 --test_proportion 0 --train_file $TRAIN_FILE_RAW --validation_file $VALIDATION_FILE_RAW --print_dist --margin 0.3 --spans_key spans --metadata_key meta --group_key note_id


# STEP 3: VALIDATION SPANS
## We write out the validation spans and also setup the token level validation dataset below
## We do this because we may have different tokenizers and to make a fair comparison, we compare models at the span level. To do this we need the span information at a character level (start of span & end of span in terms of character position). If we have information at a character level, it is easier to compare different tokenizers. Now we not only have token level performance, but also span level performance
## We use this step to write out the annotated spans to do span level evaluation. 
## One of the reason we did this is because in step 4 we modify the original annotated spans so that we can create a NER dataset. To ensure that we still evaluate on the original annotated dataset we do this step. Read step 4 to understand further why we need do this to ensure a fair comparison during evaluation.
## This step has the validation data (or any dataset that you want to test on) in the original form - spans with the specified start and end position.
## In summary, for doing span level evaluation we need a mapping between note_id and the annotated spans for that note_id 

validation_spans --input_file $VALIDATION_FILE_RAW --output_file $VALIDATION_SPANS_FILE --metadata_key meta --note_id_key note_id --spans_key spans

# STEP 4: FIX SPANS

# This step is optional and may not be required
# This code may be required if you have spans that don't line up with your tokenizer (e.g dataset was annoated at a character level and yout tokenizer doesn't split at the same position). This code fixes the spans so that the code below (creating NER datasets) runs wothout error.
# We experienced the issue above in the step where we create the NER dataset (step 5) - where we need to align the labels with the tokens based on the BILOU/BIO.. notation. Without this step, we would run into alignment issues.
# If you face the same issue, running this step should fix it - changes the label start and end positions of the annotated spans based on your tokenizer and saves the new spans.
# Sometimes there may be some label (span) overlap - the priority list assigns a priority to each label.
# Higher preference is given to labels with higher priority when resolving label overlap

fix_spans --input_file $TRAIN_FILE_RAW --output_file $TRAIN_FILE --sentencizer $SENTENCIZER --tokenizer $TOKENIZER --ner_types PATIENT STAFF AGE DATE PHONE ID EMAIL PATORG LOC HOSP OTHERPHI --ner_priorities 2 1 2 2 2 2 2 1 2 1 1 --text_key text --spans_key spans

fix_spans --input_file $VALIDATION_FILE_RAW --output_file $VALIDATION_FILE --sentencizer $SENTENCIZER --tokenizer $TOKENIZER --ner_types PATIENT STAFF AGE DATE PHONE ID EMAIL PATORG LOC HOSP OTHERPHI --ner_priorities 2 1 2 2 2 2 2 1 2 1 1 --text_key text --spans_key spans

# STEP 5: NER DATASET
## Sentencize and tokenize the raw text. We used sentences of length 128, which includes an additional 32 context tokens on either side of the sentence. These 32 tokens serve (from the previous & next sentence) serve as additional context to the current sentence.
## We used the en_core_sci_sm sentencizer and a custom tokenizer (can be found in the preprocessing module)
## The dataset stored in the ner_dataset_file will be used as input to the sequence tagger model

create_data --input_file $TRAIN_FILE --output_file $NER_TRAIN_FILE --sentencizer $SENTENCIZER --tokenizer $TOKENIZER --notation BILOU  --mode train --max_tokens 128 --max_prev_sentence_token 32 --max_next_sentence_token 32 --default_chunk_size 32 --ignore_label NA --token_text_key text --metadata_key meta --note_id_key note_id --label_key label --span_text_key spans

create_data --input_file $VALIDATION_FILE --output_file $NER_VALIDATION_FILE --sentencizer $SENTENCIZER --tokenizer $TOKENIZER --notation BILOU  --mode train --max_tokens 128 --max_prev_sentence_token 32 --max_next_sentence_token 32 --default_chunk_size 32 --ignore_label NA --token_text_key text --metadata_key meta --note_id_key note_id --label_key label --span_text_key spans

# STEP 6: SEQUENCE TAGGING
## Train the sequence model - specify parameters to the sequence model in the config file (model_config). The model will be trained with the specified parameters. For more information of these parameters, please refer to huggingface (or use the docs provided).
## You can manually pass in the parameters instead of using the config file. The config file option is recommended. In our example we are passing the parameters through a config file. If you do not want to use the config file, skip the next code block and manually enter the values in the following code blocks. You will still need to read in the training args using huggingface and change values in the training args according to your needs.

tag_sequence $MODEL_CONFIG