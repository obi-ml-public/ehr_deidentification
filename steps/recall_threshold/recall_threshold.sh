# STEPS

# We go through the 5 steps that are required to train a model using the transformer architectures from huggingface.

## STEP 1: INITIALIZE
# Initialize the location where we will store the validation data
VALIDATION_FILE_RAW=/home/pk621/projects/data/ehr_deidentification/i2b2/validation_unfixed.jsonl
# Initialize the location where we will store the validation data after fixing the spans
VALIDATION_FILE=/home/pk621/projects/data/ehr_deidentification/i2b2/validation.jsonl
# Initialize the location where the spans for hte validation data are stored
VALIDATION_SPANS_FILE=/home/pk621/projects/data/ehr_deidentification/i2b2/validation_spans.jsonl
# Initialize the location where we will store the sentencized and tokenized validation dataset (validation_file)
NER_VALIDATION_FILE=/home/pk621/projects/data/ehr_deidentification/ner_datasets/i2b2_train/validation.jsonl
# Initialize the location where we will store the model logits (predictions_file)
# Verify this file location - Ensure it's the same location that you will pass in the json file
# to the sequence tagger model. i.e. output_predictions_file in the json file should have the same
# value as below
LOGITS_FILE=/home/pk621/projects/data/ehr_deidentification/model_predictions/i2b2_train/logits.jsonl
# Initialize the model config. This config file contains the various parameters of the model.
MODEL_CONFIG=./run/i2b2/logits_i2b2.json
# Initialize the sentencizer and tokenizer
SENTENCIZER=en_core_sci_sm
TOKENIZER=clinical

# STEP 2: FIX SPANS

# This step is optional and may not be required
# This code may be required if you have spans that don't line up with your tokenizer (e.g dataset was annoated at a character level and yout tokenizer doesn't split at the same position). This code fixes the spans so that the code below (creating NER datasets) runs wothout error.
# We experienced the issue above in the step where we create the NER dataset (step 5) - where we need to align the labels with the tokens based on the BILOU/BIO.. notation. Without this step, we would run into alignment issues.
# If you face the same issue, running this step should fix it - changes the label start and end positions of the annotated spans based on your tokenizer and saves the new spans.
# Sometimes there may be some label (span) overlap - the priority list assigns a priority to each label.
# Higher preference is given to labels with higher priority when resolving label overlap

fix_spans --input_file $VALIDATION_FILE_RAW --output_file $VALIDATION_FILE --sentencizer $SENTENCIZER --tokenizer $TOKENIZER --ner_types PATIENT STAFF AGE DATE PHONE ID EMAIL PATORG LOC HOSP OTHERPHI --ner_priorities 2 1 2 2 2 2 2 1 2 1 1 --text_key text --spans_key spans

# STEP 3: NER DATASET
## Sentencize and tokenize the raw text. We used sentences of length 128, which includes an additional 32 context tokens on either side of the sentence. These 32 tokens serve (from the previous & next sentence) serve as additional context to the current sentence.
## We used the en_core_sci_sm sentencizer and a custom tokenizer (can be found in the preprocessing module)
## The dataset stored in the ner_dataset_file will be used as input to the sequence tagger model

create_data --input_file $VALIDATION_FILE --output_file $NER_VALIDATION_FILE --sentencizer $SENTENCIZER --tokenizer $TOKENIZER --notation BILOU  --mode train --max_tokens 128 --max_prev_sentence_token 32 --max_next_sentence_token 32 --default_chunk_size 32 --ignore_label NA --token_text_key text --metadata_key meta --note_id_key note_id --label_key label --span_text_key spans

# STEP 4: SEQUENCE TAGGING
## Train the sequence model - specify parameters to the sequence model in the config file (model_config). The model will be trained with the specified parameters. For more information of these parameters, please refer to huggingface (or use the docs provided).
## You can manually pass in the parameters instead of using the config file. The config file option is recommended. In our example we are passing the parameters through a config file. If you do not want to use the config file, skip the next code block and manually enter the values in the following code blocks. You will still need to read in the training args using huggingface and change values in the training args according to your needs.

tag_sequence $MODEL_CONFIG

# STEP 5: RECALL THRESHOLDING
## The objective is to modify the classification thresholds, i.e. instead of choosing the class with the highest probability as the prediction for a token (optimize F1), we modify the classification thresholds to optimize recall.
## The code below is to get these thresholds such that we get the desired level of recall. We use a validation dataset to optimize the threshold and level of recall.
## We get the thresholds by re-formulating the NER task as a binary classifiation task. PHI v/s non-PHI. We have two two methods to do this: MAX and SUM.
## MAX: 
    ### probability of PHI class = maximum SoftMax probability over all the PHI classes
    ### probability of non-PHI class
## SUM: 
    ### probability of PHI class = sum of SoftMax probabilities over all the PHI classes
    ### probability of non-PHI class
# A brief explantion of how we use these thresholds is explained below

# Threshold mode - max
# This means that an input token is tagged with the non-PHI class only if the 
# maximum probability over all PHI classes was less than the chosen threshold.
# We tag the token with the PHI class that has the highest probability

threshold_recall --logits_file $LOGITS_FILE --ner_types PATIENT STAFF AGE DATE PHONE ID EMAIL PATORG LOC HOSP OTHERPHI --notation BILOU --threshold_mode max --recall_cutoff 99.9

# Threshold mode - sum
# This means that an input token is tagged with the PHI class only if the sum of 
# probabilities over all PHI classes is greater than the chosen threshold.
# We tag the token with the PHI class that has the highest probability

threshold_recall --logits_file $LOGITS_FILE --ner_types PATIENT STAFF AGE DATE PHONE ID EMAIL PATORG LOC HOSP OTHERPHI --notation BILOU --threshold_mode sum --recall_cutoff 99.9