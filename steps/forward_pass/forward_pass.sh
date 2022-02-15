# We go through the 4 steps that are required to de-identify a dataset (i.e run the forward pass on this dataset using a trained model

# STEP 1: INITIALIZE

# Initialize the path where the dataset is located (input_file)
## Input dataset
INPUT_FILE=../../data/notes/notes.jsonl
# Initialize the location where we will store the sentencized and tokenized dataset (ner_dataset_file)
NER_DATASET_FILE=../../data/ner_datasets/test.jsonl
# Initialize the location where we will store the model predictions (predictions_file)
# Verify this file location - Ensure it's the same location that you will pass in the json file
# to the sequence tagger model. i.e. output_predictions_file in the json file should have the same
# value as below
PREDICTIONS_FILE=../../data/predictions/predictions.jsonl
# Initialize the file that will contain the original note text and the de-identified note text
DEID_FILE=../../data/notes/deid.jsonl
# Initialize the model config. This config file contains the various parameters of the model.
MODEL_CONFIG=./run/i2b2/predict_i2b2.json

# STEP 2: NER DATASET
## Sentencize and tokenize the raw text. We used sentences of length 128, which includes an additional 32 context tokens on either side of the sentence. These 32 tokens serve (from the previous & next sentence) serve as additional context to the current sentence.
## We use the en_core_sci_sm sentencizer and a custom tokenizer (can be found in the preprocessing module)
## The dataset stored in the ner_dataset_file will be used as input to the sequence tagger model

create_data --input_file $INPUT_FILE --output_file $NER_DATASET_FILE --sentencizer en_core_sci_sm --tokenizer clinical --notation BILOU  --mode predict --max_tokens 128 --max_prev_sentence_token 32 --max_next_sentence_token 32 --default_chunk_size 32 --ignore_label NA --token_text_key text --metadata_key meta --note_id_key note_id --label_key label --span_text_key spans

# STEP 3: SEQUENCE TAGGING
## Run the sequence model - specify parameters to the sequence model in the config file (model_config). The model will be run with the specified parameters. For more information of these parameters, please refer to huggingface (or use the docs provided).
## This file uses the argmax output. To use the recall threshold models (running the forward pass with a recall biased threshold for aggressively removing PHI) use the other config files.
## The config files in the i2b2 directory specify the model trained on only the i2b2 dataset. The config files in the mgb_i2b2 directory is for the model trained on both MGB and I2B2 datasets.
## You can manually pass in the parameters instead of using the config file. The config file option is recommended. In our example we are passing the parameters through a config file.

tag_sequence $MODEL_CONFIG

# STEP 4: DE-IDENTIFY TEXT
## This step uses the predictions from the previous step to de-id the text. We pass the original input file where the original text is present. We look at this text and the predictions and use both of these to de-id the text.
## De-identify the text - using deid_strategy=replace_informative doesn't drop the PHI from the text, but instead labels the PHI - which you can use to drop the PHI or do any other processing.
## If you want to drop the PHI automatically, you can use deid_strategy=remove
deid_text --input_file $INPUT_FILE --predictions_file $PREDICTIONS_FILE --notation BILOU --deid_strategy replace_informative --output_file $DEID_FILE --span_constraint super_strict
