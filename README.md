# Robust DeID: De-Identification of Medical Notes using Transformer Architectures

[![DOI](https://zenodo.org/badge/458346577.svg)](https://zenodo.org/badge/latestdoi/458346577)



This repository was used to train and evaluate various de-identification models and strategies on medical notes from the I2B2-DEID dataset and the MassGeneralBrigham (MGB) network.
The models and strategies are extensible and can be used on other datasets as well. Trained models are published on huggingface under the [OBI organization](https://huggingface.co/obi).

Main features are:

1. Transformer models - Any transformer model from the [huggingface](https://huggingface.co) library can be used for training. We make available a RoBERTa [Liu et al., 2019](https://arxiv.org/pdf/1907.11692.pdf) model and a ClinicalBERT [Alsentzer et al., 2019](https://arxiv.org/pdf/1904.03323.pdf) model fine-tuned for de-identification on huggingface: [obi_roberta_deid](https://huggingface.co/obi/deid_roberta_i2b2), [obi_bert_deid](https://huggingface.co/obi/deid_bert_i2b2). Both can be used for testing (forward pass).
2. Recall biased thresholding - Ability to use classification bias to aggressively remove PHI from notes. This is a safer and more robust option when working with sensitive data like medical notes.
3. Custom clinical tokenizer - Includes 60 regular expressions based on the structure and information generally found in medical notes. This tokenizer resolves common typographical errors and missing spaces that occur in clinical notes.
4. Context enhancement - Option to add on additional tokens to a given sequence as context on the left and right. These tokens can be used only as context, or we can also train on these tokens (which essentially mimics a sliding window approach). The reason for including context tokens was to provide additional context especially for peripheral tokens in a given sequence.

Since de-identification is a sequence labeling task, this tool can be applied to any other sequence labeling task as well.\
More details on how to use this tool, the format of data and other useful information is presented below.

Comments, feedback and improvements are welcome and encouraged!

## Dataset Annotations

* The guidelines for the dataset annotation and prodigy setup can be found here: 
[Annotation guidelines](./AnnotationGuidelines.md)

## Installation

### Dependencies

* You can either install the dependencies using conda or pip. Both are specified below.
* We developed this package using the conda environment specified in [deid.yml](./deid.yml). You can create the environment using this file, and it will install the required dependencies.
* Robust De-ID requires the packages specified in the [requirements.txt](./requirements.txt) file. You can use pip install to install these packages.
* We used the conda approach and activated the **deid** conda environment for building and testing this package

```shell
conda env create -f deid.yml
conda activate deid
```

### Robust De-Id

* To install robust-deid, first install the dependencies (as mentioned above) and then do a pip install of robust de-id package.
```shell
pip install robust-deid
```

## Data Format

* The format of the data differs slightly when training a model to running the forward pass.
* The data is in the json format, where we store the notes in a jsonl file. Each line in this file is a json object that refers to one note.
* A jsonl file will have multiple lines, where each line represents a json object that corresponds to a unique note.  
* More details on what is present in each line (json object) is presented below.
```json lines
{"...": "..."}
{"...": "..."}
{"...": "..."}
```

### Training

* The notes should be in json format. The "key" values that we mention below are the ones that we used, you are free to change the keys in the json file (make sure that these changes are reflected in the subsequent steps - train/test/evaluate).
* We show an example for a single note, for multiple notes we would add multiple of the json objects shown below in a single jsonl file.
* The default values in the package assume that the text is present in the "text" field. 
* There should be a "meta" field that should contain a unique "note_id" field. Every note should have a unique "note_id". Other metadata fields may be added if required for your needs.
* The "spans" field should contain the annotated spans for the training dataset. They should be in sorted order (based on start position)
* The "spans" field will contain a list of spans, where each span should contain the "start", "end" and "label" of the span
```json
{ "text": "Physician Discharge Summary Admit date: 10/12/1982 Discharge date: 10/22/1982 Patient Information Jack Reacher, 54 y.o. male (DOB = 1/21/1928) ...", 
  "meta": {"note_id": "1", "patient_id": "1"},
  "spans": [{"id":"0", "start": 40, "end": 50, "label": "DATE"}, {"id":"1", "start": 67, "end": 77, "label": "DATE"}, {"id":"3", "start": 98, "end": 110, "label": "PATIENT"}, {"id":"3", "start": 112, "end": 114, "label": "AGE"}, {"...": "..."}]}
```

### Test (Forward Pass/Inference)

* The format is almost the same as above. Since, at test time we don't have annotated spans, we assign an empty list to the "spans" field
* We show an example for a single note, for multiple notes we would add multiple of the json objects shown below in a single jsonl file.
```json
{ "text": "Physician Discharge Summary Admit date: 10/12/1982 Discharge date: 10/22/1982 Patient Information Jack Reacher, 54 y.o. male (DOB = 1/21/1928) ...", 
  "meta": {"note_id": "1", "patient_id": "1"},
  "spans": []}
```

## Usage
* Once you have the package installed and the data ready, follow the steps described below.
* Feel free to replace the models in the demos with any of the ones you have trained or any model from [huggingface](https://huggingface.co).

### Test (Forward Pass/Inference)
* We have demos for running the forward pass in the following folder: [steps/forward_pass](./steps/forward_pass). You can add or modify any of the values mentioned in the notebook or shell scripts based on your needs (e.g. sentencizers, tokenizers, model parameters in the config file etc.).
* The forward pass can be run via JupyterNotebook (can also be used via a python script) or a shell script.
* To use a trained model to run the forward pass on the desired dataset, using a JupyterNotebook, follow the steps shown in the [ForwardPass.ipynb](./steps/forward_pass/Forward%20Pass.ipynb) notebook.
* To use a trained model to run the forward pass on the desired dataset, using a shell script, follow the steps shown in the [forward_pass.sh](./steps/forward_pass/forward_pass.sh) script.
* We also include the step of using the model predictions to de-identify the medical notes in the notebook/script (i.e. producing the de-identified version of the original dataset/text).

### Training
* We have demos for running the forward pass in the following folder: [steps/train](./steps/train). You can add or modify any of the values mentioned in the notebook or shell scripts based on your needs (e.g. sentencizers, tokenizers, model parameters in the config file etc.).
* Training a model can be done via JupyterNotebook (can also be used via a python script) or a shell script.
* To use a trained model to run the forward pass on the desired dataset, using a JupyterNotebook, follow the steps shown in the [Train.ipynb](./steps/train/Train.ipynb) notebook.
* To use a trained model to run the forward pass on the desired dataset, using a shell script, follow the steps shown in the [train.sh](./steps/train/train.sh) script.
* We used the i2b2 2014 dataset while creating the demo (you can use any dataset of your choice). To download the i2b2 2014 dataset please visit: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

### Evaluation
* To evaluate a trained model on a dataset, refer to the demos present in the following folder: [steps/evaluation](./steps/evaluation)
* Evaluating a model can be done via JupyterNotebook (can also be used via a python script) or a shell script.
* To use a trained model and evaluate its performance on the desired dataset, using a JupyterNotebook, follow the steps shown in the [Evaluation.ipynb](./steps/evaluation/Evaluation.ipynb) notebook.
* To use a trained model and evaluate its performance on the desired dataset, using a shell script, follow the steps shown in the [evaluation.sh](./steps/evaluation/evaluation.sh) script.
* We do both token (suffixed with "_token") and span level evaluation for each entity and overall. 
* There's also an option to do binary evaluation - which can be specified via the ner_type_maps argument in the config file. These map existing PHI labels to a new set of PHI labels on which we do the evaluation

### Recall biased thresholding
* The objective is to modify the classification thresholds, i.e. instead of choosing the class with the highest probability as the prediction for a token (optimize F1), we modify the classification thresholds to optimize recall.
* While this may decrease precision, having high levels of recall is essential for sensitive datasets.
* The demos in the following folder: [steps/recall_threshold](./steps/recall_threshold) demonstrate how we can take our trained models and estimate classification thresholds to optimize recall
* To use a trained model, optimize it for a desired level of recall (based on validation data) and evaluate its performance on the test dataset, using a JupyterNotebook, follow the steps shown in the [RecallThreshold.ipynb](./steps/recall_threshold/RecallThreshold.ipynb) notebook.


## Trained Models
* Our models for de-identification of medical notes can be found in: [OBI organization](https://huggingface.co/obi).
* Models:
    * [OBI-ClinicalBERT De-Identification Model](https://huggingface.co/obi/deid_bert_i2b2)
    * [OBI-RoBERTa De-Identification Model](https://huggingface.co/obi/deid_roberta_i2b2)
* Demo:
    * [Medical Note De-Identification](https://huggingface.co/spaces/obi/Medical-Note-Deidentification)
  

