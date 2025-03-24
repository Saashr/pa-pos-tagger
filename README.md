***Survey of POS Taggers Performance on Shahmukhi Punjabi Corpus***


This repository contains the research and analysis conducted in the project "Survey of POS Taggers Performance on Shahmukhi Punjabi Corpus". Our goal is to evaluate the performance of various Part-of-Speech (POS) taggers specifically designed for the Shahmukhi script of the Punjabi language.

***Introduction***
Part-of-Speech tagging is a critical process in the pipeline of natural language processing (NLP) tasks. The performance of POS taggers on less-resourced languages like Punjabi in the Shahmukhi script is not well-documented, which motivates this survey.

***Dataset***
The data is a Punjabi Shahmukhi annotated corpus publically available at :

https://github.com/toqeerehsan/Shahmukhi-POS-Tagging

*** Models***

This data is preprocessed to merge tokens with tags and then UPOS are added for each corresponding language specific tags. The sentences are then put in a .conllu format. The following taggers are implemented on the data. Perceptron: https://github.com/ftyers/conllu-perceptron-tagger CRF : https://sklearn-crfsuite.readthedocs.io/en/latest/ MACHamp: https://github.com/machamp-nlp/machamp

***How to run the taggers ***
*** Perceptron***
``bash
# Train
cat examples/pa-ud-train.conllu | python3 tagger.py -t model.dat

# Predict
cat examples/pa-ud-test.conllu | python3 tagger.py model.dat > outputs/prediction_output

# Evaluate
python3 simple_eval.py examples/pa-ud-test.conllu outputs/prediction_output > outputs/score_perceptron.txt

***Machamp***

# Train & Predict
python3 crf_pos_tagger.py  # Outputs: pos_tagger_model.pkl, crf_pos_predictions.conllu

# Evaluate
python3 simple_eval.py examples/pa-ud-test.conllu outputs/crf_pos_predictions.conllu > outputs/score_crf.txt

***CRF***

# Setup 

!git clone https://github.com/machamp-nlp/machamp.git
!pip3 install --user -r requirements.txt
!pip install jsonnet

# Train
!python3 train.py --dataset_configs configs/pa.upos.json

# Predict
!python3 predict.py logs/pa.upos/<TIMESTAMP>/model_19.pt examples/pa-ud-test.conllu outputs/machamp-pos-predictions.output

# Evaluate
python3 simple_eval.py examples/pa-ud-test.conllu outputs/machamp-pos-predictions.output > outputs/score_machamp.txt


## Data Files

Located in the `examples/` folder for each tagger:

- `pa-ud-train.conllu` — Training data  
- `pa-ud-dev.conllu` — Development/validation data  
- `pa-ud-test.conllu` — Test data  
- `pa-combined-data.conllu` — Combined version used for cross-validation and convergence experiments  

---

## Outputs

Each tagger produces the following outputs:

- **Trained Model**  
  - Perceptron: `model.dat`  
  - CRF: `pos_tagger_model.pkl`  
  - MaChAmp: Saved in `logs/` directory  

- **Predictions Output**  
  - Tagger-generated outputs in `.conllu` or `.output` format  

- **Evaluation Score File**  
  - `.txt` file containing:
    - Accuracy  
    - Precision  
    - Recall  
    - F1-score  
    - Confusion matrix  

---

## Folds (Cross-Validation)

Generated using `combine_data.py`, stored in the `folds/` directory:

- `pa-fold-1.conllu`  
- `pa-fold-2.conllu`  
- `pa-fold-3.conllu`  
- `pa-fold-4.conllu`  
- `pa-fold-5.conllu`  


 
 


