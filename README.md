# Survey of POS Taggers Approaches on Shahmukhi Punjabi Corpus


This repository contains the research and analysis conducted in the project "Survey of POS Taggers Performance on Shahmukhi Punjabi Corpus". Our goal is to evaluate the performance of various Part-of-Speech (POS) taggers specifically designed for the Shahmukhi script of the Punjabi language.

***Introduction***
Part-of-Speech tagging is a critical process in the pipeline of natural language processing (NLP) tasks. The performance of POS taggers on less-resourced languages like Punjabi in the Shahmukhi script is not well-documented, which motivates this survey.

***Dataset***
The data is a Punjabi Shahmukhi annotated corpus publically available at :

https://github.com/toqeerehsan/Shahmukhi-POS-Tagging


.
├── examples/                   # Contains train/dev/test .conllu files
│   ├── pa-ud-train.conllu
│   ├── pa-ud-dev.conllu
│   ├── pa-ud-test.conllu
│   └── pa-combined-data.conllu (from combine_data.py)
│
├── folds/                     # 5-fold cross-validation splits
│   ├── pa-fold-1.conllu
│   └── ...
│
├── logs/                      # Model files from cross-validation or convergence
│   ├── model.dat (Perceptron)
│   ├── pos_tagger_model.pkl (CRF)
│   └── MaChAmp saved checkpoints
│
├── outputs/                   # Evaluation results & plots
│   ├── prediction_output
│   ├── score_perceptron.txt
│   ├── score_crf.txt
│   ├── score_machamp.txt
│   ├── crf_convergence_plot_TERvsIterations.png
│   ├── perceptron_convergence_plot_TERvsCorpusSize.png
│   └── ...
│
├── configs/                   # MaChAmp config files
│   ├── pa.upos.json
│   └── params.json
│
├── cross_validation.py
├── model_convergence_TERvsIterations.py
├── model_convergence_TERvsCorpusSize.py
├── tagger.py (Perceptron)
├── crf_pos_tagger.py
├── simple_eval.py
└── combine_data.py


# Models

This data is preprocessed to merge tokens with tags and then UPOS are added for each corresponding language-specific tags. The sentences are then put in a .conllu format. The following taggers are implemented on the data. Perceptron: https://github.com/ftyers/conllu-perceptron-tagger CRF : https://sklearn-crfsuite.readthedocs.io/en/latest/ MACHamp: https://github.com/machamp-nlp/machamp

***How to run the taggers ***
# Perceptron
``bash
***Train*** 
cat examples/pa-ud-train.conllu | python3 tagger.py -t model.dat

***Predict***
cat examples/pa-ud-test.conllu | python3 tagger.py model.dat > outputs/prediction_output

***Evaluate***
python3 simple_eval.py examples/pa-ud-test.conllu outputs/prediction_output > outputs/score_perceptron.txt

# CRF

***Train & Predict*** 
python3 crf_pos_tagger.py  # Outputs: pos_tagger_model.pkl, crf_pos_predictions.conllu

***Evaluate***
python3 simple_eval.py examples/pa-ud-test.conllu outputs/crf_pos_predictions.conllu > outputs/score_crf.txt

# Machamp

***Setup*** 

!git clone https://github.com/machamp-nlp/machamp.git
!pip3 install --user -r requirements.txt
!pip install jsonnet

***Train***
!python3 train.py --dataset_configs configs/pa.upos.json

****Predict***
!python3 predict.py logs/pa.upos/<TIMESTAMP>/model_19.pt examples/pa-ud-test.conllu outputs/machamp-pos-predictions.output

***Evaluate***
python3 simple_eval.py examples/pa-ud-test.conllu outputs/machamp-pos-predictions.output > outputs/score_machamp.txt


# Data Files

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

## Cross-Validation (5 Folds)

Step 1: Run (in main directory) combine_data.py to generate:

examples/pa-combined-data.conllu

folds/pa-fold-1.conllu to pa-fold-5.conllu

Step 2: Run cross-validation with increasing train sizes (20%, 40%, 60%, 80%):


cross_validation.py

Outputs: (for each model)

outputs/cross_validation_results.csv

outputs/ter_vs_train_size.png

---

## Convergence Analysis

A. TER vs Iterations (10, 20, ..., N) - Fixed Corpus Size 

Run: CRF/model_convergence_TERvsIterations.py
Outputs:
CRF: crf_convergence_metrics_TERvsIterations.csv	& crf_convergence_plot_TERvsIterations.png
Perceptron: 	perceptron_convergence_metrics_TERvsIterations.csv &	perceptron_convergence_plot_TERvsIterations.png



B. TER vs Corpus Size (10%, 20%, 30%) — Fixed 100 iterations

Run: perceptron/model_convergence_TERvsCorpusSize.py
Outputs:
CRF: 	crf_convergence_metrics_TERvsCorpusSize.csv	& crf_convergence_plot_TERvsCorpusSize.png
Perceptron: 	perceptron_convergence_TERvsCorpusSize.csv	& perceptron_convergence_plot_TERvsCorpusSize.png

-----


