# Survey of POS Taggers Approaches on Shahmukhi Punjabi Corpus


This repository contains the research and analysis conducted in the project "Survey of POS Taggers Performance on Shahmukhi Punjabi Corpus". Our goal is to evaluate the performance of various Part-of-Speech (POS) taggers specifically designed for the Shahmukhi script of the Punjabi language.

***Introduction***
Part-of-Speech tagging is a critical process in the pipeline of natural language processing (NLP) tasks. The performance of POS taggers on less-resourced languages like Punjabi in the Shahmukhi script is not well-documented, which motivates this survey.

***Dataset***
The data is a Punjabi Shahmukhi annotated corpus publically available at :

https://github.com/toqeerehsan/Shahmukhi-POS-Tagging


.<br>
├── examples/                   # Contains train/dev/test .conllu files<br>
│   ├── pa-ud-train.conllu<br>
│   ├── pa-ud-dev.conllu<br>
│   ├── pa-ud-test.conllu<br>
│   └── pa-combined-data.conllu (from combine_data.py)<br>
│<br>
├── folds/                     # 5-fold cross-validation splits<br>
│   ├── pa-fold-1.conllu<br>
│   └── ...<br>
│<br>
├── logs/                      # Model files from cross-validation or convergence<br>
│   ├── model.dat (Perceptron)<br>
│   ├── pos_tagger_model.pkl (CRF)<br>
│   └── MaChAmp saved checkpoints<br>
│<br>
├── outputs/                   # Evaluation results & plots<br>
│   ├── prediction_output<br>
│   ├── score_perceptron.txt<br>
│   ├── score_crf.txt<br>
│   ├── score_machamp.txt<br>
│   ├── crf_convergence_plot_TERvsIterations.png<br>
│   ├── perceptron_convergence_plot_TERvsCorpusSize.png<br>
│   └── ...<br>
│
├── configs/                   # MaChAmp config files<br>
│   ├── pa.upos.json<br>
│   └── params.json<br>
│
├── cross_validation.py<br>
├── model_convergence_TERvsIterations.py<br>
├── model_convergence_TERvsCorpusSize.py<br>
├── tagger.py (Perceptron)<br>
├── crf_pos_tagger.py<br>
├── simple_eval.py<br>
└── combine_data.py<br>


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

A. TER vs Iterations (10, 20, ..., N) - Fixed Corpus Size <br>

Run: CRF/model_convergence_TERvsIterations.py<br>
Outputs:<br>
CRF: crf_convergence_metrics_TERvsIterations.csv	**&** crf_convergence_plot_TERvsIterations.png<br>
Perceptron: 	perceptron_convergence_metrics_TERvsIterations.csv **&**	perceptron_convergence_plot_TERvsIterations.png<br>



B. TER vs Corpus Size (10%, 20%, 30%) — Fixed 100 iterations<br>

Run: perceptron/model_convergence_TERvsCorpusSize.py<br>
Outputs:<br>
CRF: 	crf_convergence_metrics_TERvsCorpusSize.csv	**&** crf_convergence_plot_TERvsCorpusSize.png<br>
Perceptron: 	perceptron_convergence_TERvsCorpusSize.csv	**&** perceptron_convergence_plot_TERvsCorpusSize.png<br>

-----


