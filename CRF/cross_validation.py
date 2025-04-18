import os
import pandas as pd
import numpy as np
import joblib
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import classification_report, accuracy_score


# Function to clean a .conllu file while maintaining sentence ID blocks
def clean_conllu_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        sentence_block = []
        for line in infile:
            if line.strip() == "":
                if sentence_block:
                    outfile.write("\n".join(sentence_block) + "\n\n")
                    sentence_block = []
            elif line.startswith("#") or len(line.split('\t')) == 10:
                sentence_block.append(line.strip())
        if sentence_block:
            outfile.write("\n".join(sentence_block) + "\n")


# Function to load data from a CoNLL-U file
def load_conllu(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        current_sentence = []
        for line in f:
            line = line.strip()
            if line == '':
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            elif not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) == 10:
                    token = parts[1]
                    pos_tag = parts[3]
                    current_sentence.append((token, pos_tag))
    return sentences


# Feature engineering function
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word[:1]': word[0],
        'word[-1:]': word[-1],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.istitle()': word1.istitle(),
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.istitle()': word1.istitle(),
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]


# Paths and directories
base_dir = ""
output_dir = os.path.join(base_dir, "outputs")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get fold paths
subset_paths = [f"folds/initial/pa-fold-{i + 1}.conllu" for i in range(5)]
performance_metrics = []

# Cross-validation with incremental training
for test_fold in range(5):
    print(f"\n==== Starting Cross-Validation for Test Fold {test_fold + 1} ====")

    for train_size in range(1, 5):  # Train with 1 fold (20%) up to 4 folds (80%)
        training_files = subset_paths[test_fold + 1:test_fold + 1 + train_size]

        # If training files exceed available folds, wrap around
        if len(training_files) < train_size:
            remaining_folds = train_size - len(training_files)
            training_files += subset_paths[:remaining_folds]

        print(f"Test Fold: {test_fold + 1}, Training Folds: {[f.split('-')[-1] for f in training_files]}")

        # Paths for combined training and validation files
        validation_file = subset_paths[test_fold]
        combined_training_file = os.path.join(base_dir, f"folds/pa-ud-train-fold-{test_fold + 1}-{train_size}.conllu")

        # Combine selected training folds into a single file
        with open(combined_training_file, 'w', encoding='utf-8') as outfile:
            for file_path in training_files:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())

        # Clean the validation and training files
        cleaned_validation_file = os.path.join(base_dir, f"folds/pa-ud-valid-fold-{test_fold + 1}.conllu")
        cleaned_training_file = os.path.join(base_dir, f"folds/pa-ud-train-fold-{test_fold + 1}-{train_size}-cleaned.conllu")

        clean_conllu_file(validation_file, cleaned_validation_file)
        clean_conllu_file(combined_training_file, cleaned_training_file)

        # Load training and validation data
        train_data = load_conllu(cleaned_training_file)
        validation_data = load_conllu(cleaned_validation_file)

        X_train = [sent2features(sent) for sent in train_data]
        y_train = [sent2labels(sent) for sent in train_data]
        X_val = [sent2features(sent) for sent in validation_data]
        y_val = [sent2labels(sent) for sent in validation_data]

        # Train CRF model
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(X_train, y_train)

        # Save the trained model
        model_path = os.path.join(base_dir, f"logs/pos_tagger_model_fold_{test_fold + 1}_{train_size}.pkl")
        joblib.dump(crf, model_path)

        # Predict on validation data
        y_pred = crf.predict(X_val)

        # Evaluate performance
        # Evaluate performance
        true_tags = [label for sent in y_val for label in sent]
        pred_tags = [label for sent in y_pred for label in sent]

        report = classification_report(true_tags, pred_tags, output_dict=True, zero_division=0)

        accuracy = report['accuracy']
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']

        print(f"Fold {test_fold + 1}, Train Size {train_size * 20}%, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, TER: {1-accuracy:.4f}")
        performance_metrics.append((test_fold + 1, train_size * 20, accuracy, precision, recall, f1))

metrics_df = pd.DataFrame(performance_metrics, columns=["Test_Fold", "Train_Percentage", "Accuracy", "Precision", "Recall", "F1"])
# TER = tag error rate = 1 - accuracy
metrics_df["TER"] = 1 - metrics_df["Accuracy"]
metrics_df.to_csv(os.path.join(output_dir, "cross_validation_results.csv"), index=False)
print("Cross-validation completed. Results saved.")

# === TER PLOTTING ===
import matplotlib.pyplot as plt

ter_grouped = metrics_df.groupby("Train_Percentage")["TER"].agg(["mean", "std"]).reset_index()

plt.figure(figsize=(8, 5))
plt.plot(ter_grouped["Train_Percentage"], ter_grouped["mean"], marker='o', label='TER')
plt.fill_between(ter_grouped["Train_Percentage"],
                 ter_grouped["mean"] - ter_grouped["std"],
                 ter_grouped["mean"] + ter_grouped["std"],
                 alpha=0.2)
plt.xlabel("Training Data Size (%)")
plt.ylabel("Tag Error Rate (TER)")
plt.title("Decimation Plot for CRF: TER vs Training Data Size")
plt.xticks(ter_grouped["Train_Percentage"])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "crf_TER_vs_train_size_CrossValidation.png"))
plt.show()
