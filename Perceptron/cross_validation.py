import os
import subprocess
import pandas as pd
from sklearn.metrics import classification_report

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

# Paths
base_dir = ""
initial_folds_dir = os.path.join(base_dir, "folds", "initial")
output_dir = os.path.join(base_dir, "outputs")
logs_dir = os.path.join(base_dir, "logs")
folds_dir = os.path.join(base_dir, "folds")

# Ensure required directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(folds_dir, exist_ok=True)

# Get fold paths
subset_paths = [f"folds/initial/pa-fold-{i + 1}.conllu" for i in range(5)]
performance_metrics = []

# Begin cross-validation
for test_fold in range(5):
    print(f"\n==== Starting Cross-Validation for Test Fold {test_fold + 1} ====")
    #test_file = fold_paths[test_idx]

    for train_size in range(1, 5):  # 1 to 4 folds
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

        # Train model
        model_path = os.path.join(logs_dir, f"model_fold_{test_fold + 1}_{train_size}.dat")
        with open(cleaned_training_file, "r", encoding="utf-8") as train_input:
            subprocess.run(["python3", "tagger.py", "-t", model_path], stdin=train_input)

        # Predict
        pred_output = os.path.join(folds_dir, f"pred_fold_{test_fold+1}_{train_size}.conllu")
        with open(cleaned_validation_file, "r", encoding="utf-8") as val_input, open(pred_output, "w", encoding="utf-8") as out_file:
            subprocess.run(["python3", "tagger.py", model_path], stdin=val_input, stdout=out_file)

        # Load gold and predicted data
        gold_sents = load_conllu(cleaned_validation_file)
        pred_sents = load_conllu(pred_output)

        # Flatten tag sequences
        y_true = [tag for sent in gold_sents for _, tag in sent]
        y_pred = [tag for sent in pred_sents for _, tag in sent]

        # Evaluate
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        accuracy = report["accuracy"]
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1 = report["weighted avg"]["f1-score"]

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
plt.title("Decimation Plot for Perceptron: TER vs Training Data Size")
plt.xticks(ter_grouped["Train_Percentage"])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "perceptron_TER_vs_train_size_CrossValidation.png"))
plt.show()
