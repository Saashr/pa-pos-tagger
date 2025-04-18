import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report

# === Paths ===
train_path = "examples/pa-ud-train.conllu"
dev_path = "examples/pa-ud-dev.conllu"
output_dir = "outputs"
logs_dir = "logs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# === Helper: Clean CoNLL-U
def clean_conllu_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        block = []
        for line in infile:
            if line.strip() == "":
                if block:
                    outfile.write("\n".join(block) + "\n\n")
                    block = []
            elif line.startswith("#") or len(line.split('\t')) == 10:
                block.append(line.strip())
        if block:
            outfile.write("\n".join(block) + "\n")

# === Helper: Load CoNLL-U
def load_conllu(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sent = []
        for line in f:
            line = line.strip()
            if line == '':
                if sent:
                    sentences.append(sent)
                    sent = []
            elif not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) == 10:
                    sent.append((parts[1], parts[3]))
    return sentences

# === Clean files
cleaned_train = train_path.replace(".conllu", "-cleaned.conllu")
cleaned_dev = dev_path.replace(".conllu", "-cleaned.conllu")
clean_conllu_file(train_path, cleaned_train)
clean_conllu_file(dev_path, cleaned_dev)

# === Load data
train_data = load_conllu(cleaned_train)
dev_data = load_conllu(cleaned_dev)

# Combine full training pool
training_pool = train_data + dev_data
random.seed(42)
random.shuffle(training_pool)

# === Evaluation data
y_true = [tag for sent in dev_data for _, tag in sent]

# === Loop: % of data used (fixed 100 iterations)
results = []
ter_window = []
convergence_patience = 5
converged_percent = None

for percent in range(10, 110, 10):  # 10% to 100%
    subset_size = int(len(training_pool) * percent / 100)
    subset = training_pool[:subset_size]

    subset_path = os.path.join(logs_dir, f"subset_{percent}.conllu")
    with open(subset_path, "w", encoding="utf-8") as out:
        for sent in subset:
            for token, tag in sent:
                out.write(f"0\t{token}\t_\t{tag}\t_\t_\t_\t_\t_\t_\n")
            out.write("\n")

    model_path = os.path.join(logs_dir, f"perceptron_model_{percent}.dat")
    pred_path = os.path.join(logs_dir, f"pred_{percent}.conllu")

    print(f"\n[INFO] Training on {percent}% data ({subset_size} sentences)")

    # Train
    with open(subset_path, "r", encoding="utf-8") as train_input:
        subprocess.run(["python3", "tagger.py", "-t", model_path, "100"],
                       stdin=train_input)

    # Predict
    with open(cleaned_dev, "r", encoding="utf-8") as dev_input, open(pred_path, "w", encoding="utf-8") as out_file:
        subprocess.run(["python3", "tagger.py", model_path],
                       stdin=dev_input, stdout=out_file)

    # Load predictions
    pred_sents = load_conllu(pred_path)
    y_pred = [tag for sent in pred_sents for _, tag in sent]

    if len(y_pred) != len(y_true):
        print(f"[WARNING] Mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
        continue

    # Evaluate
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    acc = report['accuracy']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    ter = 1 - acc

    print(f"[RESULT] {percent}% data → TER: {ter:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
    results.append((percent, acc, ter, precision, recall, f1))

    # Convergence check based on TER
    ter_window.append(ter)
    if len(ter_window) > convergence_patience:
        ter_window.pop(0)
        if max(ter_window) - min(ter_window) < 0.0005:
            converged_percent = percent
            print(f"\n[STOP] Model converged at {percent}% corpus.")
            break

# === Save results
df = pd.DataFrame(results, columns=["Train_Percentage", "Accuracy", "TER", "Precision", "Recall", "F1"])
df.to_csv(os.path.join(output_dir, "perceptron_convergence_metrics_TERvsCorpusSize.csv"), index=False)

# === Plot
plt.figure(figsize=(8, 5))
plt.plot(df["Train_Percentage"], df["TER"], linestyle='-', color='blue', marker='')




plt.xlabel("Training Data Used (%)")
plt.ylabel("Tag Error Rate (TER)")
plt.title("Perceptron Convergence: TER vs Corpus Size")
plt.grid(True)
plt.xticks(df["Train_Percentage"])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "perceptron_convergence_plot_TERvsCorpusSize.png"))
plt.show()
