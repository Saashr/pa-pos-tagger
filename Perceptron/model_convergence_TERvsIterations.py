import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# === Paths ===
train_path = "examples/pa-ud-train.conllu"
valid_path = "examples/pa-ud-dev.conllu"
output_dir = "outputs"
logs_dir = "logs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# === Clean CoNLL-U files ===
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

# === Load CoNLL-U Data ===
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

# === Clean the files
cleaned_train = train_path.replace(".conllu", "-cleaned.conllu")
cleaned_valid = valid_path.replace(".conllu", "-cleaned.conllu")
clean_conllu_file(train_path, cleaned_train)
clean_conllu_file(valid_path, cleaned_valid)

# === Training Loop with Early Stopping
results = []
best_f1 = 0.0
patience = 5
no_improve_counter = 0
max_iter_limit = 100

for n_iter in range(1, max_iter_limit + 1):
    print(f"\n[INFO] Training Perceptron for {n_iter} iteration(s)...")

    model_path = os.path.join(logs_dir, f"perceptron_model_{n_iter}.dat")
    pred_output = os.path.join(logs_dir, f"pred_perceptron_{n_iter}.conllu")

    # Train the model using your updated tagger.py
    with open(cleaned_train, "r", encoding="utf-8") as train_input:
        subprocess.run(["python3", "tagger.py", "-t", model_path, str(n_iter)],
                       stdin=train_input)

    # Predict
    with open(cleaned_valid, "r", encoding="utf-8") as dev_input, open(pred_output, "w", encoding="utf-8") as out_file:
        subprocess.run(["python3", "tagger.py", model_path],
                       stdin=dev_input, stdout=out_file)

    # Load predictions and gold data
    gold_sents = load_conllu(cleaned_valid)
    pred_sents = load_conllu(pred_output)

    y_true = [tag for sent in gold_sents for _, tag in sent]
    y_pred = [tag for sent in pred_sents for _, tag in sent]

    if len(y_pred) == 0:
        print(f"[ERROR] No predictions at iteration {n_iter}. Skipping.")
        continue

    if len(y_true) != len(y_pred):
        print(f"[WARNING] Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    # Evaluate
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    acc = report['accuracy']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    ter = 1 - acc

    print(f"[RESULT] Iter {n_iter} → Acc: {acc:.4f}, TER: {ter:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")
    results.append((n_iter, acc, ter, precision, recall, f1))

    # Early stopping
    if f1 > best_f1:
        best_f1 = f1
        no_improve_counter = 0
    else:
        no_improve_counter += 1
        print(f"[INFO] No improvement. Patience counter: {no_improve_counter}/5")

    if no_improve_counter >= patience:
        print(f"\n[STOP] Model converged. Early stopping at iteration {n_iter}.")
        break

# === Save results + plot ===
df = pd.DataFrame(results, columns=["Iterations", "Accuracy", "TER", "Precision", "Recall", "F1"])
df.to_csv(os.path.join(output_dir, "perceptron_convergence_metrics_TERvsIterations.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.plot(df["Iterations"], df["TER"], marker='o')
plt.xlabel("Number of Iterations")
plt.ylabel("Tag Error Rate (TER)")
plt.title("Perceptron Convergence: TER vs Iterations")
plt.grid(True)
plt.xticks(df["Iterations"])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "perceptron_convergence_plot_TERvsIterations.png"))
plt.show()
