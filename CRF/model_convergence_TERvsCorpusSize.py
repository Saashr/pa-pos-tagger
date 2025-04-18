import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sklearn_crfsuite
import random
from sklearn.metrics import classification_report

# === Paths ===
train_path = "examples/pa-ud-train.conllu"
validation_path = "examples/pa-ud-dev.conllu"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# === Helper functions ===
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

def sent2features(sent): return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent): return [label for token, label in sent]

# === Clean and load data ===
cleaned_train = train_path.replace(".conllu", "-cleaned.conllu")
cleaned_valid = validation_path.replace(".conllu", "-cleaned.conllu")
clean_conllu_file(train_path, cleaned_train)
clean_conllu_file(validation_path, cleaned_valid)

train_data = load_conllu(cleaned_train)
val_data = load_conllu(cleaned_valid)

X_val = [sent2features(s) for s in val_data]
y_val = [sent2labels(s) for s in val_data]

# === Corpus Size Convergence ===
random.seed(42)
results = []

for percent in range(10, 110, 10):  # 10% to 100%
    subset_size = int(len(train_data) * percent / 100)
    train_subset = train_data[:subset_size]  # or use random.sample(train_data, subset_size)

    X_train = [sent2features(s) for s in train_subset]
    y_train = [sent2labels(s) for s in train_subset]

    print(f"\nTraining CRF with {percent}% of corpus ({subset_size} sentences)...")
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,  # fixed
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_val)
    true_tags = [label for sent in y_val for label in sent]
    pred_tags = [label for sent in y_pred for label in sent]

    report = classification_report(true_tags, pred_tags, output_dict=True, zero_division=0)
    accuracy = report['accuracy']
    TER = 1 - accuracy
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']

    print(f"[RESULT] {percent}% data → TER: {TER:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")
    results.append((percent, accuracy, TER, precision, recall, f1))

# === Save metrics ===
df = pd.DataFrame(results, columns=["Train_Percentage", "Accuracy", "TER", "Precision", "Recall", "F1"])
df.to_csv(os.path.join(output_dir, "crf_convergence_metrics_TERvsCorpusSize.csv"), index=False)

# === Plot TER vs Corpus Size ===
plt.figure(figsize=(8, 5))
plt.plot(df["Train_Percentage"], df["TER"], linestyle='-', marker='', color='blue', label="TER")



plt.xlabel("Training Data Used (%)")
plt.ylabel("Tag Error Rate (TER)")
plt.title("CRF Convergence: TER vs Training Corpus Size")
plt.grid(True)
plt.xticks(df["Train_Percentage"])
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "crf_convergence_plot_TERvsCorpusSize.png"))
plt.show()
