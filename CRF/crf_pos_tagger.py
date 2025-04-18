import sys
import joblib
from sklearn import metrics as skmetrics
import sklearn_crfsuite
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn_crfsuite import metrics

# Step 1: Prepare Data
# Define the filenames or paths for your training and testing data
train_file = "examples/pa-ud-train.conllu"
test_file = "examples/pa-ud-test.conllu"

# Load your training and testing data from the files
def load_conllu(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        current_sentence = []
        conllu_sentence = []
        for line in f:
            line = line.strip()
            conllu_sentence.append(line)
            if line == '':
                if current_sentence:
                    sentences.append((current_sentence, conllu_sentence))
                    conllu_sentence = []
                    current_sentence = []
            elif not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) == 10:
                    token = parts[1]
                    pos_tag = parts[3]
                    current_sentence.append((token, pos_tag))
    return sentences

train_data = load_conllu(train_file)
test_data = load_conllu(test_file)


# Step 2: Feature Engineering
# Define a function to extract features from a sentence
def word2features(sent, i):
    word = sent[i][0]
    # Add feature extraction logic here
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.first1)': word[0],
        'word.last1': word[-1],

    }
    #print(features, file=sys.stderr)   
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

# Step 3: Train the CRF Model
# Convert your training data into features
X_train = [sent2features(sent[0]) for sent in train_data]
y_train = [sent2labels(sent[0]) for sent in train_data]

# Initialize CRF model
crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)

# Train the model
crf.fit(X_train, y_train)

# Step 4: Evaluate the Model
# Convert testing data into features
X_test = [sent2features(sent[0]) for sent in test_data]
y_test = [sent2labels(sent[0]) for sent in test_data]

# Make predictions
y_pred = crf.predict(X_test)


print('test:', y_test[0:5])
print('pred:',y_pred[0:5])

joblib.dump(crf, 'logs/pos_tagger_model.pkl')
# Evaluate the model

y_pred_conllu = []
for i, j in zip(test_data, y_pred):
    conllu_sent = []
    for line in i[1]:
        if not line:
            continue
        if line.startswith("#"):
            conllu_sent.append(line)
        else:
            fields = line.split("\t")
            if "-" in fields[0] or "." in fields[0]:
                conllu_sent.append(line)
            else:
                idx = int(fields[0]) - 1
                try:
                    fields[3] = j[idx]
                except Exception:
                    import pdb;pdb.set_trace()
                conllu_sent.append("\t".join(fields))
    y_pred_conllu.append("\n".join(conllu_sent))

with open("outputs/crf_pos_predictions.conllu", "w") as f:
    f.write("\n\n".join(y_pred_conllu))

    #print('sent:',i[0])
    #print('conllu:',i[1])
    #print('tags:',j)
    #print()

# res = metrics.classification_report(y_test, y_pred)

#print(res)

# Step 5: Save the trained model

