import argparse
from sklearn.metrics import classification_report, confusion_matrix

def extract_pos_conllu(fname):
    with open(fname) as f:
        return [line.split("\t")[3] for line in f if line.strip() and not line.startswith("#")]

def main(labels_fname, preds_fname):
    labels = extract_pos_conllu(labels_fname)
    preds = extract_pos_conllu(preds_fname)
    
    # Calculate the set of unique labels
    unique_labels = sorted(set(labels + preds))

    # Print classification report
    print(classification_report(labels, preds, labels=unique_labels))
    
    # Print confusion matrix with labels
    cm = confusion_matrix(labels, preds, labels=unique_labels)
    print("\nConfusion Matrix:")
    print("Labels:", unique_labels)
    print(cm)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("label_file")
    argparser.add_argument("preds_file")
    args = argparser.parse_args()
    main(args.label_file, args.preds_file)
