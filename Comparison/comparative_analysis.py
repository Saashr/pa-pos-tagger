import pandas as pd
import matplotlib.pyplot as plt

# === Load CSVs (assumed in same directory as script) ===
crf_corpus = pd.read_csv("crf_convergence_metrics_TERvsCorpusSize.csv")
perc_corpus = pd.read_csv("perceptron_convergence_metrics_TERvsCorpusSize.csv")

crf_iter = pd.read_csv("crf_convergence_metrics_TERvsIterations.csv")
perc_iter = pd.read_csv("perceptron_convergence_metrics_TERvsIterations.csv")

crf_cv = pd.read_csv("cross_validation_results_crf.csv")
perc_cv = pd.read_csv("cross_validation_results-perceptron.csv")

# === Average Cross-Validation TER by train size ===
crf_cv_avg = crf_cv.groupby("Train_Percentage")["TER"].mean().reset_index()
perc_cv_avg = perc_cv.groupby("Train_Percentage")["TER"].mean().reset_index()

# === PLOT: TER vs Corpus Size ===
plt.figure(figsize=(8, 5))
plt.plot(crf_corpus["Train_Percentage"], crf_corpus["TER"], label="CRF", marker='o')
plt.plot(perc_corpus["Train_Percentage"], perc_corpus["TER"], label="Perceptron", marker='s')
plt.title("TER vs Corpus Size (Fixed Iterations)")
plt.xlabel("Training Data Used (%)")
plt.ylabel("Tag Error Rate (TER)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("compare_convergence_TER_vs_corpus_size.png")
plt.show()

# === PLOT: TER vs Iterations ===
plt.figure(figsize=(8, 5))
plt.plot(crf_iter["Iterations"], crf_iter["TER"], label="CRF", marker='o')
plt.plot(perc_iter["Iterations"], perc_iter["TER"], label="Perceptron", marker='s')
plt.title("TER vs Iterations (Fixed Corpus)")
plt.xlabel("Number of Iterations")
plt.ylabel("Tag Error Rate (TER)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("compare_convergence_TER_vs_iterations.png")
plt.show()

# === PLOT: TER vs Corpus Size (Cross-Validation) ===
plt.figure(figsize=(8, 5))
plt.plot(crf_cv_avg["Train_Percentage"], crf_cv_avg["TER"], label="CRF (CV)", marker='o')
plt.plot(perc_cv_avg["Train_Percentage"], perc_cv_avg["TER"], label="Perceptron (CV)", marker='s')
plt.title("TER vs Corpus Size (Cross-Validation)")
plt.xlabel("Training Data Used (%)")
plt.ylabel("Tag Error Rate (TER)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("compare_crossval_TER_vs_corpus_size_cv.png")
plt.show()
