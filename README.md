# Seizure Prediction: Preprocessing, Regularisation & Generalisation Study



A controlled experiment in applied machine learning. The classifier is held fixed — plain logistic regression — and everything around it is varied instead: the order of preprocessing steps, the type of regularisation penalty, and the strategy used to handle class imbalance. The goal is to find out how much of a model's performance actually comes from the model, and how much comes from decisions made before the model ever sees the data.

---

## What this project specifies

This is not a single end-to-end pipeline. It's a comparative study built around four specific, falsifiable questions:

1. **Does the order of preprocessing steps change downstream performance**, or does putting steps inside a scikit-learn pipeline make ordering irrelevant in practice?
2. **Which regularisation penalty — L1 (Lasso), L2 (Ridge), or Elastic Net — generalises best** when the same model is tested against three datasets it has never seen?
3. **Is Elastic Net's reputation for combining the strengths of L1 and L2 actually borne out**, or is the improvement marginal?
4. **How does the choice of imbalance-handling strategy interact with the choice of regulariser**, and does that interaction change as imbalance gets worse?

Every other design decision in the project — the datasets, the two preprocessing pipelines, the regularisation sweep, the imbalance-handling comparison — exists to answer one of these four questions with numbers, not intuition.

## Datasets

Three datasets are generated synthetically, built to mirror the statistical signature of three real, frequently-cited seizure-detection corpora. Synthetic generation was a deliberate choice: real intracranial and clinical EEG data carries consent and access restrictions that don't fit a semester timeline, and generating data with a fixed random seed (`np.random.seed(42)`) keeps every number in the report exactly reproducible.

| Dataset | Mirrors | Samples | Seizure Rate | Feature Type |
|---|---|---|---|---|
| DS1 | UCI Epileptic Seizure Recognition (Andrzejak et al., 2001) | 11,500 | 20% | Spectral EEG bands |
| DS2 | CHB-MIT Scalp EEG Database (PhysioNet) | 8,000 | 8% | Time-series statistics |
| DS3 | Kaggle Melbourne intracranial EEG | 6,000 | 5% | Frequency-domain |

The three were chosen specifically to span a difficulty gradient. DS1 is large and only mildly imbalanced — a forgiving baseline. DS3 is small and severely imbalanced — the case closest to real bedside seizure monitoring, and the one every preprocessing or regularisation choice is ultimately judged against.



## What the notebook does

The notebook runs eight stages end to end, each with live, inline-rendered output — no cell depends on an external file:

1. **Setup** — scikit-learn, imbalanced-learn, pandas, matplotlib, seaborn
2. **Dataset generation** — three synthetic EEG-like datasets with controlled class imbalance
3. **Preprocessing pipelines**
   - *Pipeline A (signal-first):* z-score normalisation → 3σ artifact clipping → ANOVA F-test feature selection
   - *Pipeline B (feature-first):* statistical feature extraction → robust scaling → PCA
4. **Baseline logistic regression** — accuracy, F1, precision, recall, PR-AUC, ROC-AUC, computed on all three datasets
5. **Overfitting/underfitting demonstration** — five deliberately constructed bias-variance scenarios, from severe underfitting to severe overfitting
6. **Regularisation study** — L1 vs. L2 vs. Elastic Net swept across six values of `C`, then stability-checked across all three datasets at a fixed `C`
7. **Class-imbalance handling** — class weighting, SMOTE, random oversampling, random undersampling, and SMOTE+weighting, compared on DS3 (the 5%-seizure case)
8. **Comparative analysis** — a full pipeline × regulariser × imbalance-strategy grid across all three datasets, used to answer the four questions above directly

## What was found

| Question | Answer |
|---|---|
| Does preprocessing order affect results? | Yes, on DS1 (+0.0011 F1) — sigma-clipping is only statistically valid after normalisation. The effect is dataset-dependent: DS2 and DS3 show none at this operating point. |
| Which regulariser generalises best? | **Elastic Net** — highest mean F1 (0.978) across the full grid, all three datasets, all three imbalance strategies. |
| Does Elastic Net always beat L1/L2? | Partially. Its advantage grows specifically as class imbalance worsens (DS2, DS3). On the mildly-imbalanced DS1, all three penalties land within about 1% F1 of each other. |
| How does imbalance handling interact with regularisation? | Strongly, and the right combination depends on what's being optimised for. SMOTE + L1 maximises recall. Class weighting + Elastic Net gives the best balance of F1 and PR-AUC. Undersampling + L2 gives the weakest recall of any combination tested. |

**Practical recommendation:** for a deployed seizure predictor, where missing a seizure is far costlier than a false alarm, the combination this project supports is **SMOTE + class weighting + Elastic Net** — not because it wins on any single metric in isolation, but because it degrades least across recall, F1, and PR-AUC together.

## Running the notebook

```bash
pip install imbalanced-learn scikit-learn pandas numpy matplotlib seaborn scipy
jupyter notebook seizure_prediction_notebook.ipynb
```

Or open directly in Google Colab. Every cell runs top to bottom with no external data dependency — all three datasets are generated inside the notebook itself.




## Limitations

- The datasets are synthetic. They're built to mirror the statistical signature of real corpora, not drawn from real patients, so absolute performance numbers (F1 above 0.94 throughout) should not be read as a claim about real clinical accuracy.
- Logistic regression is linear by construction. It cannot capture the nonlinear EEG dynamics that motivated the original Bonn and CHB-MIT research in the first place.
- All results come from a single train/test split per configuration rather than repeated cross-validation. The small F1 gaps reported for preprocessing order and regulariser stability should be read as suggestive, not statistically definitive.

## License

Academic coursework, shared for educational and portfolio purposes.
