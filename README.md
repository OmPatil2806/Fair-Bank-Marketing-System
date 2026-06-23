# Fair Bank Marketing System
### Detecting and Mitigating Socioeconomic Bias in AI-Driven Financial Product Targeting

---

## About This Project

A Portuguese bank uses an AI system to predict which customers are likely to subscribe to a term deposit. Based on this prediction, the bank decides who gets a phone call and who gets excluded from the campaign entirely.

This project investigates whether the AI discriminates against customers with lower education levels — even when their financial profile is identical to university-educated customers. We detect the bias, prove it through three independent methods, and fix it at both the model level and the campaign level.

---

## Dataset

**Source:** [Bank Marketing Dataset — Kaggle](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)

- **Records:** 10,665 customers
- **Target:** Did the customer subscribe to a term deposit? (Yes / No)
- **Sensitive Attribute:** Education Level
  - High Education (0) = Tertiary / University degree
  - Low Education (1) = Primary + Secondary (no university degree)

---

## Key Findings

| Group | Subscription Rate |
|---|---|
| High Education (Tertiary) | 54.11% |
| Low Education (Primary + Secondary) | 43.07% |

The worst case: Young customers with the same high balance were predicted to subscribe at **77.1%** if High Education — but only **51.9%** if Low Education. Same financial profile. Different prediction. Only education changed.

Two customers with **perfectly identical financial profiles** (Jaccard Similarity = 1.0) received **opposite predictions** based solely on education level.

---

## Three-Component Approach

### 1. Fairness Pipeline
Per-group confusion matrices and four error-rate metrics to detect discrimination:

```
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)
FDR = FP / (FP + TP)
FOR = FN / (FN + TN)
```

Fix applied: **Pre-processing Reweighing**
```
w = P(education) x P(outcome) / P(education, outcome)
```

### 2. Similarity Analysis
Proves financially similar customers get different predictions:

```
Jaccard:  JSim(A,B) = |A ∩ B| / |A ∪ B|
Cosine:   Cos(A,B)  = A·B / (|A| × |B|)
```

### 3. Fair Influence Maximization
Ensures the marketing campaign itself reaches both groups equally using Fair TCIM-Budget with fairness constraint c:

```
max Σ f_t(S; Vi, G)   subject to |S| ≤ B
AND unfairness ≤ c
```

---

## Results

| Metric | Before | After | Result |
|---|---|---|---|
| Overall Accuracy | 0.8401 | 0.8368 | -0.33% cost |
| FPR Disparity | 0.0368 | 0.0178 | 51.6% Improved |
| FOR Disparity | 0.1036 | 0.1264 | Worsened* |
| TCIM Unfairness | 0.0215 | 0.0197 | Improved |

*Worsening of some metrics confirms the mathematical Impossibility Result — improving one error rate unavoidably affects others when group base rates differ.

---

## Project Structure

```
├── bank.csv                        # Dataset
├── Bank_Fairness_Assessment.ipynb  # Main notebook
├── Bank_Fairness_Report.docx       # Full written report
└── README.md
```

---

## How to Run

```bash
# Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn networkx

# Place bank.csv in the same folder as the notebook
# Open Jupyter and run all cells
jupyter notebook Bank_Fairness_Assessment.ipynb
```

---

## Libraries

| Library | Purpose |
|---|---|
| pandas | Data loading and manipulation |
| numpy | Numerical calculations |
| matplotlib / seaborn | Visualizations |
| scikit-learn | ML models and metrics |
| networkx | Customer influence network |

---

> Socioeconomic bias in AI is measurable, explainable, and correctable — but requires fixing both the model and the campaign to truly break the cycle of financial exclusion.
