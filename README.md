# Project setup

## Enviroment

- **Python Version:** Python >= 3.9

## Project setup

Run the following commands in your terminal:

```bash
git clone https://github.com/Avcuongy/social-analysis-project.git

cd social-anaylysis-project

python -m venv .venv

.venv\Scripts\Activate.ps1

pip install -r requirements.txt

pip install -e .

python scripts/config.py
```

## Result

| Metric | AUC | AUPR | MRR | Precision@10 | Hits@10 | NDCG@10 | Precision@50 | Hits@50 | NDCG@50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | 0.9512 | 0.9578 | 0.9672 | 0.2657 | 1.0 | 0.8490 | 0.0634 | 1.0 | 0.8499 |
| **Random Forest** | 0.9495 | 0.9562 | 0.9649 | 0.2657 | 1.0 | 0.8473 | 0.0634 | 1.0 | 0.8481 |
| **Logistic Regression** | 0.9284 | 0.9400 | 0.9616 | 0.2654 | 1.0 | 0.8437 | 0.0634 | 1.0 | 0.8447 |
| **SVM** | 0.9248 | 0.9374 | 0.9592 | 0.2654 | 1.0 | 0.8419 | 0.0634 | 1.0 | 0.8429 |