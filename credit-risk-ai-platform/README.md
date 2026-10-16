# Credit Card Default Prediction

An ML project that predicts whether a credit card customer will default next
month, built on the real UCI "Default of Credit Card Clients" dataset
(30,000 customers, ~22% default rate).

## What this project does

1. **Data cleaning** — fixes undocumented category codes in the raw data
   (`src/data_prep.py`)
2. **EDA with Pandas, Matplotlib and Seaborn** — class balance, default rate by
   education, correlation heatmap, etc. (`src/eda.py`)
3. **Feature engineering** — adds `CREDIT_UTIL_RATIO` (how much of their credit
   limit a customer is using) and `NUM_DELAYED_PAYMENTS` (how many of the last
   6 months had a late payment) (`src/features.py`)
4. **Model training** — trains and compares Logistic Regression, Decision Tree,
   and Random Forest with `GridSearchCV`, handles the 22%/78% class imbalance
   with `class_weight="balanced"`, and evaluates with accuracy, precision,
   recall, F1-score and a confusion matrix (`src/train.py`)
5. **Deployment** — serves the best model through a FastAPI REST API with
   Pydantic input validation (`api/`)

## Project layout

```
credit-risk-ai-platform/
├── data/
│   └── credit_card_default.csv   # raw UCI dataset
├── src/
│   ├── data_prep.py               # load + clean the data
│   ├── features.py                # the 2 engineered features
│   ├── eda.py                     # EDA plots -> reports/figures/
│   └── train.py                   # train, tune, evaluate, save the best model
├── api/
│   ├── main.py                    # FastAPI app: /health, /predict
│   ├── schemas.py                 # Pydantic request/response models
│   ├── model_service.py           # loads the model, makes predictions
│   └── tests/test_api.py
├── tests/test_features.py         # unit tests for data cleaning & features
├── reports/figures/                # saved EDA + confusion matrix plots
├── models/                        # trained model + metrics (generated)
├── explanation/                    # PDFs: plain-English write-up of every plot, model & decision — read before an interview
├── Dockerfile
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset is already included at `data/credit_card_default.csv`.

## Run it, in order

```bash
# 1. EDA — prints summary stats, saves plots to reports/figures/
python -m src.eda

# 2. Train & compare all 3 models (a couple of minutes — GridSearchCV)
python -m src.train

# 3. Run the tests
pytest tests/ api/tests/ -v

# 4. Start the API
uvicorn api.main:app --reload
```

Then open **http://localhost:8000/docs** for interactive Swagger docs, or:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "limit_bal": 20000, "sex": 2, "education": 2, "marriage": 1, "age": 24,
  "pay_1": 2, "pay_2": 2, "pay_3": -1, "pay_4": -1, "pay_5": -2, "pay_6": -2,
  "bill_amt1": 3913, "bill_amt2": 3102, "bill_amt3": 689,
  "bill_amt4": 0, "bill_amt5": 0, "bill_amt6": 0,
  "pay_amt1": 0, "pay_amt2": 689, "pay_amt3": 0,
  "pay_amt4": 0, "pay_amt5": 0, "pay_amt6": 0
}
EOF
```

Response:

```json
{"default_probability": 0.87, "risk_category": "high_risk"}
```

## Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

