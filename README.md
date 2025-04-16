# Stock-RAG-SHAP-DeepSeek

This project demonstrates an innovative integration of SHAP (SHapley Additive exPlanations), XGBoost, and Retrieval-Augmented Generation (RAG) using DeepSeek to enhance interpretability in stock price prediction. It allows users to interactively understand predictions by connecting stock data modeling with explainable AI and context-aware retrieval.

## 🔍 Features

- 📈 Stock Price Prediction using XGBoost
- 🔎 Model Explainability using SHAP visualizations
- 🔁 Retrieval-Augmented Generation (RAG) using DeepSeek
- 📊 Context-based responses for enhanced financial reasoning

## 🧠 Project Architecture

```
        ┌────────────┐
        │  Stock CSV │
        └────┬───────┘
             ▼
    ┌────────────────────┐
    │  XGBoost Prediction │
    └────────┬────────────┘
             ▼
     ┌──────────────┐
     │  SHAP Export │────► SHAP Plot
     └──────────────┘
             ▼
     ┌──────────────────────────┐
     │ DeepSeek Prompt Engine   │────► LLM-based Response
     └──────────────────────────┘
```

## 🗂️ Repository Contents

- `Vantage_test.py` – Pulls and prepares stock data
- `shapExport.py` – Generates SHAP explanations and plots
- `promptDeepSeek.py` – Executes the DeepSeek-based RAG process
- `xgb_model_subset.pkl` – Trained XGBoost model (Pickle format)
- `xgb_model_subset.onnx` – ONNX version of the XGBoost model
- `Stock_data.csv` – Complete training dataset
- `X_test_subset.csv` / `y_test_subset.csv` – Subset test samples
- `Shap Plot.png` – SHAP value heatmap

## ⚙️ Setup Instructions

1. **Clone the Repository**:
```bash
git clone https://github.com/alankit04/Stock-RAG-SHAP-DeepSeek.git
cd Stock-RAG-SHAP-DeepSeek
```

2. **Create a Virtual Environment (Recommended)**:
```bash
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

> Note: If `requirements.txt` is not available, install:
```bash
pip install pandas numpy shap xgboost matplotlib onnx openai
```

## 🚀 How to Run

1. **Preprocess Stock Data**:
```bash
python Vantage_test.py
```

2. **Generate SHAP Interpretations**:
```bash
python shapExport.py
```
This will produce `Shap Plot.png`.

3. **Run DeepSeek Prompt with RAG**:
```bash
python promptDeepSeek.py
```
This integrates prediction output with retrieved knowledge via DeepSeek to generate interpretable responses.

## 📷 Sample Output

- SHAP plot image reveals top features driving predictions.
- RAG pipeline provides data-aware answers like:  
  _"The price spike was largely driven by higher volume and MACD divergence, consistent with similar historical movements."_

## 📄 License

This repository is licensed under the **MIT License**.

## 🙏 Acknowledgements

- [SHAP](https://github.com/slundberg/shap) – For game-theoretic explainability
- [XGBoost](https://github.com/dmlc/xgboost) – Core model engine
- [DeepSeek](https://github.com/deepseek-ai) – Retrieval and generation API

---

**Project Maintainer**: [Alankrit Srivastava](https://github.com/alankit04)
