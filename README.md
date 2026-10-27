<div align="center">

# 🏠 House Price Prediction Model

**An end-to-end machine learning project that predicts house sale prices and serves them through an interactive web app**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

<div align="center">

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Open_App-FF4B4B?style=for-the-badge)](https://house-price-prediction-model-lvpn7gdtfqsc3ykuvfdjuh.streamlit.app/)

</div>

---

## 📌 Overview

Buying or selling a home starts with one question: what is it worth? This project answers it with machine learning. It is trained on the **Ames Housing dataset** and predicts a house's sale price from its features.

The project covers the full workflow: exploratory data analysis, feature engineering, model comparison, tuning, and deployment as a web app where anyone can enter a house's details and get a prediction instantly.

## 🎬 Live Demo

👉 **[Try the app here](https://house-price-prediction-model-lvpn7gdtfqsc3ykuvfdjuh.streamlit.app/)**

> The app may take a few seconds to wake up if it hasn't been used recently.

## ✨ Features

- 🔍 **Exploratory data analysis** to find the key predictors of sale price
- 🧹 **Data cleaning** that handles missing values and outliers
- 🛠️ **Feature engineering** to give the models more useful signals
- ⚙️ **Preprocessing and regression pipelines** built with scikit-learn
- 🏆 **Multiple models compared** to find the best-performing algorithm
- 📈 **Cross-validation and hyperparameter tuning** to improve accuracy and reduce overfitting
- 🌐 **Interactive Streamlit app** for real-time predictions from user input

## 🧠 Project Workflow

```
Raw Data ➜ EDA ➜ Cleaning ➜ Feature Engineering ➜ Pipelines ➜ Model Comparison ➜ Tuning ➜ Streamlit App
```

1. **Explore:** analyse the Ames Housing dataset to understand which features drive sale price.
2. **Clean:** handle missing values and outliers so they don't distort the models.
3. **Engineer:** create and select features that improve predictions.
4. **Build:** set up preprocessing and regression pipelines in scikit-learn.
5. **Compare:** train several models and pick the best-performing algorithm.
6. **Tune:** use cross-validation and hyperparameter tuning to improve accuracy and limit overfitting.
7. **Deploy:** wrap the final model in a Streamlit app for real-time predictions.

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Data Handling | Pandas |
| Machine Learning | scikit-learn |
| Web App | Streamlit |

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- pip

### Installation

1. Download or clone this repository and open a terminal in the project folder.
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   # source venv/bin/activate     # macOS / Linux
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. Enter the house details to see the estimated sale price.

## 🔮 Future Improvements

- Try advanced models such as Gradient Boosting or XGBoost
- Add feature-importance charts to explain each prediction
- Show a prediction range instead of a single number

## 👤 Author

**Asutosh Khairnar**

[![GitHub](https://img.shields.io/badge/GitHub-asutosh8460-181717?style=flat&logo=github)](https://github.com/asutosh8460)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-asutoshkhairnar-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/asutoshkhairnar)

---

<div align="center">

⭐ If you found this project useful, consider giving it a star.

</div>
