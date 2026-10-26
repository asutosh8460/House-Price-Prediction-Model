import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- App Title & Sidebar Layout ---
st.set_page_config(page_title="House Price Prediction Dashboard", layout="wide")
st.title("🏡 House Price Prediction Dashboard")
st.markdown("This dashboard walks through Data Collection, EDA, Pipeline Preprocessing, and Model Training.")

# --- Phase 2: Data Collection ---
st.header("📦 Phase 2 — Data Collection")

@st.cache_data  # Cache the data so it doesn't download on every reload
def load_data():
    housing = fetch_openml(name='house_prices', as_frame=True, parser='auto')
    return housing.frame

df = load_data().copy()

col1, col2 = st.columns(2)
with col1:
    st.metric("Dataset Shape (Rows, Columns)", f"{df.shape[0]} x {df.shape[1]}")
with col2:
    st.subheader("Data Types Count")
    st.write(df.dtypes.value_counts())

st.subheader("Dataset Preview (First 5 Rows)")
st.dataframe(df.head())

# --- Phase 3: Exploratory Data Analysis (EDA) ---
st.header("📊 Phase 3 — Explanatory Data Analysis (EDA)")

df["SalePrice"] = df["SalePrice"].astype(float)

# 1. Target Distribution
st.subheader("1. Target Price Distribution")
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(df["SalePrice"], kde=True, ax=ax1)
ax1.set_title("Sale price distribution (skewed)")
sns.histplot(np.log1p(df["SalePrice"]), kde=True, color="teal", ax=ax2)
ax2.set_title("Log-transformed (normal)")
plt.tight_layout()
st.pyplot(fig1)

# 2. Missing Values & Correlations
col3, col4 = st.columns(2)
with col3:
    st.subheader("2. Missing Values (Top 20)")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    st.dataframe(missing.head(20))

with col4:
    st.subheader("3. Correlation with SalePrice (Top 10)")
    num_cols = df.select_dtypes(include=np.number).columns
    corr = df[num_cols].corr()["SalePrice"].sort_values(ascending=False)
    st.dataframe(corr.head(10))

# 4. Key Scatterplots
st.subheader("4. Key Features vs. Sale Price Scatterplots")
fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["GrLivArea", "OverallQual", "GarageCars"]):
    ax.scatter(df[col], df["SalePrice"], alpha=0.4, s=10)
    ax.set_xlabel(col)
    ax.set_ylabel("SalePrice")
plt.tight_layout()
st.pyplot(fig2)

# --- Phase 4: Preprocessing Pipeline ---
# Log-transform target
y = np.log1p(df['SalePrice'])
X = df.drop(columns=['SalePrice'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nums_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_train.select_dtypes(include="object").columns.tolist()

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("encode", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, nums_cols),
    ("cat", cat_pipeline, cat_cols)
])

# --- Phase 5: Model Building and Evaluation ---
st.header("🤖 Phase 5 — Model Building & Evaluation")

# Optimization: Only run training once using session state or a button
if st.button("🚀 Train & Evaluate Models (Includes Hyperparameter Tuning)"):
    with st.spinner("Training models and running GridSearchCV... please wait..."):
        
        models = {
            "Ridge": Ridge(alpha=10),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
        }

        results = {}
        for name, model in models.items():
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            results[name] = {"RMSE (log)": round(rmse, 4), "R2": round(r2, 4)}
        
        st.subheader("Baseline Model Performance")
        st.dataframe(pd.DataFrame(results).T)

        # GridSearchCV Tuning
        param_grid = {
            "model__n_estimators":,
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth":,
        }

        best_pipe = Pipeline([("prep", preprocessor),
                              ("model", GradientBoostingRegressor(random_state=42))])

        gs = GridSearchCV(best_pipe, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
        gs.fit(X_train, y_train)
        
        st.success("Tuning Complete!")
        st.write("**Best Parameters:**", gs.best_params_)

        # Final predictions evaluation
        final_model = gs.best_estimator_
        final_preds = final_model.predict(X_test)

        # Visualizing results side-by-side
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("Actual vs. Predicted Prices")
            fig3, ax_eval = plt.subplots(figsize=(6, 5))
            ax_eval.scatter(y_test, final_preds, alpha=0.4, s=12)
            ax_eval.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
            ax_eval.set_xlabel("Actual log price")
            ax_eval.set_ylabel("Predicted log price")
            st.pyplot(fig3)

        with col6:
            st.subheader("Top 15 Feature Importances")
            gb_model = final_model.named_steps["model"]
            all_cols = final_model.named_steps["prep"].get_feature_names_out()
            importances = pd.Series(gb_model.feature_importances_, index=all_cols)
            top15 = importances.nlargest(15)
            
            fig4, ax_imp = plt.subplots(figsize=(8, 5))
            top15.sort_values().plot(kind="barh", color="steelblue", ax=ax_imp)
            st.pyplot(fig4)

        # Reverse Log Conversions
        pred_prices = np.expm1(final_preds)
        actual_prices = np.expm1(y_test.values)
        
        st.subheader("Final Valuation Summaries ($)")
        st.info(f"💡 **Mean Predicted House Price:** ${pred_prices.mean():,.0f}  |  **Mean Actual House Price:** ${actual_prices.mean():,.0f}")
else:
    st.info("Click the 'Train & Evaluate Models' button above to run the machine learning workloads.")
