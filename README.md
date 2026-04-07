# 🏨 Hotel Booking Cancellation Prediction

## 📌 Project Overview

- Hotel cancellations significantly impact revenue, inventory planning, and customer experience.
- This project builds a machine learning pipeline to predict whether a hotel booking will be canceled using historical booking data.
- The model enables hotels to take proactive actions like dynamic pricing, overbooking strategies, and targeted customer retention.

## 🎯 Problem Statement

Can we predict if a customer will cancel their hotel booking before arrival?

## ✅ Goals
- Predict booking cancellation (is_canceled)
- Identify key factors influencing cancellations
- Build a production-ready ML pipeline
- Ensure robustness and avoid data leakage

## 🤖 Model Hosting
Model is hosted on Hugging Face Hub for efficient deployment

## 📊 Dataset Summary
- Feature	Description
- Total Records	~119,000+
- Features	30+
- Target	is_canceled (0 = No, 1 = Yes)

## 🔍 Feature Types
- 🟣 Categorical: hotel, meal, market_segment, deposit_type, customer_type
- 🔵 Numerical: lead_time, adr, nights stayed, guests

## 🔍 Exploratory Data Analysis (EDA)
📌 Key Insights
- ⏳ Lead Time Effect
→ Longer lead time = higher cancellation probability
- 💰 Deposit Type Matters
→ Non-refundable bookings rarely cancel
- 🔁 Customer Loyalty
→ Repeated guests are significantly less likely to cancel
- 📊 Market Segments
→ Online bookings show higher cancellation rates
- 💸 Pricing Influence (ADR)
→ Higher prices slightly increase cancellation likelihood

## 🧹 Data Preprocessing

✔️ Handling Missing Values
- Categorical → Filled with "Unknown" / Mode
- Numerical → Median imputation
  
✔️ Feature Engineering
- Total stay duration
- Total guests
- Booking behavior patterns

✔️ Encoding
- One-Hot Encoding / Label Encoding

✔️ Pipeline

Used ColumnTransformer + Pipeline to:

- Prevent data leakage
- Maintain reproducibility
- Streamline workflow

## ⚙️ Model Development

🤖 Models Trained
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

## 📈 Model Performance

🤖 Model |	Accuracy | ROC-AUC
- Logistic Regression | ~0.80 |	~0.85
- Random Forest |	~0.87 |	~0.91
- XGBoost |	~0.89 |	~0.93

👉 Ensemble models significantly outperform baseline models

## 🔧 Hyperparameter Tuning

Used RandomizedSearchCV / GridSearchCV

Tuned:
- Tree depth
- Learning rate
- Number of estimators

## 🚀 Result
- Improved generalization
- Reduced overfitting
- Boosted ROC-AUC
  
## 🏆 Final Model
🥇 Selected: XGBoost / Gradient Boosting

Why?

- Best performance
- Handles non-linear relationships
- Strong generalization

## 📤 Predictions & Use Case

The trained model can be used to:

- 📉 Reduce revenue loss from cancellations
- 📊 Improve demand forecasting
- 🎯 Target high-risk customers with offers
- 🏨 Optimize overbooking strategies


## 🧠 Key Learnings
- Feature engineering > complex models
- Business understanding improves ML performance
- Pipelines are essential to avoid leakage
- Ensemble models dominate structured datasets

## 🚀 Future Improvements

- 🔥 Use CatBoost for native categorical handling
- 🧠 Advanced ensembling (stacking/blending)
- 🌐 Deploy via Flask / FastAPI
- 📊 Build dashboard for real-time insights

## 🛠️ Tech Stack

✔️ Language: Python

✔️ Libraries:
- Pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM
- Matplotlib, Seaborn

## 📂 Project Structure

Hotel-Booking-Cancellation-Prediction/

│

├── app.py

├── requirements.txt

├── README.md

└── hotel_booking_prediction.ipynb


## ⚡ How to Run
### #Clone repository
git clone <https://github.com/sarowarahmed/Hotel-Booking-Cancellation-Prediction>

### #Navigate into project
cd Hotel-Booking-Cancellation-Prediction

### #Install dependencies
pip install -r requirements.txt

### #Run notebook
jupyter notebook

## 🌐 Live Demo App
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hotel-cancellation-predictor-sarowar.streamlit.app/)

👉 https://
## 👤 Author

# Sarowar Ahmed

- Data Science Enthusiast
- Kaggle Competitor

## ⭐ Support

If you found this project useful:

- ⭐ Star the repo
- 🍴 Fork it
- 📢 Share with others

## 💡 Recruiter Note
This project demonstrates:

- End-to-end ML pipeline building
- Strong EDA and feature engineering
- Real-world business problem solving
- Model optimization & evaluation
