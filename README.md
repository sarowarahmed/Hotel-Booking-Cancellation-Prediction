# Hotel Booking Cancellation Prediction
Predicting whether a customer will honor or cancel their hotel reservation to help hotels optimize revenue management.

## Project Overview
This project analyzes a dataset of hotel bookings to identify key factors leading to cancellations. I developed a classification model that predicts the likelihood of a booking being canceled, allowing hotel management to implement better overbooking strategies.

## Key Features & Workflow
Data Cleaning: Handled missing values in 'children' and 'country' columns.

Feature Engineering: Encoded categorical variables (Meal, Market Segment, etc.) and scaled numerical features.

Exploratory Data Analysis (EDA): Visualized booking trends and lead-time impacts on cancellations.

Modeling: Try and Implemented different Classifier Models to handle non-linear relationships.

Evaluation: Achieved high precision and recall, visualized via a Confusion Matrix.

## Tech Stack
Language: Python

Libraries: Pandas, Scikit-Learn, Matplotlib, Seaborn

Environment: Kaggle Kernels / Jupyter

## Results
The model successfully identifies high-risk bookings, with Lead Time and Total Of Special Requests emerging as the most significant predictors.
