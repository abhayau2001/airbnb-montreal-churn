Airbnb Montreal Churn Prediction Using Logistic Regression and Power BI
Developed an end-to-end predictive analytics pipeline to identify host churn in Airbnb listings specific to Montreal. This project applied logistic regression to classify listings as “churned” or “active” based on historical activity, availability, and listing behavior. The goal was to generate actionable insights into host disengagement using both machine learning and business intelligence tools.

Key Components
Data Preparation and Cleaning
Cleaned and preprocessed the Montreal Airbnb dataset using pandas. Filled missing values, one-hot encoded room_type, and selected relevant features like availability_365 and reviews_per_month.

Feature Engineering
Created a churned label by identifying listings with zero availability or recent reviews — a proxy for inactive hosts.

Model Building with Scikit-learn
Trained a logistic regression model using balanced class weights to address data imbalance. Evaluated with accuracy, ROC-AUC, and classification report.

Export for Visualization
Saved model predictions to airbnb_churn_output.csv for further analysis.

Power BI Dashboard
Designed visuals to show:

Churn rate by stay duration

Churn across room types

Comparison of active vs. churned hosts

Tech Stack
Python: pandas, scikit-learn, matplotlib, seaborn

Power BI: For dashboard creation and insights

Git & GitHub: For version control and public sharing
