# Airbnb Montreal Churn Prediction

This project builds an end-to-end predictive analytics pipeline to identify **host churn** in Airbnb listings across **Montreal**. It uses **logistic regression** and **Power BI** to classify listings as “churned” or “active” based on historical activity, availability, and host behavior.

---

## 🔍 Key Components

- **Data Preparation and Cleaning**  
  Preprocessed the dataset using `pandas`: filled missing values, one-hot encoded `room_type`, and selected features like `availability_365` and `reviews_per_month`.

- **Feature Engineering**  
  Created a binary `churned` label using availability and reviews as proxies for host activity.

- **Model**  
  Trained a `LogisticRegression` model with `class_weight='balanced'` to handle data imbalance.

- **Evaluation Metrics**  
  - Accuracy: **66%**  
  - AUC Score: **0.75**  
  - Confusion matrix and classification report

- **Export for Visualization**  
  Saved predictions to `airbnb_churn_output.csv` for Power BI dashboarding.

- **Power BI Dashboard**  
  Visualized:
  - Churn rate by **stay duration**
  - Churn across **room types**

---

## ⚙️ Tech Stack

- **Python:** pandas, scikit-learn, matplotlib, seaborn  
- **Power BI:** For dashboards and visual insights  
- **Git & GitHub:** For version control and collaboration

---

## 🚀 Getting Started

```bash
git clone https://github.com/abhayau2001/airbnb-montreal-churn.git
cd airbnb-montreal-churn
pip install -r requirements.txt
python model/airbnb_dynamic_pricing.py
