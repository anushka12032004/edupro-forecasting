# EduPro - Predictive Modeling for Course Demand and Revenue Forecasting

Predicts **enrollment count**, **course revenue** and **category revenue** from course, instructor and transaction data, and serves the results in a Streamlit dashboard.

## Run
```bash
pip install -r requirements.txt
# put "EduPro Online Platform.xlsx" in data/
python train.py            # builds artifacts/ and figures/
streamlit run app.py
```

## Pipeline
1. Merge Courses <-> Teachers <-> Transactions; aggregate transactions per course.
2. Missing ratings -> median; courses with no sales -> 0 enrollments/revenue.
3. Features: price band, duration bucket, rating tier, level encoding, experience bucket, teacher rating, expertise-category match; one-hot for category/type; standardised numerics; redundant features (|r|>0.9) dropped.
4. Models: Linear, Ridge, Lasso, Random Forest, Gradient Boosting. Metrics: MAE, RMSE, R2 (80/20 hold-out + 5-fold CV).
5. Permutation feature importance on the best model per target.
6. Two modes: *structural* (course attributes only, no leakage, powers the what-if simulator) and *history-based* (past enrollments/revenue -> next period via time split).

## App modules
Demand dashboard | Revenue forecast | Feature importance | Category comparison | Model performance, plus a sidebar what-if simulator (price, duration, level, instructor experience and rating).
