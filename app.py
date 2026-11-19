import json
from pathlib import Path
import numpy as np, pandas as pd, joblib
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="EduPro Demand & Revenue Forecasting", page_icon="📈", layout="wide")
ART = Path("artifacts") if Path("artifacts").exists() else Path(".")

@st.cache_resource
def load_bundle():
    return joblib.load(ART / "models.joblib")

@st.cache_data
def load_tables():
    return {"data": pd.read_csv(ART / "course_data.csv"),
            "pred": pd.read_csv(ART / "predictions.csv"),
            "metrics": pd.read_csv(ART / "metrics.csv"),
            "monthly": pd.read_csv(ART / "monthly_revenue.csv", parse_dates=["TransactionDate"]),
            "imp_Enrollments": pd.read_csv(ART / "importance_Enrollments.csv"),
            "imp_Revenue": pd.read_csv(ART / "importance_Revenue.csv"),
            "hist": pd.read_csv(ART / "metrics_history_forecast.csv")
            if (ART / "metrics_history_forecast.csv").exists() else None}

if not (ART / "models.joblib").exists():
    st.error("Run `python train.py` first to create the artifacts folder.")
    st.stop()

B, T = load_bundle(), load_tables()
data, pred = T["data"], T["pred"]
num, ranges = B["num"], B["ranges"]

# ------------------------------------------------ sidebar: what-if course
st.sidebar.header("🎛️ Course what-if simulator")
def rng(c): lo, hi, med = ranges[c]; return lo, hi, med
inp = {}
if "CoursePrice" in num:
    lo, hi, med = rng("CoursePrice"); inp["CoursePrice"] = st.sidebar.slider("Course price", lo, hi, med)
if "CourseDuration" in num:
    lo, hi, med = rng("CourseDuration"); inp["CourseDuration"] = st.sidebar.slider("Duration (hours)", lo, hi, med)
if "CourseRating" in num:
    lo, hi, med = rng("CourseRating"); inp["CourseRating"] = st.sidebar.slider("Course rating", lo, hi, med)
level = st.sidebar.selectbox("Course level", sorted(B["level_map"], key=B["level_map"].get))
if "LevelEnc" in num: inp["LevelEnc"] = B["level_map"][level]
if "YearsOfExperience" in num:
    lo, hi, med = rng("YearsOfExperience"); inp["YearsOfExperience"] = st.sidebar.slider("Instructor experience (yrs)", lo, hi, med)
if "TeacherRating" in num:
    lo, hi, med = rng("TeacherRating"); inp["TeacherRating"] = st.sidebar.slider("Instructor rating", lo, hi, med)
if "ExpertiseMatch" in num:
    inp["ExpertiseMatch"] = st.sidebar.slider("Expertise-category match (0-1)", 0.0, 1.0, 1.0)
for c in B["cat"]:
    inp[c] = st.sidebar.selectbox(c, B["cat_options"][c])
row = pd.DataFrame([inp])[num + B["cat"]]
p_enr = max(float(B["Enrollments"]["model"].predict(row)[0]), 0)
p_rev = max(float(B["Revenue"]["model"].predict(row)[0]), 0)
st.sidebar.metric("Predicted enrollments", f"{p_enr:,.0f}")
st.sidebar.metric("Predicted revenue", f"{p_rev:,.0f}")

st.title("📈 EduPro - Course Demand & Revenue Forecasting")
st.caption("From reactive reporting to proactive planning")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Demand dashboard", "Revenue forecast", "Feature importance", "Category comparison", "Model performance"])

# ------------------------------------------------ 1 demand dashboard
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Courses", f"{len(data):,}")
    c2.metric("Total enrollments", f"{int(data.Enrollments.sum()):,}")
    c3.metric("Total revenue", f"{data.Revenue.sum():,.0f}")
    c4.metric("Avg revenue / enrollment", f"{data.Revenue.sum() / max(data.Enrollments.sum(), 1):,.1f}")
    cats = ["All"] + sorted(data.CourseCategory.unique())
    sel = st.selectbox("Filter category", cats)
    d = pred.merge(data[["CourseID", "CoursePrice", "CourseLevel"]], on="CourseID")
    if sel != "All": d = d[d.CourseCategory == sel]
    top = d.sort_values("Pred_Enrollments", ascending=False).head(15)
    fig = px.bar(top.melt(id_vars="CourseID", value_vars=["Actual_Enrollments", "Pred_Enrollments"]),
                 x="CourseID", y="value", color="variable", barmode="group",
                 title="Top 15 courses by predicted enrollments (cross-validated predictions)")
    st.plotly_chart(fig, width="stretch")
    st.plotly_chart(px.scatter(d, x="CoursePrice", y="Pred_Enrollments", color="CourseLevel",
                               trendline="ols" if len(d) > 3 else None,
                               title="Price vs predicted demand"), width="stretch")

# ------------------------------------------------ 2 revenue forecast
with tab2:
    m = T["monthly"].dropna().copy()
    horizon = st.slider("Forecast horizon (months)", 3, 12, 6)
    t = np.arange(len(m)); k = np.polyfit(t, m.Revenue, 1)
    sd = np.std(m.Revenue - np.polyval(k, t))
    ft = np.arange(len(m), len(m) + horizon)
    fdates = pd.date_range(m.TransactionDate.iloc[-1], periods=horizon + 1, freq="MS")[1:]
    f = pd.DataFrame({"TransactionDate": fdates, "Revenue": np.polyval(k, ft)})
    f["Lower"], f["Upper"] = f.Revenue - 1.96 * sd, f.Revenue + 1.96 * sd
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_scatter(x=m.TransactionDate, y=m.Revenue, name="Actual")
    fig.add_scatter(x=f.TransactionDate, y=f.Revenue, name="Trend forecast", line=dict(dash="dash"))
    fig.add_scatter(x=list(f.TransactionDate) + list(f.TransactionDate[::-1]),
                    y=list(f.Upper) + list(f.Lower[::-1]), fill="toself", opacity=.2,
                    line=dict(width=0), name="95% band")
    fig.update_layout(title="Monthly revenue and linear-trend forecast")
    st.plotly_chart(fig, width="stretch")
    st.caption("Platform-level trend forecast. Course-level forecasts come from the ML models.")
    cr = pred.groupby("CourseCategory")[["Actual_Revenue", "Pred_Revenue"]].sum().reset_index()
    st.plotly_chart(px.bar(cr.melt(id_vars="CourseCategory"), x="CourseCategory", y="value",
                           color="variable", barmode="group",
                           title="Category revenue: actual vs model-predicted"), width="stretch")

# ------------------------------------------------ 3 feature importance
with tab3:
    tgt = st.radio("Target", ["Enrollments", "Revenue"], horizontal=True)
    imp = T[f"imp_{tgt}"].head(12)
    st.plotly_chart(px.bar(imp[::-1], x="Importance", y="Feature", orientation="h",
                           title=f"Key drivers of {tgt.lower()} ({B[tgt]['name']}, permutation importance)"),
                    width="stretch")
    st.info(f"Top driver: **{imp.iloc[0].Feature}**. Higher importance = larger drop in R² when the feature is shuffled.")
    corr = pd.read_csv(ART / "spearman_corr.csv", index_col=0)
    st.subheader("Spearman correlation with targets"); st.dataframe(corr.round(3))

# ------------------------------------------------ 4 category comparison
with tab4:
    g = data.groupby("CourseCategory").agg(Courses=("CourseID", "count"), Enrollments=("Enrollments", "sum"),
                                           Revenue=("Revenue", "sum"), AvgPrice=("CoursePrice", "mean"),
                                           AvgRating=("CourseRating", "mean")).reset_index()
    g["Enroll_per_Course"] = g.Enrollments / g.Courses
    metric = st.selectbox("Compare by", ["Enrollments", "Revenue", "Enroll_per_Course", "AvgPrice", "AvgRating"])
    st.plotly_chart(px.bar(g.sort_values(metric), x=metric, y="CourseCategory", orientation="h"),
                    width="stretch")
    st.dataframe(g.round(2), width="stretch")

# ------------------------------------------------ 5 model performance
with tab5:
    st.subheader("Structural models (course attributes -> demand/revenue)")
    st.dataframe(T["metrics"].round(3), width="stretch")
    if T["hist"] is not None:
        st.subheader("History-based forecast (past performance -> next period)")
        st.dataframe(T["hist"].round(3), width="stretch")
    st.caption("MAE = average error | RMSE = penalises large errors | R² = variance explained. CV = 5-fold cross-validation.")
