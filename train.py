"""EduPro - Course demand & revenue forecasting. Run: python train.py [path_to_xlsx]
Outputs -> artifacts/ (models, metrics, predictions) and figures/ (paper charts)."""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/EduPro Online Platform.xlsx")
ART, FIG = Path("artifacts"), Path("figures")
ART.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
SEED = 42
NUM = ["CoursePrice", "CourseDuration", "CourseRating", "LevelEnc",
       "YearsOfExperience", "TeacherRating", "ExpertiseMatch"]
CAT = ["CourseCategory", "CourseType"]
HIST = ["PastEnrollments", "PastAvgRevenue", "PastRevenuePerEnrollment"]
TARGETS = {"Enrollments": "Enrollment Count", "Revenue": "Course Revenue"}

# ---------------------------------------------------------------- 1. load & merge
def load(path):
    sheets = pd.read_excel(path, sheet_name=None)
    s = {k.strip().lower(): v.rename(columns=lambda c: str(c).strip()) for k, v in sheets.items()}
    for need in ("courses", "teachers", "transactions"):
        if need not in s:
            sys.exit(f"Sheet '{need}' not found. Sheets present: {list(sheets)}")
    return s["courses"], s["teachers"], s["transactions"]

def prepare(courses, teachers, tx):
    if "CourseID" not in courses.columns:
        sys.exit("Courses sheet needs CourseID")
    tx = tx.copy()
    tx["TransactionDate"] = pd.to_datetime(tx["TransactionDate"], errors="coerce")
    tx["Amount"] = pd.to_numeric(tx["Amount"], errors="coerce")
    tx = tx.dropna(subset=["CourseID"]).drop_duplicates()
    if "TeacherID" not in courses.columns:
        g = tx.dropna(subset=["TeacherID"]).groupby("CourseID")["TeacherID"]
        main = g.agg(lambda x: x.mode().iloc[0]).rename("TeacherID").reset_index()
        courses = courses.merge(main, on="CourseID", how="left")
    agg = tx.groupby("CourseID").agg(Enrollments=("CourseID", "size"),
                                     Revenue=("Amount", "sum")).reset_index()
    df = (courses.drop_duplicates("CourseID")
          .merge(teachers.drop_duplicates("TeacherID"), on="TeacherID", how="left")
          .merge(agg, on="CourseID", how="left"))
    df[["Enrollments", "Revenue"]] = df[["Enrollments", "Revenue"]].fillna(0)
    for c in ["CoursePrice", "CourseDuration", "CourseRating", "YearsOfExperience", "TeacherRating"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())          # missing ratings -> median
    return df, tx

# ---------------------------------------------------------------- 2. feature engineering
def tier(s, labels):
    return pd.qcut(s.rank(method="first"), len(labels), labels=labels)

def engineer(df):
    df = df.copy()
    df["PriceBand"] = tier(df.CoursePrice, ["Low", "Medium", "High"])
    df["DurationBucket"] = tier(df.CourseDuration, ["Short", "Medium", "Long"])
    df["RatingTier"] = tier(df.CourseRating, ["Low", "Mid", "High"])
    df["ExperienceBucket"] = tier(df.YearsOfExperience, ["Junior", "Mid", "Senior"])
    known = {"beginner": 0, "intermediate": 1, "advanced": 2, "all levels": 1}
    lv = df.CourseLevel.astype(str).str.strip().str.lower()
    df["LevelEnc"] = lv.map(known).fillna(1).astype(int)
    def match(r):
        a = set(str(r.get("Expertise", "")).lower().replace("/", " ").split())
        b = set(str(r.get("CourseCategory", "")).lower().replace("/", " ").split())
        if str(r.get("Expertise", "")).strip().lower() == str(r.get("CourseCategory", "")).strip().lower():
            return 1.0
        return len(a & b) / max(len(b), 1)
    df["ExpertiseMatch"] = df.apply(match, axis=1)
    df["RevenuePerEnrollment"] = np.where(df.Enrollments > 0, df.Revenue / df.Enrollments.clip(lower=1), 0)
    return df, known

def drop_redundant(df, cols, thr=0.9):
    corr = df[cols].corr().abs()
    drop = set()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if a not in drop and b not in drop and corr.loc[a, b] > thr:
                drop.add(b)
    return [c for c in cols if c not in drop], sorted(drop)

# ---------------------------------------------------------------- 3. models
def make_pre(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])

def model_zoo():
    return {"Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1, max_iter=20000),
            "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                          max_depth=3, random_state=SEED)}

def evaluate(X, y, num, cat, tag):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    kf = KFold(5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, m in model_zoo().items():
        pipe = Pipeline([("pre", make_pre(num, cat)), ("m", m)])
        pipe.fit(Xtr, ytr); p = pipe.predict(Xte)
        cv = cross_validate(pipe, X, y, cv=kf, scoring=("r2", "neg_mean_absolute_error",
                                                      "neg_root_mean_squared_error"))
        rows.append(dict(Target=tag, Model=name,
                         MAE=mean_absolute_error(yte, p),
                         RMSE=np.sqrt(mean_squared_error(yte, p)),
                         R2=r2_score(yte, p),
                         CV_MAE=-cv["test_neg_mean_absolute_error"].mean(),
                         CV_RMSE=-cv["test_neg_root_mean_squared_error"].mean(),
                         CV_R2=cv["test_r2"].mean()))
        fitted[name] = pipe
    return pd.DataFrame(rows), fitted, (Xte, yte)

def savefig(name):
    plt.tight_layout(); plt.savefig(FIG / name, dpi=150); plt.close()

# ---------------------------------------------------------------- main
def main():
    courses, teachers, tx = load(DATA)
    df, tx = prepare(courses, teachers, tx)
    df, level_map = engineer(df)
    num, dropped = drop_redundant(df, NUM)
    X = df[num + CAT]
    print(f"Courses: {len(df)} | Transactions: {len(tx)} | dropped redundant: {dropped}")

    # ---- EDA charts
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    df.Enrollments.hist(ax=ax[0], bins=20); ax[0].set_title("Enrollments per course")
    df.Revenue.hist(ax=ax[1], bins=20); ax[1].set_title("Revenue per course")
    savefig("01_target_distributions.png")
    cat_rev = df.groupby("CourseCategory")[["Enrollments", "Revenue"]].sum().sort_values("Revenue")
    cat_rev["Revenue"].plot.barh(figsize=(8, 5), title="Revenue by category"); savefig("02_category_revenue.png")
    plt.figure(figsize=(6, 4.5)); plt.scatter(df.CoursePrice, df.Enrollments, alpha=.6)
    plt.xlabel("Course price"); plt.ylabel("Enrollments"); plt.title("Price vs enrollments")
    savefig("03_price_vs_enrollments.png")
    c = df[NUM + ["Enrollments", "Revenue"]].corr()
    plt.figure(figsize=(8, 6.5)); plt.imshow(c, cmap="coolwarm", vmin=-1, vmax=1); plt.colorbar()
    plt.xticks(range(len(c)), c.columns, rotation=60, ha="right"); plt.yticks(range(len(c)), c.columns)
    plt.title("Correlation matrix"); savefig("04_correlation.png")
    monthly = tx.dropna(subset=["TransactionDate"]).set_index("TransactionDate").resample("MS")["Amount"].sum()
    monthly.rename("Revenue").to_csv(ART / "monthly_revenue.csv")
    monthly.plot(figsize=(9, 4), title="Monthly revenue"); savefig("05_monthly_revenue.png")
    corr = df[NUM + ["Enrollments", "Revenue"]].corr(method="spearman")[["Enrollments", "Revenue"]].drop(
        ["Enrollments", "Revenue"]); corr.to_csv(ART / "spearman_corr.csv")

    # ---- structural models (no leakage; supports what-if app)
    all_metrics, bundle, imp_all, oof = [], {}, {}, df[["CourseID", "CourseCategory"]].copy()
    for t, label in TARGETS.items():
        mt, fitted, (Xte, yte) = evaluate(X, df[t], num, CAT, label)
        all_metrics.append(mt)
        best = mt.sort_values("CV_R2", ascending=False).iloc[0].Model
        print(f"[{label}] best by CV R2: {best}")
        pi = permutation_importance(fitted[best], Xte, yte, n_repeats=30, random_state=SEED, scoring="r2")
        imp = pd.DataFrame({"Feature": Xte.columns, "Importance": pi.importances_mean}).sort_values(
            "Importance", ascending=False)
        imp.to_csv(ART / f"importance_{t}.csv", index=False); imp_all[t] = imp
        best_pipe = Pipeline([("pre", make_pre(num, CAT)), ("m", model_zoo()[best])])
        oof[f"Actual_{t}"] = df[t]
        oof[f"Pred_{t}"] = cross_val_predict(best_pipe, X, df[t], cv=KFold(5, shuffle=True, random_state=SEED)).clip(min=0)
        best_pipe.fit(X, df[t]); bundle[t] = {"model": best_pipe, "name": best}
        plt.figure(figsize=(7, 4.5)); imp.head(10)[::-1].plot.barh(x="Feature", y="Importance", legend=False)
        plt.title(f"Feature importance - {label} ({best})"); savefig(f"06_importance_{t}.png")
        plt.figure(figsize=(5, 5)); plt.scatter(oof[f"Actual_{t}"], oof[f"Pred_{t}"], alpha=.6)
        m = max(oof[f"Actual_{t}"].max(), oof[f"Pred_{t}"].max()); plt.plot([0, m], [0, m], "r--")
        plt.xlabel("Actual"); plt.ylabel("Predicted (CV)"); plt.title(f"{label}: actual vs predicted")
        savefig(f"07_actual_vs_pred_{t}.png")
    pd.DataFrame(oof).to_csv(ART / "predictions.csv", index=False)

    # ---- historical-performance forecast (time split: history -> future period)
    hist_metrics = None
    d = tx.dropna(subset=["TransactionDate"])
    if len(d) > 50:
        cut = d.TransactionDate.quantile(0.7)
        h, f = d[d.TransactionDate <= cut], d[d.TransactionDate > cut]
        hh = h.groupby("CourseID").agg(PastEnrollments=("CourseID", "size"),
                                       PastAvgRevenue=("Amount", "mean"), PastRev=("Amount", "sum"))
        hh["PastRevenuePerEnrollment"] = hh.PastRev / hh.PastEnrollments
        ff = f.groupby("CourseID").agg(FutEnr=("CourseID", "size"), FutRev=("Amount", "sum"))
        dh = df.merge(hh.drop(columns="PastRev"), on="CourseID", how="left").merge(ff, on="CourseID", how="left")
        dh[HIST + ["FutEnr", "FutRev"]] = dh[HIST + ["FutEnr", "FutRev"]].fillna(0)
        Xh = dh[num + HIST + CAT]; rows = []
        for tcol, label in (("FutEnr", "Future Enrollments"), ("FutRev", "Future Revenue")):
            mt, _, _ = evaluate(Xh, dh[tcol], num + HIST, CAT, label); rows.append(mt)
        hist_metrics = pd.concat(rows); hist_metrics.to_csv(ART / "metrics_history_forecast.csv", index=False)
        print("\nHistory-based forecast metrics:\n", hist_metrics.round(3).to_string(index=False))

    metrics = pd.concat(all_metrics); metrics.to_csv(ART / "metrics.csv", index=False)
    print("\nStructural model metrics:\n", metrics.round(3).to_string(index=False))
    df.to_csv(ART / "course_data.csv", index=False)
    bundle.update(num=num, cat=CAT, level_map=level_map,
                  cat_options={c: sorted(df[c].dropna().astype(str).unique()) for c in CAT},
                  ranges={c: [float(df[c].min()), float(df[c].max()), float(df[c].median())] for c in num})
    joblib.dump(bundle, ART / "models.joblib")
    json.dump({"dropped_redundant": dropped, "n_courses": len(df), "n_transactions": len(tx)},
              open(ART / "info.json", "w"))
    print("\nDone. Artifacts in ./artifacts, charts in ./figures")

if __name__ == "__main__":
    main()
