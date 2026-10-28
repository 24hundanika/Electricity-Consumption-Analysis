import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_auc_score

st.set_page_config(page_title="Electricity Consumption Dashboard", page_icon="⚡", layout="wide")

SEASONS = ["Winter", "Summer", "Monsoon", "Post-monsoon"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FEATURES = ["Avg_Temp_C", "Max_Temp_C", "Is_Weekend", "Is_Holiday", "Month"]
CLF_FEATURES = FEATURES + ["Outages"]


@st.cache_data
def load_data():
    hourly = pd.read_csv("electricity_hourly.csv", parse_dates=["Timestamp", "Date"])
    daily = pd.read_csv("electricity_daily.csv", parse_dates=["Date"])
    return hourly, daily


@st.cache_resource
def train_models(daily):
    X, y = daily[FEATURES], daily["Total_kWh"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso Regression": make_pipeline(StandardScaler(), Lasso(alpha=1.0)),
        "Decision Tree": make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=5, random_state=42)),
        "Random Forest": make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)),
    }
    rows, preds = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        preds[name] = p
        rows.append({"Model": name, "R2": r2_score(yte, p), "RMSE": np.sqrt(mean_squared_error(yte, p)),
                     "MAE": mean_absolute_error(yte, p)})
    res = pd.DataFrame(rows).set_index("Model").sort_values("R2", ascending=False)
    best = res.index[0]
    cv = cross_val_score(models[best], X, y, cv=5, scoring="r2")

    d = daily.copy()
    d["High_Demand"] = (d["Total_kWh"] > d["Total_kWh"].quantile(0.75)).astype(int)
    a, b, c, e = train_test_split(d[CLF_FEATURES], d["High_Demand"], test_size=0.2, random_state=42, stratify=d["High_Demand"])
    logreg = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=42)).fit(a, c)
    log_acc = accuracy_score(e, logreg.predict(b))
    log_auc = roc_auc_score(e, logreg.predict_proba(b)[:, 1])

    d["Level"] = pd.qcut(d["Total_kWh"], 3, labels=["Low", "Medium", "High"]).astype(str)
    a, b, c, e = train_test_split(d[CLF_FEATURES], d["Level"], test_size=0.2, random_state=42, stratify=d["Level"])
    cvs = [cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)), a, c, cv=5).mean() for k in range(1, 26)]
    best_k = int(np.argmax(cvs)) + 1
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(a, c)
    knn_acc = accuracy_score(e, knn.predict(b))
    return dict(res=res, best=best, model=models[best], cv=cv, yte=yte, pred=preds[best], logreg=logreg,
                log_acc=log_acc, log_auc=log_auc, knn_acc=knn_acc, best_k=best_k, knn_cv=cvs, models=models)


hourly, daily = load_data()
M = train_models(daily)

st.sidebar.title("⚡ Electricity Dashboard")
page = st.sidebar.radio("Go to", ["Overview", "Patterns", "Statistical tests", "Predictions"])
st.sidebar.caption("Electricity Consumption Pattern Analysis · hourly smart-meter data, 2023")

st.title("⚡ Electricity Consumption Pattern Analysis")

# ------------------------------------------------------------------ Overview
if page == "Overview":
    prof = hourly.groupby(["Hour", "Consumer_Type"])["Consumption_kWh"].mean().unstack()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average daily demand", f"{daily['Total_kWh'].mean():,.0f} kWh")
    c2.metric("System peak hour", f"{int(prof.sum(axis=1).idxmax())}:00")
    c3.metric("Highest day", f"{daily['Total_kWh'].max():,.0f} kWh")
    c4.metric("Hourly readings", f"{len(hourly):,}")

    left, right = st.columns(2)
    with left:
        st.subheader("Average hourly load profile")
        st.line_chart(prof)
    with right:
        st.subheader("Share of annual consumption")
        share = hourly.groupby("Consumer_Type")["Consumption_kWh"].sum()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(share.values, labels=share.index, autopct="%1.1f%%", startangle=90)
        st.pyplot(fig)

    st.subheader("Daily demand through the year")
    yr = daily.set_index("Date")[["Total_kWh"]].rename(columns={"Total_kWh": "Daily"})
    yr["7-day average"] = yr["Daily"].rolling(7).mean()
    st.line_chart(yr)

    st.subheader("Peak-hour table")
    pk = pd.DataFrame({"Peak hour": prof.idxmax(), "Peak load (kWh)": prof.max().round(1), "Average load (kWh)": prof.mean().round(1)})
    pk["Load factor"] = (pk["Average load (kWh)"] / pk["Peak load (kWh)"]).round(2)
    st.dataframe(pk)

# ------------------------------------------------------------------ Patterns
elif page == "Patterns":
    st.subheader("Weekly pattern: day of week × hour")
    th = hourly.groupby("Timestamp")["Consumption_kWh"].sum().reset_index()
    th["Hour"] = th["Timestamp"].dt.hour
    th["Day"] = th["Timestamp"].dt.day_name()
    heat = th.pivot_table(index="Day", columns="Hour", values="Consumption_kWh", aggfunc="mean").reindex(DAYS)
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(heat, cmap="YlOrRd", cbar_kws={"label": "Average total load (kWh)"}, ax=ax)
    st.pyplot(fig)

    a, b = st.columns(2)
    with a:
        st.subheader("Average daily demand by month")
        mo = daily.groupby("Month")["Total_kWh"].mean()
        mo.index = [MONTHS[i - 1] for i in mo.index]
        st.bar_chart(mo)
    with b:
        st.subheader("Demand by season")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=daily, x="Season", y="Total_kWh", order=SEASONS, hue="Season", palette="Set2", legend=False, ax=ax)
        st.pyplot(fig)

    c, d = st.columns(2)
    with c:
        st.subheader("Temperature vs daily demand")
        st.scatter_chart(daily, x="Avg_Temp_C", y="Total_kWh", color="Season")
    with d:
        st.subheader("Average demand by weekday")
        wk = daily.groupby(daily["Date"].dt.day_name())["Total_kWh"].mean().reindex(DAYS)
        st.bar_chart(wk)

    st.subheader("Unusual days (IQR rule)")
    q1, q3 = daily["Total_kWh"].quantile([0.25, 0.75])
    iqr = q3 - q1
    out = daily[(daily["Total_kWh"] < q1 - 1.5 * iqr) | (daily["Total_kWh"] > q3 + 1.5 * iqr)]
    st.write(f"Fences: {q1 - 1.5 * iqr:,.0f} to {q3 + 1.5 * iqr:,.0f} kWh → **{len(out)}** unusual day(s).")
    if len(out):
        st.dataframe(out[["Date", "Total_kWh", "Avg_Temp_C"]])

# ------------------------------------------------------------------ Statistical tests
elif page == "Statistical tests":
    wd = daily.loc[daily["Is_Weekend"] == 0, "Total_kWh"]
    we = daily.loc[daily["Is_Weekend"] == 1, "Total_kWh"]
    t, p = stats.ttest_ind(wd, we, equal_var=False)
    diff = wd.mean() - we.mean()
    v1, v2 = wd.var() / len(wd), we.var() / len(we)
    dof = (v1 + v2) ** 2 / (v1 ** 2 / (len(wd) - 1) + v2 ** 2 / (len(we) - 1))
    lo, hi = stats.t.interval(0.95, dof, loc=diff, scale=np.sqrt(v1 + v2))
    st.subheader("Weekday vs weekend (Welch t-test)")
    a, b, c = st.columns(3)
    a.metric("Weekday mean", f"{wd.mean():,.0f} kWh")
    b.metric("Weekend mean", f"{we.mean():,.0f} kWh")
    c.metric("Difference", f"{diff:,.0f} kWh", f"95% CI {lo:,.0f} to {hi:,.0f}")
    st.write(f"t = {t:.2f}, p = {p:.3g} → " + ("**significant difference**" if p < 0.05 else "no significant difference"))

    st.subheader("Seasons (one-way ANOVA)")
    F, pa = stats.f_oneway(*[daily.loc[daily["Season"] == s, "Total_kWh"] for s in SEASONS])
    st.write(f"F = {F:.1f}, p = {pa:.3g} → " + ("**demand differs between seasons**" if pa < 0.05 else "no significant difference"))
    st.dataframe(daily.groupby("Season")["Total_kWh"].agg(["mean", "std", "min", "max"]).reindex(SEASONS).round(0))

    st.subheader("Temperature vs demand (correlation)")
    pr, pp = stats.pearsonr(daily["Avg_Temp_C"], daily["Total_kWh"])
    sr, _ = stats.spearmanr(daily["Avg_Temp_C"], daily["Total_kWh"])
    st.write(f"Pearson r = {pr:.3f} (p = {pp:.3g}), Spearman ρ = {sr:.3f}")
    cols = ["Total_kWh", "Avg_Temp_C", "Max_Temp_C", "Is_Weekend", "Is_Holiday", "Outages", "Month"]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(daily[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

# ------------------------------------------------------------------ Predictions
else:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Predict a day's demand")
        avg_t = st.slider("Average temperature (°C)", 10.0, 42.0, 30.0, 0.5)
        max_t = st.slider("Maximum temperature (°C)", 15.0, 48.0, 36.0, 0.5)
        month = st.select_slider("Month", options=list(range(1, 13)), value=5, format_func=lambda m: MONTHS[m - 1])
        outages = st.slider("Expected power cuts", 0, 5, 0)
        weekend = st.checkbox("Weekend")
        holiday = st.checkbox("Public holiday")
        max_t = max(max_t, avg_t)
        row = pd.DataFrame([[avg_t, max_t, int(weekend), int(holiday), month]], columns=FEATURES)
        kwh = float(M["model"].predict(row)[0])
        prob = float(M["logreg"].predict_proba(pd.DataFrame([[avg_t, max_t, int(weekend), int(holiday), month, outages]], columns=CLF_FEATURES))[0, 1])
        st.metric(f"Predicted total demand ({M['best']})", f"{kwh:,.0f} kWh", f"{kwh / 1000:,.1f} MWh", delta_color="off")
        st.metric("Chance of a high-demand day", f"{prob * 100:.0f}%")
        st.progress(min(max(prob, 0.0), 1.0))
        st.write("🔴 **High-demand day likely**" if prob > 0.5 else "🟢 Normal day expected")
    with right:
        st.subheader("Model comparison (test set)")
        st.dataframe(M["res"].round(3))
        st.caption(f"Best model: {M['best']} · 5-fold CV R² = {M['cv'].mean():.3f} (± {M['cv'].std():.3f})")
        st.write(f"Logistic Regression accuracy **{M['log_acc']:.3f}**, AUC **{M['log_auc']:.3f}** · "
                 f"KNN (k={M['best_k']}) accuracy **{M['knn_acc']:.3f}**")

    a, b = st.columns(2)
    with a:
        st.subheader("Actual vs predicted")
        ap = pd.DataFrame({"Actual": M["yte"].values, "Predicted": M["pred"]})
        st.scatter_chart(ap, x="Actual", y="Predicted")
    with b:
        st.subheader("Typical-weekday forecast by month")
        clim = daily.groupby("Month")[["Avg_Temp_C", "Max_Temp_C"]].mean()
        fc = pd.DataFrame({"Avg_Temp_C": clim["Avg_Temp_C"], "Max_Temp_C": clim["Max_Temp_C"],
                           "Is_Weekend": 0, "Is_Holiday": 0, "Month": clim.index})[FEATURES]
        out = pd.DataFrame({"Predicted typical weekday": M["model"].predict(fc),
                            "Observed monthly average": daily.groupby("Month")["Total_kWh"].mean().values},
                           index=[MONTHS[m - 1] for m in clim.index])
        st.line_chart(out)
