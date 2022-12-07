import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from plotnine import *
import statsmodels.api as sm
import statsmodels.formula.api as smf

# load in pre-cleaned deaths data for Washington and comparison states
deaths = pd.read_csv("05_cleaned_data/deaths_wa.csv")
# two separate dfs - one for washington and one for comp states
washington = deaths[deaths["StateName"] == "Washington"]
# comp = deaths[deaths["StateName"] != "Washington"]
comp = deaths[
    (deaths["StateName"] == "Oregon")
    | (deaths["StateName"] == "Idaho")
    | (deaths["StateName"] == "Montana")
    | (deaths["StateName"] == "Nevada")
    | (deaths["StateName"] == "Wyoming")
]


def select_overdose(record):
    """Simple function to select only overdose records"""

    if record == "All other non-drug and non-alcohol causes":
        return 0

    if record == "All other alcohol-induced causes":
        return 0

    if record == "All other drug-induced causes":
        return 0

    if record == "Alcohol poisonings (overdose) (X45, X65, Y15)":
        return 0

    if record == "Drug poisonings (overdose) Unintentional (X40-X44)":
        return 1

    if record == "Drug poisonings (overdose) Suicide (X60-X64)":
        return 1

    if record == "Drug poisonings (overdose) Undetermined (Y10-Y14)":
        return 1

    else:
        return "error"


# copy to fix the dreaded "A value is trying to be set on a copy of a slice" error
wa_deaths = washington.copy()
comp_deaths = comp.copy()

# apply new function to our df
wa_deaths["overdose"] = wa_deaths["Drug/Alcohol Induced Cause"].apply(
    lambda x: select_overdose(x)
)
comp_deaths["overdose"] = comp_deaths["Drug/Alcohol Induced Cause"].apply(
    lambda x: select_overdose(x)
)

# filter accordingly based on new column
wa_deaths = wa_deaths[wa_deaths["overdose"] != 0]
comp_deaths = comp_deaths[comp_deaths["overdose"] != 0]

wa_deaths["overdose_per_100k"] = wa_deaths["Deaths"] / wa_deaths["Population"] * 100_000
comp_deaths["overdose_per_100k"] = (
    comp_deaths["Deaths"] / comp_deaths["Population"] * 100_000
)

wa_result = (
    wa_deaths.groupby(["Year", "County"])["overdose_per_100k"].sum().reset_index()
)
comp_result = (
    comp_deaths.groupby(["Year", "StateName", "CountyName"])["overdose_per_100k"]
    .sum()
    .reset_index()
)
comp_result[comp_result["StateName"] == "Oregon"].groupby("Year")[
    "overdose_per_100k"
].mean()
comp_result[comp_result["StateName"] == "Idaho"].groupby("Year")[
    "overdose_per_100k"
].mean()
comp_result[comp_result["StateName"] == "Montana"].groupby("Year")[
    "overdose_per_100k"
].mean()
comp_result[comp_result["StateName"] == "Nevada"].groupby("Year")[
    "overdose_per_100k"
].mean()
comp_result[comp_result["StateName"] == "Wyoming"].groupby("Year")[
    "overdose_per_100k"
].mean()
wa_result.groupby("Year")["overdose_per_100k"].mean()
comp_result.groupby("Year")["overdose_per_100k"].mean()
wa_result = wa_result.groupby("Year")["overdose_per_100k"].mean().reset_index()

comp_result = comp_result.groupby(["Year"])["overdose_per_100k"].mean().reset_index()
# create a scale for number of years before and after 2012 (target year)


def scale_years(year):
    if year == 2009:
        return -3
    if year == 2010:
        return -2
    if year == 2011:
        return -1
    if year == 2012:
        return 0
    if year == 2013:
        return 1
    if year == 2014:
        return 2
    if year == 2015:
        return 3


wa_result["year relative to policy"] = wa_result["Year"].apply(lambda x: scale_years(x))
comp_result["year relative to policy"] = comp_result["Year"].apply(
    lambda x: scale_years(x)
)

# double check no nulls in "year relative to policy"

assert wa_result["year relative to policy"].isnull().sum() == 0
assert comp_result["year relative to policy"].isnull().sum() == 0

# doing this in case the float == int comparison causes issues
# split into before 2007 and after 2007

wa_b4 = wa_result[wa_result["Year"] < 2012]
wa_after = wa_result[wa_result["Year"] >= 2012]


wa_after = wa_after[wa_after["Year"] != 2012]  # may need to handle this differently

from sklearn.linear_model import LinearRegression

regressor_b4 = LinearRegression()
regressor_after = LinearRegression()


X_b4 = np.array(wa_b4["year relative to policy"]).reshape(-1, 1)
y_b4 = np.array(wa_b4["overdose_per_100k"]).reshape(-1, 1)

X_after = np.array(wa_after["year relative to policy"]).reshape(-1, 1)
y_after = np.array(wa_after["overdose_per_100k"]).reshape(-1, 1)


regressor_b4.fit(X_b4, y_b4)
regressor_after.fit(X_after, y_after)


y_pred_b4 = regressor_b4.predict(X_b4)
y_pred_after = regressor_after.predict(X_after)

# Diff-in-Diff

comp_b4 = comp_result[comp_result["Year"] < 2012]
comp_after = comp_result[comp_result["Year"] >= 2012]

comp_after = comp_after[
    comp_after["Year"] != 2012
]  # may need to handle this differently
regressor_b41 = LinearRegression()
regressor_after1 = LinearRegression()


X_b41 = np.array(comp_b4["year relative to policy"]).reshape(-1, 1)
y_b41 = np.array(comp_b4["overdose_per_100k"]).reshape(-1, 1)

X_after1 = np.array(comp_after["year relative to policy"]).reshape(-1, 1)
y_after1 = np.array(comp_after["overdose_per_100k"]).reshape(-1, 1)


regressor_b41.fit(X_b41, y_b41)
regressor_after1.fit(X_after1, y_after1)


y_pred_b41 = regressor_b41.predict(X_b41)
y_pred_after1 = regressor_after1.predict(X_after1)

plt.xlim(-3, 3)
# plt.ylim(0, 500)
plt.title("Overdose Deaths before and after policy implementation in Washington")
plt.xlabel("year relative to policy")
plt.ylabel("Deaths due to drug overdose per 100k")
plt.plot(X_b41, y_pred_b41, color="k", label="comp before")
plt.plot(X_after1, y_pred_after1, color="k", label="comp after")
plt.plot(X_b4, y_pred_b4, color="b", label="washington before")
plt.plot(X_after, y_pred_after, color="b", label="washington after")
plt.legend()
plt.show()

## Helper functions

# 1) vertical_line()
#     - takes in a year and plots a vertical line at that year
# 2) get_charts()
#     - takes in two dataframes (one for before policy, one for after)
#     - returns chart for each
#     - not much utility by itself - used as a parameter for our get_fit() function


def vertical_line(year):
    """Function to plot a vertical line at year of policy implementation"""
    line = (
        alt.Chart(pd.DataFrame({"Date": [year], "color": ["black"]}))
        .mark_rule()
        .encode(
            x="Date:Q",  # use q for "quantitative" - as per altair docs
            color=alt.Color("color:N", scale=None),
        )
    )

    return line


def get_charts(b4, after, title_b4, title_after):
    """
    Function to plot the pre and post charts.
    Will not use in final plot - used as a baseline for our fit charts later.

    """

    base_before = (
        alt.Chart(b4)
        .mark_point()
        .encode(
            y=alt.Y("overdose_per_100k", scale=alt.Scale(zero=False)),
            x=alt.X("year relative to policy", scale=alt.Scale(zero=False)),
        )
        .properties(title=title_b4)
    )

    base_after = (
        alt.Chart(after)
        .mark_point()
        .encode(
            y=alt.Y("overdose_per_100k", scale=alt.Scale(zero=False)),
            x=alt.X("year relative to policy", scale=alt.Scale(zero=False)),
        )
        .properties(title=title_after)
    )

    return base_before, base_after


base_before, base_after = get_charts(
    b4=wa_b4,
    after=wa_after,
    title_b4="deaths before policy",
    title_after="deaths after policy",
)
base_before + base_after


def get_fits(chart_b4, chart_after):
    """
    Function to plot the regression lines for the pre and post charts.
    """

    fit_wa_b4 = (
        chart_b4.transform_regression("year relative to policy", "overdose_per_100k")
        .mark_line()
        .encode(color=alt.value("red"))
    )

    fit_wa_after = (
        chart_after.transform_regression("year relative to policy", "overdose_per_100k")
        .mark_line()
        .encode(color=alt.value("red"))
    )

    return fit_wa_b4, fit_wa_after


band_b4 = (
    alt.Chart(wa_b4)
    .mark_errorband(extent="ci")
    .encode(x=alt.X("Year"), y=alt.Y("overdose_per_100k"))
)


band_after = (
    alt.Chart(wa_after)
    .mark_errorband(extent="ci")
    .encode(x=alt.X("Year"), y=alt.Y("overdose_per_100k"))
)

# fit_b4 + fit_after

error_bars = base_before.mark_rule().encode(
    x="ci0(overdose_per_100k):Q",
    x2="ci1(overdose_per_100k):Q",
)
