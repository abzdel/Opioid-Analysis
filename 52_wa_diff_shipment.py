# importing the required libraries
import pandas as pd
import numpy as np
import altair as alt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# loading data for drug prescriptions in Florida and neighbouring states
prescriptions = pd.read_csv("05_cleaned_data/prescriptions_wa.csv", low_memory=False)

# reducing the number of columns in the drug prescriptions dataset by including only the relevant attributes to create a new dataset

prescriptions_reduced = prescriptions[
    [
        "DRUG_CODE",
        "DRUG_NAME",
        "QUANTITY",
        "UNIT",
        "STRENGTH",
        "CALC_BASE_WT_IN_GM",
        "DOSAGE_UNIT",
        "Product_Name",
        "Ingredient_Name",
        "Measure",
        "MME_Conversion_Factor",
        "dos_str",
        "Year",
        "Month",
        "StateFIPS",
        "StateName",
        "CountyFIPS",
        "state_abbrev",
        "FIP_unique",
        "Population",
        "county_test",
    ]
]

# creating a copy of reduced dataset of prescriptions and converting some of the attributes to appropriate data type

prescriptions_reduced_copy = prescriptions_reduced.copy()

prescriptions_reduced_copy["Year"] = prescriptions_reduced_copy["Year"].astype("int64")
prescriptions_reduced_copy["DRUG_CODE"] = prescriptions_reduced_copy[
    "DRUG_CODE"
].astype("int64")
prescriptions_reduced_copy["Month"] = prescriptions_reduced_copy["Month"].astype(
    "int64"
)
prescriptions_reduced_copy["Population"] = prescriptions_reduced_copy[
    "Population"
].astype("int64")

# creating a dataset that has all the drug prescriptions in the state of Florida

washington_prescriptions = prescriptions_reduced_copy[
    prescriptions_reduced_copy["StateName"] == "Washington"
]
# washington_prescriptions_copy = washington_prescriptions.copy()
washington_prescriptions["shipment_per_100k"] = (
    (
        washington_prescriptions["dos_str"]
        * washington_prescriptions["DOSAGE_UNIT"]
        * washington_prescriptions["MME_Conversion_Factor"]
    )
    / (washington_prescriptions["Population"])
    * 100000
)

washington_prescriptions

# calculating and displaying total number of drug prescriptions that took place in washington. results are grouped and displayed year and county wise

washington_prescriptions_result = (
    washington_prescriptions.groupby(["Year", "county_test"])["shipment_per_100k"]
    .sum()
    .reset_index()
)
# washington_prescriptions_result = washington_prescriptions.groupby(["Year"])["shipment_per_100k"].sum().reset_index()

washington_prescriptions_result.head()

## Diff-in-Diff

other_states_shipment = prescriptions_reduced_copy[
    (prescriptions_reduced_copy["StateName"] != "Washington")
]
# other_states_shipment = prescriptions_reduced_copy[(prescriptions_reduced_copy["StateName"] == "Alabama")|(prescriptions_reduced_copy["StateName"] == "Georgia")|(prescriptions_reduced_copy["StateName"] == "Mississippi")|(prescriptions_reduced_copy["StateName"] == "South Carolina")|(prescriptions_reduced_copy["StateName"] == "Tennessee")]

other_states_shipment

other_states_shipment_copy = other_states_shipment.copy()

# other_states_shipment_copy["shipment_per_100k"] = (other_states_shipment_copy["QUANTITY"] / other_states_shipment_copy["Population"]) * 100_000
other_states_shipment_copy["shipment_per_100k"] = (
    (
        other_states_shipment_copy["dos_str"]
        * other_states_shipment_copy["DOSAGE_UNIT"]
        * other_states_shipment_copy["MME_Conversion_Factor"]
    )
    / (other_states_shipment_copy["Population"])
    * 100000
)

other_states_shipment_copy

# calculating and displaying results for drug related deaths - year, state and county wise

other_states_result = (
    other_states_shipment_copy.groupby(["Year", "StateName", "county_test"])[
        "shipment_per_100k"
    ]
    .sum()
    .reset_index()
)
other_states_result.head()

washington_summary = pd.DataFrame(
    washington_prescriptions_result.describe()["shipment_per_100k"]
).rename(columns={"shipment_per_100k": "Opioid Shipment per 100k Residents - Texas"})
other_states_summary = pd.DataFrame(
    other_states_result.describe()["shipment_per_100k"]
).rename(
    columns={"shipment_per_100k": "Opioid Shipment per 100k Residents - Comp States"}
)
stats = pd.concat([washington_summary, other_states_summary], axis=1)
stats
# calculating yearwise drug related deaths results for Oregon
Oregon_results = (
    other_states_shipment_copy[other_states_shipment_copy["StateName"] == "Oregon"]
    .groupby("Year")["shipment_per_100k"]
    .mean()
)
Oregon_results
# calculating yearwise drug related deaths results for Idaho
Idaho_results = (
    other_states_shipment_copy[other_states_shipment_copy["StateName"] == "Idaho"]
    .groupby("Year")["shipment_per_100k"]
    .mean()
)
Idaho_results
# calculating yearwise drug related deaths results for Montana
montana_results = (
    other_states_shipment_copy[other_states_shipment_copy["StateName"] == "Montana"]
    .groupby("Year")["shipment_per_100k"]
    .mean()
)
montana_results
# calculating yearwise drug related deaths results for Nevada
nevada_results = (
    other_states_shipment_copy[other_states_shipment_copy["StateName"] == "Nevada"]
    .groupby("Year")["shipment_per_100k"]
    .mean()
)
nevada_results

wa_result = (
    washington_prescriptions_result.groupby("Year")["shipment_per_100k"]
    .mean()
    .reset_index()
)
comp_result = (
    other_states_result.groupby(["Year"])["shipment_per_100k"].mean().reset_index()
)
comp_result
wa_result

florida_summary = pd.DataFrame(wa_result.describe()["shipment_per_100k"]).rename(
    columns={"shipment_per_100k": "Opioid Shipment per 100k Residents - Washington"}
)
comp_summary = pd.DataFrame(comp_result.describe()["shipment_per_100k"]).rename(
    columns={"shipment_per_100k": "Opioid Shipment per 100k Residents - Control States"}
)
stats = pd.concat([florida_summary, comp_summary], axis=1)
stats
# create a scale for number of years before and after 2012 (target year)


def scale_years(year):
    if year == 2009:
        return -2
    if year == 2010:
        return -1
    if year == 2011:
        return 0
    if year == 2012:
        return 1


wa_result["year relative to policy"] = wa_result["Year"].apply(lambda x: scale_years(x))
comp_result["year relative to policy"] = comp_result["Year"].apply(
    lambda x: scale_years(x)
)

# double check no nulls in "year relative to policy"

assert wa_result["year relative to policy"].isnull().sum() == 0
assert comp_result["year relative to policy"].isnull().sum() == 0

# graphing
# split into before 2011 and after 2011

wa_b4 = wa_result[wa_result["Year"] <= 2011]
wa_after = wa_result[wa_result["Year"] >= 2011]


# wa_after = wa_after[wa_after["Year"] != 2011] # may need to handle this differently
from sklearn.linear_model import LinearRegression

regressor_b4 = LinearRegression()
regressor_after = LinearRegression()


X_b4 = np.array(wa_b4["year relative to policy"]).reshape(-1, 1)
y_b4 = np.array(wa_b4["shipment_per_100k"]).reshape(-1, 1)

X_after = np.array(wa_after["year relative to policy"]).reshape(-1, 1)
y_after = np.array(wa_after["shipment_per_100k"]).reshape(-1, 1)


regressor_b4.fit(X_b4, y_b4)
regressor_after.fit(X_after, y_after)


y_pred_b4 = regressor_b4.predict(X_b4)
y_pred_after = regressor_after.predict(X_after)

comp_b4 = comp_result[comp_result["Year"] <= 2011]
comp_after = comp_result[comp_result["Year"] >= 2011]

# comp_after = comp_after[comp_after["Year"] != 2011] # may need to handle this differently

regressor_b41 = LinearRegression()
regressor_after1 = LinearRegression()


X_b41 = np.array(comp_b4["year relative to policy"]).reshape(-1, 1)
y_b41 = np.array(comp_b4["shipment_per_100k"]).reshape(-1, 1)

X_after1 = np.array(comp_after["year relative to policy"]).reshape(-1, 1)
y_after1 = np.array(comp_after["shipment_per_100k"]).reshape(-1, 1)


regressor_b41.fit(X_b41, y_b41)
regressor_after1.fit(X_after1, y_after1)


y_pred_b41 = regressor_b41.predict(X_b41)
y_pred_after1 = regressor_after1.predict(X_after)

plt.xlim(-3, 3)
# plt.ylim(0, 500)
plt.title("Opioid Shipment before and after policy implementation in Washington")
plt.xlabel("year relative to policy")
plt.ylabel("Opioid Shipment per 100k")
plt.plot(X_b41, y_pred_b41, color="k", label="comp before")
plt.plot(X_after1, y_pred_after1, color="k", label="comp after")
plt.plot(X_b4, y_pred_b4, color="b", label="Washington before")
plt.plot(X_after, y_pred_after, color="b", label="Washington after")
plt.legend()
plt.show()
