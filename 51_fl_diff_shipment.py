# importing the required libraries
import pandas as pd
import numpy as np
import altair as alt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# loading data for drug prescriptions in Florida and neighbouring states
prescriptions = pd.read_csv("05_cleaned_data/prescriptions_fl.csv", low_memory=False)
prescriptions

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

florida_prescriptions = prescriptions_reduced_copy[
    prescriptions_reduced_copy["StateName"] == "Florida"
]
# florida_prescriptions["shipment_per_100k"] = (florida_prescriptions["QUANTITY"] / florida_prescriptions["Population"]) * 100_000

florida_prescriptions["shipment_per_100k"] = (
    (
        florida_prescriptions["dos_str"]
        * florida_prescriptions["DOSAGE_UNIT"]
        * florida_prescriptions["MME_Conversion_Factor"]
    )
    / (florida_prescriptions["Population"])
    * 100000
)
florida_prescriptions
# calculating and displaying total number of drug prescriptions that took place in Florida. results are grouped and displayed year and county wise

florida_prescriptions_result = (
    florida_prescriptions.groupby(["Year", "county_test"])["shipment_per_100k"]
    .sum()
    .reset_index()
)
# florida_prescriptions_result = florida_prescriptions.groupby(["Year"])["shipment_per_100k"].sum().reset_index()

florida_prescriptions_result

# creating a new dataset, that contains drug related deaths in other neighboring states of Florida - Alabama, Georgia, Mississippi, South Carolina and Tennessee

other_states_shipment = prescriptions_reduced_copy[
    (prescriptions_reduced_copy["StateName"] != "Florida")
]
# other_states_shipment = prescriptions_reduced_copy[(prescriptions_reduced_copy["StateName"] == "Alabama")|(prescriptions_reduced_copy["StateName"] == "Georgia")|(prescriptions_reduced_copy["StateName"] == "Mississippi")|(prescriptions_reduced_copy["StateName"] == "South Carolina")|(prescriptions_reduced_copy["StateName"] == "Tennessee")]

other_states_shipment

other_states_shipment_copy = other_states_shipment.copy()

other_states_shipment_copy["shipment_per_100k"] = (
    other_states_shipment_copy["QUANTITY"] / other_states_shipment_copy["Population"]
) * 100_000
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
# calculating yearwise drug related deaths results for Alabama
alabama_results = (
    other_states_shipment_copy[other_states_shipment_copy["StateName"] == "Alabama"]
    .groupby("Year")["shipment_per_100k"]
    .mean()
)
alabama_results
# calculating yearwise drug related deaths results for Georgia
georgia_results = (
    other_states_shipment_copy[other_states_shipment_copy["StateName"] == "Georgia"]
    .groupby("Year")["shipment_per_100k"]
    .mean()
)
georgia_results
# calculating yearwise drug related deaths results for Mississippi
mississippi_results = (
    other_states_shipment_copy[other_states_shipment_copy["StateName"] == "Mississippi"]
    .groupby("Year")["shipment_per_100k"]
    .mean()
)
mississippi_results
# calculating yearwise drug related deaths results for South Carolina
south_carolina_results = (
    other_states_shipment_copy[
        other_states_shipment_copy["StateName"] == "South Carolina"
    ]
    .groupby("Year")["shipment_per_100k"]
    .mean()
)
south_carolina_results
# calculating yearwise drug related deaths results for Tennessee
tennessee_results = (
    other_states_shipment_copy[other_states_shipment_copy["StateName"] == "Tennessee"]
    .groupby("Year")["shipment_per_100k"]
    .mean()
)
tennessee_results
fl_result = (
    florida_prescriptions_result.groupby("Year")["shipment_per_100k"]
    .mean()
    .reset_index()
)
comp_result = (
    other_states_result.groupby(["Year"])["shipment_per_100k"].mean().reset_index()
)
florida_summary = pd.DataFrame(fl_result.describe()["shipment_per_100k"]).rename(
    columns={"shipment_per_100k": "Opioid Shipment per 100k Residents - Florida"}
)
comp_summary = pd.DataFrame(comp_result.describe()["shipment_per_100k"]).rename(
    columns={"shipment_per_100k": "Opioid Shipment per 100k Residents - Control States"}
)
stats = pd.concat([florida_summary, comp_summary], axis=1)
# create a scale for number of years before and after 2012 (target year)


def scale_years(year):
    if year == 2007:
        return -3
    if year == 2008:
        return -2
    if year == 2009:
        return -1
    if year == 2010:
        return 0
    if year == 2011:
        return 1
    if year == 2012:
        return 2


fl_result["year relative to policy"] = fl_result["Year"].apply(lambda x: scale_years(x))
comp_result["year relative to policy"] = comp_result["Year"].apply(
    lambda x: scale_years(x)
)

# double check no nulls in "year relative to policy"

assert fl_result["year relative to policy"].isnull().sum() == 0
assert comp_result["year relative to policy"].isnull().sum() == 0
# split into before 2007 and after 2007

fl_b4 = fl_result[fl_result["Year"] < 2010]
fl_after = fl_result[fl_result["Year"] >= 2010]


fl_after = fl_after[fl_after["Year"] != 2010]  # may need to handle this differently
from sklearn.linear_model import LinearRegression

regressor_b4 = LinearRegression()
regressor_after = LinearRegression()


X_b4 = np.array(fl_b4["year relative to policy"]).reshape(-1, 1)
y_b4 = np.array(fl_b4["shipment_per_100k"]).reshape(-1, 1)

X_after = np.array(fl_after["year relative to policy"]).reshape(-1, 1)
y_after = np.array(fl_after["shipment_per_100k"]).reshape(-1, 1)


regressor_b4.fit(X_b4, y_b4)
regressor_after.fit(X_after, y_after)


y_pred_b4 = regressor_b4.predict(X_b4)
y_pred_after = regressor_after.predict(X_after)

comp_b4 = comp_result[comp_result["Year"] < 2010]
comp_after = comp_result[comp_result["Year"] >= 2010]

comp_after = comp_after[
    comp_after["Year"] != 2010
]  # may need to handle this differently

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
plt.title("Opioid Shipment before and after policy implementation in Florida")
plt.xlabel("year relative to policy")
plt.ylabel("Opioid Shipment per 100k")
plt.plot(X_b41, y_pred_b41, color="k", label="comp before")
plt.plot(X_after1, y_pred_after1, color="k", label="comp after")
plt.plot(X_b4, y_pred_b4, color="b", label="Florida before")
plt.plot(X_after, y_pred_after, color="b", label="Florida after")
plt.legend()
plt.show()
