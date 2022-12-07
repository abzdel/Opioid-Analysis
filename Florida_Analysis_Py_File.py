# Florida Data Analysis - Opioid Shipment And Overdosage Related Mortality

## Pre-Post Analysis Of Opioid Shipments In Florida

# importing the required libraries
import pandas as pd
import numpy as np
import altair as alt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# loading data for drug prescriptions in Florida and neighbouring states
prescriptions = pd.read_csv(
    r"C:\Users\annap\OneDrive\Desktop\Opioid Project\Prescriptions.csv",
    low_memory=False,
)
prescriptions.columns

prescriptions.head(3)

prescriptions.columns
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

prescriptions_reduced.head(5)

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

prescriptions_reduced_copy.rename(
    columns={
        "DRUG_CODE": "Drug Code",
        "DRUG_NAME": "Drug Name",
        "QUANTITY": "Quantity",
        "UNIT": "Unit",
        "STRENGTH": "Strength",
        "CALC_BASE_WT_IN_GM": "Calc Base Weight (In Gm)",
        "DOSAGE_UNIT": "Dosage Unit",
        "dos_str": "Dosage Strength",
        "StateFIPS": "State FIPS",
        "StateName": "State",
        "CountyFIPS": "County FIPS",
        "state_abbrev": "State Abbreviation",
        "FIP_unique": "FIPS_Unique",
        "county_test": "County",
    },
    inplace=True,
)

prescriptions_reduced_copy.head(3)

# creating a dataset that has all the drug prescriptions in the state of Florida

florida_prescriptions = prescriptions_reduced_copy[
    prescriptions_reduced_copy["State"] == "Florida"
]

florida_prescriptions.head()
florida_prescriptions_copy = florida_prescriptions.copy()

florida_prescriptions_copy["Opioid_Shipment_Per_100K"] = (
    (
        florida_prescriptions_copy["Dosage Strength"]
        * florida_prescriptions_copy["Dosage Unit"]
        * florida_prescriptions_copy["MME_Conversion_Factor"]
    )
    / (florida_prescriptions_copy["Population"])
    * 100000
)

florida_prescriptions_copy.head(3)
florida_opioid_result = (
    florida_prescriptions_copy.groupby(["Year", "County"])["Opioid_Shipment_Per_100K"]
    .sum()
    .reset_index()
)
florida_opioid_result.head()
florida_opioid_result_yearwise = (
    florida_opioid_result.groupby("Year")["Opioid_Shipment_Per_100K"]
    .sum()
    .reset_index()
)
florida_opioid_result_yearwise.head()
florida_opioid_result_yearwise_copy = florida_opioid_result_yearwise.copy()


florida_opioid_before = florida_opioid_result_yearwise_copy[
    florida_opioid_result_yearwise_copy["Year"] < 2010
]
florida_opioid_after = florida_opioid_result_yearwise_copy[
    florida_opioid_result_yearwise_copy["Year"] > 2010
]
# split into before 2010 and after 2010


from sklearn.linear_model import LinearRegression

regressor_opioid_before = LinearRegression()
regressor_opioid_after = LinearRegression()


x_opioid_before = np.array(florida_opioid_before["Year"]).reshape(-1, 1)
y_opioid_before = np.array(florida_opioid_before["Opioid_Shipment_Per_100K"]).reshape(
    -1, 1
)

x_opioid_after = np.array(florida_opioid_after["Year"]).reshape(-1, 1)
y_opioid_after = np.array(florida_opioid_after["Opioid_Shipment_Per_100K"]).reshape(
    -1, 1
)


regressor_opioid_before.fit(x_opioid_before, y_opioid_before)
regressor_opioid_after.fit(x_opioid_after, y_opioid_after)


y_pred_opioid_before = regressor_opioid_before.predict(x_opioid_before)
y_pred_opioid_after = regressor_opioid_after.predict(x_opioid_after)
import altair as alt

base_before = (
    alt.Chart(florida_opioid_before, title="FLORIDA DEATHS")
    .mark_point()
    .encode(
        x=alt.X("Year", title="Year", scale=alt.Scale(zero=False)),
        y=alt.Y(
            "Opioid_Shipment_Per_100K",
            title="Opioid Shipments",
            scale=alt.Scale(zero=False),
        ),
    )
)


fit_before = base_before.transform_regression(
    "Year", "Opioid_Shipment_Per_100K"
).mark_line(color="red")

base_after = (
    alt.Chart(florida_opioid_after, title="FLORIDA DEATHS")
    .mark_point()
    .encode(
        x=alt.X("Year", title="Year", scale=alt.Scale(zero=False)),
        y=alt.Y(
            "Opioid_Shipment_Per_100K",
            title="Opioid Shipments",
            scale=alt.Scale(zero=False),
        ),
    )
)


fit_after = (
    base_after.transform_regression("Year", "Opioid_Shipment_Per_100K")
    .mark_errorband(extent="ci")
    .mark_line(color="blue")
)

base_before + fit_before + base_after + fit_after
import altair as alt
from vega_datasets import data

source = data.cars()

line = alt.Chart(source).mark_line().encode(x="Year", y="mean(Miles_per_Gallon)")

band = (
    alt.Chart(source)
    .mark_errorband(extent="ci")
    .encode(
        x="Year",
        y=alt.Y("Miles_per_Gallon", title="Miles/Gallon"),
    )
)

band + line
import matplotlib.pyplot as plt

plt.xlim(-3, 3)
# plt.ylim(0, 500)

plt.xlabel("Year Relative To Policy Change Year (2010)")
plt.ylabel("Opioid Shipment (Per 100K)")
plt.title("Pre-Post Analysis Of Opioid Shipment In Florida")

plt.plot(x_opioid_before, y_pred_opioid_before, color="blue")
plt.plot(x_opioid_after, y_pred_opioid_after, color="red")

plt.grid()
## Pre-Post Analysis Of Drug Overdose Deaths In Florida

# loading data on deaths in Florida and its neighbouring states
deaths = pd.read_csv(r"C:\Users\annap\OneDrive\Desktop\Opioid Project\Deaths.csv")
deaths.head(3)

deaths.columns
# creating a copy of deaths dataset, to prevent SettingWithCopy warnings. The irrelevant columns are dropped.

deaths_copy = deaths.copy()
deaths_copy.drop(
    [
        "County Code",
        "Year Code",
        "Drug/Alcohol Induced Cause Code",
        "State",
        "CountyName",
        "StateAbbr",
        "STATE_COUNTY",
        "county_test",
    ],
    axis=1,
    inplace=True,
)
deaths_copy.head(3)
# converting some of the columns to more appropriate data type - integer data type
deaths_copy["Year"] = deaths_copy["Year"].astype("int64")
deaths_copy["Deaths"] = deaths_copy["Deaths"].astype("int64")
deaths_copy["Population"] = deaths_copy["Population"].astype("int64")


# renaming some of the columns to appear more conventional
deaths_copy.rename(
    columns={
        "StateFIPS": "State FIPS",
        "CountyFIPS": "County FIPS",
        "StateName": "State",
        "state_abbrev": "State Abbreviation",
        "FIP_unique": "FIPS_Unique",
    },
    inplace=True,
)

deaths_copy.head(3)
deaths_copy["State"].unique()

deaths_copy["Drug/Alcohol Induced Cause"].unique()
# creating the dataset for Florida which has all deaths that are drug related

florida_deaths = deaths_copy[
    (deaths_copy["State"] == "Florida")
    & (
        (
            deaths_copy["Drug/Alcohol Induced Cause"]
            == "Drug poisonings (overdose) Unintentional (X40-X44)"
        )
        | (
            deaths_copy["Drug/Alcohol Induced Cause"]
            == "Drug poisonings (overdose) Undetermined (Y10-Y14)"
        )
        | (
            deaths_copy["Drug/Alcohol Induced Cause"]
            == "Drug poisonings (overdose) Suicide (X60-X64)"
        )
    )
]

florida_deaths.head(5)
florida_deaths_copy = florida_deaths.copy()
# calculating the drug related death rates in Florida per 100000 people

florida_deaths_copy["Overdose_Per_100K"] = (
    florida_deaths_copy["Deaths"] / florida_deaths_copy["Population"]
) * 100_000

florida_deaths_copy.head(3)
# calculating and displaying year and county wise results for drug related deaths in Florida

florida_result = (
    florida_deaths_copy.groupby(["Year", "County"])["Overdose_Per_100K"]
    .sum()
    .reset_index()
)
florida_result.head(5)
# calculating and displaying yearwise results for drug related deaths in Florida

florida_results_yearwise = florida_deaths_copy.groupby("Year")[
    "Overdose_Per_100K"
].mean()
florida_results_yearwise

florida_result_copy = florida_result.copy()

# create a scale for number of years before and after 2010 (target year)
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
    if year == 2013:
        return 3


florida_result_copy["Year Relative To Policy"] = florida_result_copy["Year"].apply(
    lambda x: scale_years(x)
)

# double check no nulls in "year relative to policy"

assert florida_result_copy["Year Relative To Policy"].isnull().sum() == 0

# doing this in case the float == int comparison causes issues

# split into before 2010 and after 2010

florida_mortality_before_policy_change = florida_result_copy[
    florida_result_copy["Year"] < 2010
]
florida_mortality_after_policy_change = florida_result_copy[
    florida_result_copy["Year"] > 2010
]

from sklearn.linear_model import LinearRegression

regressor_before = LinearRegression()
regressor_after = LinearRegression()


x_before = np.array(
    florida_mortality_before_policy_change["Year Relative To Policy"]
).reshape(-1, 1)
y_before = np.array(
    florida_mortality_before_policy_change["Overdose_Per_100K"]
).reshape(-1, 1)

x_after = np.array(
    florida_mortality_after_policy_change["Year Relative To Policy"]
).reshape(-1, 1)
y_after = np.array(florida_mortality_after_policy_change["Overdose_Per_100K"]).reshape(
    -1, 1
)


regressor_before.fit(x_before, y_before)
regressor_after.fit(x_after, y_after)


y_pred_before = regressor_before.predict(x_before)
y_pred_after = regressor_after.predict(x_after)

import matplotlib.pyplot as plt

plt.xlim(-3, 3)
# plt.ylim(0, 500)

plt.xlabel("Year Relative To Policy Change Year (2010)")
plt.ylabel("Deaths due to Drug Overdose (Per 100K)")
plt.title("Pre-Post Analysis Of Drug Overdose Deaths in Florida")

plt.plot(x_before, y_pred_before, color="blue")
plt.plot(x_after, y_pred_after, color="red")

plt.grid()
# plot avg value in each year

## Difference-In-Difference Analysis Of Drug Overdose Deaths In Florida

# creating a new dataset, that contains drug related deaths in other neighboring states of Florida - Alabama, Georgia, Mississippi, South Carolin and Tennessee

other_states_deaths = deaths_copy[
    (deaths_copy["State"] != "Florida")
    & (
        (
            deaths_copy["Drug/Alcohol Induced Cause"]
            == "Drug poisonings (overdose) Unintentional (X40-X44)"
        )
        | (
            deaths_copy["Drug/Alcohol Induced Cause"]
            == "Drug poisonings (overdose) Undetermined (Y10-Y14)"
        )
        | (
            deaths_copy["Drug/Alcohol Induced Cause"]
            == "Drug poisonings (overdose) Suicide (X60-X64)"
        )
    )
]
other_states_deaths_copy = other_states_deaths.copy()

other_states_deaths_copy["Overdose_Per_100K"] = (
    other_states_deaths_copy["Deaths"] / other_states_deaths_copy["Population"]
) * 100_000
# calculating and displaying results for drug related deaths - year, state and county wise

other_states_result = (
    other_states_deaths_copy.groupby(["Year", "State", "County"])["Overdose_Per_100K"]
    .sum()
    .reset_index()
)
other_states_result.head()
# calculating yearwise drug related deaths results for Alabama
alabama_results_yearwise = (
    other_states_deaths_copy[other_states_deaths_copy["State"] == "Alabama"]
    .groupby("Year")["Overdose_Per_100K"]
    .mean()
)
alabama_results_yearwise
# calculating yearwise drug related deaths results for Georgia
georgia_results_yearwise = (
    other_states_deaths_copy[other_states_deaths_copy["State"] == "Georgia"]
    .groupby("Year")["Overdose_Per_100K"]
    .mean()
)
georgia_results_yearwise
# calculating yearwise drug related deaths results for Mississippi
mississippi_results_yearwise = (
    other_states_deaths_copy[other_states_deaths_copy["State"] == "Mississippi"]
    .groupby("Year")["Overdose_Per_100K"]
    .mean()
)
mississippi_results_yearwise
# calculating yearwise drug related deaths results for South Carolina
south_carolina_results_yearwise = (
    other_states_deaths_copy[other_states_deaths_copy["State"] == "South Carolina"]
    .groupby("Year")["Overdose_Per_100K"]
    .mean()
)
south_carolina_results_yearwise
# calculating yearwise drug related deaths results for Tennessee
tennessee_results_yearwise = (
    other_states_deaths_copy[other_states_deaths_copy["State"] == "Tennessee"]
    .groupby("Year")["Overdose_Per_100K"]
    .mean()
)
tennessee_results_yearwise
# calculating and displaying yearwise results for drug related deaths in neighboring states around Florida (combined)

other_states_results_combined_yearwise = other_states_deaths_copy.groupby("Year")[
    "Overdose_Per_100K"
].mean()
other_states_results_combined_yearwise

# these yearwise results have been calculated while doing the pre-post analysis of drug overdose deaths in Florida
florida_results_yearwise
florida_summary = pd.DataFrame(florida_result.describe()["Overdose_Per_100K"]).rename(
    columns={"Overdose_Per_100K": "Overdoses per 100k Residents - Florida"}
)
other_summary = pd.DataFrame(
    other_states_result.describe()["Overdose_Per_100K"]
).rename(columns={"Overdose_Per_100K": "Overdoses Per 100K Residents - Control States"})
summary_statistics = pd.concat([florida_summary, other_summary], axis=1)
summary_statistics
other_states_result_copy = other_states_result.copy()

other_states_result_copy["Year Relative To Policy"] = other_states_result_copy[
    "Year"
].apply(lambda x: scale_years(x))

# double check no nulls in "year relative to policy"

assert other_states_result_copy["Year Relative To Policy"].isnull().sum() == 0

# doing this in case the float == int comparison causes issues

other_states_before = other_states_result_copy[other_states_result_copy["Year"] < 2010]
other_states_after = other_states_result_copy[other_states_result_copy["Year"] > 2010]

regressor_other_states_before = LinearRegression()
regressor_other_states_after = LinearRegression()


x_other_before = np.array(other_states_before["Year Relative To Policy"]).reshape(-1, 1)
y_other_before = np.array(other_states_before["Overdose_Per_100K"]).reshape(-1, 1)

x_other_after = np.array(other_states_after["Year Relative To Policy"]).reshape(-1, 1)
y_other_after = np.array(other_states_after["Overdose_Per_100K"]).reshape(-1, 1)


regressor_other_states_before.fit(x_other_before, y_other_before)
regressor_other_states_after.fit(x_other_after, y_other_after)


y_pred_other_before = regressor_other_states_before.predict(x_other_before)
y_pred_other_after = regressor_other_states_after.predict(x_other_after)
plt.xlim(-3, 3)
# plt.ylim(0, 500)

plt.xlabel("Year Relative To Policy Change Year (2010)")
plt.ylabel("Deaths due to Drug Overdose (Per 100K)")
plt.title("Difference in Difference Analysis of Drug Overdose Deaths In Florida")

plt.plot(x_other_before, y_pred_other_before, color="green", label="Other States")
plt.plot(x_other_after, y_pred_other_after, color="green", label="Other States")

plt.plot(x_before, y_pred_before, color="orange", label="Florida")
plt.plot(x_after, y_pred_after, color="orange", label="Florida")

plt.grid()
## Still trying to figure out error bands

####Trying to add more.
