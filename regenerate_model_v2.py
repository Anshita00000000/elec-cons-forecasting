#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd

# ------------------------------------------------------------
# 🔑 ENTER YOUR EIA API KEY HERE
# ------------------------------------------------------------
API_KEY = "vU3wYhrfqLBd4nWvEsm6i3XCZXJay1fVXpbcfzSm"

# ------------------------------------------------------------
# 🔗 BASE URL (exactly as shown in your screenshot)
# ------------------------------------------------------------
url = "https://api.eia.gov/v2/electricity/retail-sales/data/"

# ------------------------------------------------------------
# 📌 QUERY PARAMETERS (exact x-params shown in screenshot)
# ------------------------------------------------------------
params = {
    "frequency": "monthly",
    "data[0]": "customers",
    "data[1]": "price",
    "data[2]": "revenue",
    "data[3]": "sales",
    "facets[sectorid][]": "ALL",
    "facets[stateid][]": "US",
    "start": "2001-01",
    "end": "2025-08",
    "offset": 0,
    "length": 5000,
    "api_key": API_KEY
}

# ------------------------------------------------------------
# 📥 MAKE REQUEST
# ------------------------------------------------------------
response = requests.get(url, params=params)

# Check for errors
response.raise_for_status()

# ------------------------------------------------------------
# 📦 CONVERT RESPONSE TO DATAFRAME
# ------------------------------------------------------------
json_data = response.json()
electricity_consumption_df = pd.DataFrame(json_data["response"]["data"])

# ------------------------------------------------------------
# 📊 SHOW FIRST ROWS
# ------------------------------------------------------------
print("Fetched rows:", len(electricity_consumption_df))
electricity_consumption_df.head()
electricity_consumption_df.tail()


# In[2]:


#Million MMBTU energy Consumption for production of electricity for all sectors(99) for all over the country(USA) for all types of fuesls(All Fuels)


# In[3]:


# ------------------------------------------------------------
# 🔗 BASE URL (exactly as shown in your screenshot)
# ------------------------------------------------------------
#url = "https://api.eia.gov/v2/electricity/retail-sales/data/"
url = "https://api.eia.gov/v2/electricity/electric-power-operational-data/data/"
API_KEY = "vU3wYhrfqLBd4nWvEsm6i3XCZXJay1fVXpbcfzSm"
# ------------------------------------------------------------
# 📌 QUERY PARAMETERS (exact x-params shown in screenshot)
# ------------------------------------------------------------

params = {
    "frequency": "monthly",
    "data[0]": "consumption-for-eg-btu",    
    "facets[sectorid][]": 99,
    "facets[location][]": "US",
    "facets[fueltypeid][]":"ALL",
    "start": "2001-01",
    "end": "2025-08",
    "sort[0][column]": "period",
    "sort[0][direction]": "desc",
    "offset": 0,
    "length": 5000,
    "api_key": API_KEY
}

# ------------------------------------------------------------
# 📥 MAKE REQUEST
# ------------------------------------------------------------
response = requests.get(url, params=params)

# Check for errors
response.raise_for_status()

# ------------------------------------------------------------
# 📦 CONVERT RESPONSE TO DATAFRAME
# ------------------------------------------------------------
json_data = response.json()
electricity_energy_consum_df = pd.DataFrame(json_data["response"]["data"])

# ------------------------------------------------------------
# 📊 SHOW FIRST ROWS
# ------------------------------------------------------------
print("Fetched rows:", len(electricity_consumption_df))
electricity_energy_consum_df.head()
electricity_energy_consum_df.tail()


# In[4]:


#holidays count
#! pip install holidays


# In[5]:


import holidays
import pandas as pd

# ----------------- CONFIG -----------------
start = "2001-01"
end   = "2025-08"
years = range(2001, 2026)   # 2025 inclusive

# ----------------- HOLIDAYS -----------------
records = []
for year in years:
    for date, name in holidays.India(years=year).items():
        if start <= str(date) <= end:
            records.append([date, name])

hol_df = pd.DataFrame(records, columns=["date", "holiday_name"])
hol_df["date"] = pd.to_datetime(hol_df["date"])
hol_df["month"] = hol_df["date"].dt.to_period("M")

# public holidays per month
holiday_counts = (
    hol_df.groupby("month").size().reset_index(name="public_holidays")
)

# holidays that fall on Saturday/Sunday (for overlap correction)
hol_df["weekday"] = hol_df["date"].dt.dayofweek  # Mon=0 ... Sun=6
weekend_hols = hol_df[hol_df["weekday"] >= 5]    # 5=Sat, 6=Sun
overlap = (
    weekend_hols.groupby("month").size()
    .reset_index(name="holiday_on_weekend")
)

# ----------------- WEEKENDS -----------------
all_days = pd.date_range(start=start + "-01", end=end + "-28", freq="D")
days_df = pd.DataFrame({"date": all_days})
days_df["month"] = days_df["date"].dt.to_period("M")
days_df["weekday"] = days_df["date"].dt.dayofweek  # Mon=0 ... Sun=6

weekend_counts = (
    days_df.groupby("month")
    .agg(
        saturdays=("weekday", lambda x: (x == 5).sum()),
        sundays=("weekday",   lambda x: (x == 6).sum()),
    )
    .reset_index()
)
weekend_counts["weekend_days"] = (
    weekend_counts["saturdays"] + weekend_counts["sundays"]
)

# ----------------- FULL MONTH RANGE -----------------
full_range = pd.period_range(start=start, end=end, freq="M")
final_df = pd.DataFrame({"month": full_range})

# merge everything
final_df = (
    final_df
    .merge(holiday_counts, on="month", how="left")
    .merge(weekend_counts, on="month", how="left")
    .merge(overlap, on="month", how="left")
)

# fill NaNs
final_df[["public_holidays","saturdays","sundays",
          "weekend_days","holiday_on_weekend"]] = (
    final_df[["public_holidays","saturdays","sundays",
              "weekend_days","holiday_on_weekend"]].fillna(0).astype(int)
)

# total non-working days (no double counting)
final_df["total_off_days"] = (
    final_df["public_holidays"]
    + final_df["weekend_days"]
    - final_df["holiday_on_weekend"]
)

print(final_df)
print(len(final_df))   # should be 296


# In[6]:


# IPI data
# https://fred.stlouisfed.org/series/IPG2211S


# In[7]:


# 1. Read the IPI(Industrial Production index
df_ipi = pd.read_csv(
    "IPG2211S.csv",   # <-- your file path
    comment="#",         # ignore lines beginning with '#'
)

print(df_ipi)


# In[8]:


#weather data
#https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/tavg/1/0/2001-2025


# In[9]:


import pandas as pd

# 1. Read the NOAA CSV – skip comment lines
df = pd.read_csv(
    "Average Temperature.csv",   # <-- your file path
    comment="#",         # ignore lines beginning with '#'
)

# 2. Clean up columns
df.columns = df.columns.str.strip()     # just in case
df.rename(columns={"Value": "TAVG_F"}, inplace=True)

# 3. Convert Date (YYYYMM) to proper datetime
df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m")

# 4. Compute HDD and CDD (base 65°F)
df["HDD"] = (65 - df["TAVG_F"]).clip(lower=0)
df["CDD"] = (df["TAVG_F"] - 65).clip(lower=0)

# 5. Optional: month period column and final selection
df["Month"] = df["Date"].dt.to_period("M")
df_temp = df[["Month", "TAVG_F", "HDD", "CDD"]]

print(df_temp.head())
print(len(df_temp))


# In[10]:


# 1. Read the precipitation
df_precep = pd.read_csv(
    "Precipitation.csv",   # <-- your file path
    comment="#",         # ignore lines beginning with '#'
)

print(df_precep)


# In[11]:


import pandas as pd

# =======================
# Helpers
# =======================

def standardize_columns(df):
    """Lowercase, strip, replace spaces/hyphens with underscore."""
    out = df.copy()
    out.columns = (
        out.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )
    return out

def normalize_window(df, month_col, start="2001-01", end="2025-08"):
    """Restrict to our month window and drop duplicates per month."""
    start_p = pd.Period(start, freq="M")
    end_p   = pd.Period(end,   freq="M")
    out = df.copy()
    out = out[(out[month_col] >= start_p) & (out[month_col] <= end_p)]
    out = out.sort_values(month_col).drop_duplicates(subset=[month_col], keep="first")
    return out

# canonical month index
start_period = pd.Period("2001-01", freq="M")
end_period   = pd.Period("2025-08", freq="M")
month_index  = pd.period_range(start=start_period, end=end_period, freq="M")
master_df = pd.DataFrame({"month": month_index})


# =======================
# 1. Retail electricity data
#    electricity_consumption_df
# =======================
elec_retail_raw = standardize_columns(electricity_consumption_df)

# 'period' is like '2025-08'
elec_retail_raw["month"] = pd.PeriodIndex(elec_retail_raw["period"].astype(str), freq="M")
elec_retail = normalize_window(elec_retail_raw, "month")
print(elec_retail)
# After standardization, expected col names: customers, price, revenue, sales
elec_retail = elec_retail[["month", "customers", "price", "revenue", "sales"]].rename(
    columns={
        "customers": "customers",
        "price": "price_cents_per_kwh",
        "revenue": "revenue_musd",
        "sales": "sales_mkwh",
    }
)

master_df = master_df.merge(elec_retail, on="month", how="left", validate="1:1")


# =======================
# 2. Energy consumption (MMBtu)
#    electricity_energy_consum_df
# =======================
elec_energy_raw = standardize_columns(electricity_energy_consum_df)
elec_energy_raw["month"] = pd.PeriodIndex(elec_energy_raw["period"].astype(str), freq="M")
elec_energy = normalize_window(elec_energy_raw, "month")

# column becomes 'consumption_for_eg_btu' after standardization
elec_energy = elec_energy[["month", "consumption_for_eg_btu"]].rename(
    columns={"consumption_for_eg_btu": "energy_consumption_million_mmbtu"}
)

master_df = master_df.merge(elec_energy, on="month", how="left", validate="1:1")


# =======================
# 3. Holidays & weekends
#    final_df
# =======================
hol_raw = standardize_columns(final_df)
# month is already a Period, but make sure
hol_raw["month"] = hol_raw["month"].astype("period[M]")
holidays = normalize_window(hol_raw, "month")

holidays = holidays[
    [
        "month",
        "public_holidays",
        "saturdays",
        "sundays",
        "weekend_days",
        "holiday_on_weekend",
        "total_off_days",
    ]
]

master_df = master_df.merge(holidays, on="month", how="left", validate="1:1")


# =======================
# 4. Industrial Production Index (df_ipi)
# =======================
ipi_raw = standardize_columns(df_ipi)
ipi_raw["month"] = pd.to_datetime(ipi_raw["observation_date"]).dt.to_period("M")
ipi = normalize_window(ipi_raw, "month")

# ipg2211s -> industrial_production_index
ipi = ipi[["month", "ipg2211s"]].rename(
    columns={"ipg2211s": "industrial_production_index"}
)

master_df = master_df.merge(ipi, on="month", how="left", validate="1:1")


# =======================
# 5. Temperature + HDD + CDD (df_temp)
# =======================
temp_raw = standardize_columns(df_temp)
# df_temp had 'Month' -> now 'month'; ensure Period type
if not isinstance(temp_raw["month"].dtype, pd.PeriodDtype):
    temp_raw["month"] = pd.PeriodIndex(temp_raw["month"].astype(str), freq="M")

temp = normalize_window(temp_raw, "month")

temp = temp[["month", "tavg_f", "hdd", "cdd"]].rename(
    columns={
        "tavg_f": "temp_avg_f",
        "hdd": "heating_degree_days",
        "cdd": "cooling_degree_days",
    }
)

master_df = master_df.merge(temp, on="month", how="left", validate="1:1")


# =======================
# 6. Precipitation (df_precep)
# =======================
precip_raw = standardize_columns(df_precep)
# Date (YYYYMM) -> month Period
precip_raw["month"] = pd.to_datetime(precip_raw["date"].astype(str), format="%Y%m").dt.to_period(
    "M"
)
precip = normalize_window(precip_raw, "month")

precip = precip[["month", "value"]].rename(columns={"value": "precip_inches"})

master_df = master_df.merge(precip, on="month", how="left", validate="1:1")


# =======================
# 7. Final touches
# =======================
master_df["year"] = master_df["month"].dt.year
master_df["month_num"] = master_df["month"].dt.month
master_df = master_df.sort_values("month").reset_index(drop=True)

print(master_df.head())
print(master_df.tail())
print("Rows in master_df:", len(master_df))   # should be 296
master_df.to_csv("master_df.csv")


# In[12]:


# =========================
# 0. Imports & Settings
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True

# =========================
# 1. Load Data
# =========================
FILE_PATH = "master_df.csv"   # change path if needed

df = pd.read_csv(FILE_PATH)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
df.head()


# In[13]:


# =========================
# 2. Basic Cleaning
# =========================

# Drop index-like column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Convert month to datetime if it is not already
if "month" in df.columns:
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").reset_index(drop=True)

# Show info
df.info()


# In[14]:


# =========================
# 3. Column Summary
# =========================

n_rows = len(df)

summary = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing_count": df.isna().sum(),
    "missing_pct": (df.isna().sum() / n_rows * 100).round(2),
    "n_unique": df.nunique(dropna=True)
}).sort_index()

summary


# In[15]:


# =========================
# 4. Separate Numeric & Categorical
# =========================

num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)


# In[16]:


# =============================
# TARGET-FOCUSED EDA FOR sales_mkwh
# =============================

target = "sales_mkwh"

print("Target variable:", target)
df[target].describe()


# In[17]:


plt.figure(figsize=(6,4))
plt.boxplot(df['sales_mkwh'], vert=True)
plt.title("Boxplot of sales_mkwh")
plt.ylabel("sales_mkwh")
plt.show()


# ## Target Variable Analysis: Sales_mkwh
# 
# ### The target variable sales_mkwh represents the monthly electricity sales measured in million kWh. The descriptive statistics reveal the following characteristics
# 
# #### The dataset contains 296 monthly observations
# #### The mean sales over the period is 310767 MWH with a standard deviation of 32720MWH, indicating moderate month-to-month variability
# #### The minimum monthly sales recorded is 253033 MWh while the maximum is 407230MWh giving us a range of approximately 154000MWh
# #### The quartiles 25th = 286934, 50th = 305223, 75th = 329015 show that the distribution is right skewed with the upper quartile extending further from the median than the lower quartile
# #### This skewness is expected due to seasonal peaks from heating or cooling demand and long term growth
# 
# ### Box plot
# #### A boxplot confirms that no extreme outliers are present. The spread between the quartiles is moderate, and all observations lie within a reasonable distance from the whiskers. This suggests that the variable is clean, stable and suitable for modeling without requiring outlier correction
# 
# ### Overall sales_mkwh displays:
# #### Strong natural seasonality
# #### Gradual upward trend across years
# #### No data quality issues or implausible values

# In[18]:


# =========================
# 5. Numeric Summary
# =========================

num_desc = df[num_cols].describe().T
num_desc


# ### Core Business Metrics
# 
# #### Customers:
# #### - Available for 212 months (missing values exist from earlier years)
# #### - The Average number of customer is appx. 151.9 million with a narrow standard deviation indicating slow but steady growth
# #### - The range is relatively tight from 142.8M to 165.2M is consistent with a large stable consumer base
# 
# #### Revenue: (revenue_musd)
# #### - Mean monthly revenue: 31,220 million USD Appx.
# #### - Standard deviation: 7568 indicating moderate variability
# #### - Revenue ranges from 17753 to 58596 reflecting seasonal demand changes and price x volume effects
# 
# ### Energy Consumption & Weather Variables:
# 
# #### Energy Consumption:(MMBTU)
# #### - Mean 3,196 and standard devation is 383
# #### - Strong alignmnet between this variable and electricity sales is expected
# #### - The spread suggests natural seasonal variation
# 
# #### Temperature (temp_avg_f)
# #### - Average montly temperature: 53.8 F ranging from 29 to 77 F
# #### - Seasonal cycles are strongly represented
# 
# #### Heating Degree Days (HDD) & Cooling Degree Daya (CDD)
# #### - HDD mean 13.4, CDD mean: 21.4
# #### - Both show significant variance confirming seasonal drivers
# #### - These variables are essential for modeling electricity demand related to heating and air conditioning
# 
# #### Calender features:
# 
# #### Saturdays, Sundays, Holidays
# #### - All months contain 4-5 Saturdays and Sundays as expected
# #### - Weekend totals range from 8-10 days per month, reflecting calendar mechanics
# #### - Variation is minimal, indicating low predictive power
# 
# #### Public Holidays & Total Off Days
# #### - Average: 1.3 Holidays/month
# #### - Total monthly off-days range: 8-13, combining weekends + holidays
# #### - Holidays show low variability, suggesting theymah have a weak realtionship with electricity consumption
# 
# #### Economic Activity Indicator:
# 
# #### Industrial Production Index:
# #### - Mean: 100.05, SD: 4.45
# #### - Range: 82 to 112 reflecting economic cycles, recessions and recoveries
# #### - This variable is a macroeconomic driver and may correlate positively with target variable. 
# 
# #### Time Derived Features
# #### Year:
# #### - Ranges from 2001 to 2025 covering about 25 years of data
# 
# #### Month Number:
# #### - Ranges from 1-12 capturing monthly seasonal effects
# 
# #### These fields are useful for encoding
# #### - Seasonality
# #### - Trend
# #### - Cyclic Patterns
# 
# #### Summary of Insights:
# #### - Variables like HDD, CDD, temperature, customers and energy consumptioin show wide ranges, confirming their suitability as drivers of electricity sales.
# #### - Calender variables show limited variation, so their predictive values may be low
# #### - Economic index shows meaningful variance, indicating potential correlation with electricity consumption
# #### - Economic Index shows meaningful variance indicating potential correlation with electricity consumption
# #### - The target variable exhibits clear seasonality and moderate, variability, making it well-behaved for forecasting
# #### - No extreme values appear in the data, suggesting high data quality and readiness for modeling

# In[19]:


# =========================
# 6. Categorical Summary
# =========================

cat_summary_list = []

for col in cat_cols:
    vc = df[col].value_counts(dropna=False)
    top_val = vc.index[0]
    top_freq = vc.iloc[0]
    top_pct = round(top_freq / n_rows * 100, 2)

    cat_summary_list.append({
        "column": col,
        "n_unique": df[col].nunique(dropna=True),
        "top_value": top_val,
        "top_freq": top_freq,
        "top_pct": top_pct
    })

cat_summary = pd.DataFrame(cat_summary_list)
cat_summary


# In[20]:


# =========================
# 7. Histograms for Numeric Columns
# =========================

from math import ceil

cols_to_plot = num_cols
n = len(cols_to_plot)
if n > 0:
    rows = ceil(n / 2)
    plt.figure(figsize=(12, 4 * rows))

    for i, col in enumerate(cols_to_plot, 1):
        plt.subplot(rows, 2, i)
        df[col].hist(bins=30)
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# In[21]:


plt.figure(figsize=(8,5))
df[target].hist(bins=30)
plt.title("Distribution of sales_mkwh")
plt.xlabel("sales_mkwh")
plt.ylabel("Frequency")
plt.show()


# ## Univariate Analysis
# ### This section examines the distribution of each numerical variable to understand its natural range, variability, skewness and potential outliers. The histograms reveal important behavioral patterns across customer metrics, pricing, energy use, weather, calendar counts, economic indicators, and engineering time features.
# 
# ### Customer, Price, Revenue and Sales Variables
# 
# ### Customers:
# #### - The distriution spans roughly 142M to 165M customers, reflecting gradual lontg-term growth
# #### - The left side of the histogram shows higher concentration, indicating earlier years have lower customer counts
# #### - No extreme values appear, confirming clean and stable customer series
# 
# ### Price(price_cents_per_kwh)
# #### - The distribution clusters around 10-11 cents with some lower values (7-8 cents) in earlier years
# #### - A few higher price episodes (12-14 cents) appear, like representing tariff adjustments
# #### - Mild skewness suggests occasional price hikes but no volatility
# 
# ### Revenue (revenue_musd):
# #### - Revenue exhibits a right skewed distribution with most months between 25000 to 35000 million USD
# #### - Occossional months exceed 45000-58000 million uSD aligining with seasonal electricity spikes
# #### - No impossible spikes are observed. The pattern matches expected business cycles.
# 
# ### Electricity Sales(sales_mkwh) (Target Variable)
# #### - Sales cluster prominently around 290000-330000MWh
# #### - Upper tail rises smoothly to 400000MWh consistent with seasonal summer/winter peaks
# #### - No sharp discontinues or anomalies reinforcing the stability of the target variable
# 
# ### Energy consumption and Holiday Metrics
# 
# ### Energy Consumption(energy_consumption_million_mmbtu)
# #### - Distribution resembles a smooth bell curve centered around 3000-3300MMBTU
# #### - Values extended from 2275 to 4167 consistent with weather driven variablity
# #### - Clean and normally varying distribution with no problematic outliers
# 
# ### Public holidays
# #### - Most months have 1-2 holidays with peaks at 0,1 and 2
# #### - Very few months contain 3 or 4 holidays
# #### - The distribution is discrete and low-variance implying limited impact on overall energy demand
# 
# ### Weekend and calender counts
# #### Saturdays/Sundays
# #### - Both variables take values 4 or 5
# #### - Histogram spikes at these two values confirm calendar structure
# #### - Very low variablility indicating these are weak predictors
# 
# ### Weekend days
# #### -Tkaes values 8,9,10 depending on the month
# #### - Similar calendar-driven behavior with minimal variation
# 
# ### Holiday on weekend
# #### - Strong peak at 0, meaning most holidays fall on weekdays
# #### - Secondary peak at 1, and rare occurrences of 2 or 3
# #### - This feature is well behaved but low-variance
# 
# ### Total Off Days
# #### - Typically 8-11 days combining weekends + holidays
# #### - Creates distinct bars(8-13)
# #### - Slightly higher frequence for 9 and 10
# #### - Useful but still limited in predictive power
# 
# ### Economic and weather variables
# 
# ### Industrial Production Index:
# #### - Distributed close to a normal curve centered around 100, the baseline of economic activity
# #### - Ranges from appx 82 to 113 reflecting macroeconomic cylces
# #### - Good candidate for modeling demand tied to overall economic strength
# 
# ### Temperature (temp_avg_f)
# #### -The distribution is multi modal capturing seasonal temperature patterns
# #### 30F in winter
# #### 40-60F in spring/fall
# #### 70-75F in summer
# #### - Well distributed with no distortion
# #### - Clear seasonal signal indicating strong predictor for electricity sales
# 
# ### Heating Degree Days(HDD)
# #### - This histogram spikes heavily at 0 indicating many months require no heating
# #### - Values gradually increase upto 35 for cold months
# #### - Strong seasonality and clear winter signiture
# 
# ### Cooling Degree Days
# #### - Similar structure as HDD but for summer
# #### - Majore spike at 0 CDD(non-summer months)
# #### - Tail extending to 10-12 CDD in peak hot months
# #### - This variable strongly correlates with air conditioning usage
# 
# ### Precipitation (precip_inches)
# #### - A rougly normal distribution centered around 2.5 to 2.8 inches
# #### - Occassional higher rainfall months exist but no extreme outliers
# #### - May have limited predictive power of slaes
# 
# ### Engineered Time Features:
# #### Year
# #### - Unifrom distribution: each year contributes 12 observations
# #### - confirms complete montly coverage across the time range
# 
# ### Month number(1-12)
# #### - Perfectly uniform every month of the year repeats equally
# #### - captures seasonality and cyclical patterns
# 
# ### Summary of Insights from Histograms
# #### 1. Business variables (customers, revenue, sales) show organic growth and seasonal variation without anomalies
# #### 2. Calendar variables exhibit extremely low variance. their impact will be limited
# #### 3. Weather variables (HDD, CDD, temperature) present strong seasonal signitures and are likely powerful predictors
# #### 4. Industrial production appears normally distributed and aligned with economic cycles
# #### 5. No variables show problematic outliers making the dataset clean and ready for modeling
# #### 6. Target variables(sales_mkwh) is well behaved with natural seasonality and moderate spread

# In[22]:


# =========================
# 8. Bar Plots for Categorical Columns
#    (Top 10 categories)
# =========================

cols_to_plot = cat_cols

for col in cols_to_plot:
    vc = df[col].value_counts().head(10)
    plt.figure()
    vc.plot(kind="bar")
    plt.title(f"{col} (top 10)")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# ### Categorical Variable: Month(Top 10 Values)
# #### This month variable represents calendar months in the datasetand is encoded as a datetime type. since the dataset contain exactly one data record per month, each month occurs once. The top 10 months shown in the plot each have a frequency of 1, confirming that the time series has no duplication of monthly entries
# 
# ### This behavior is expected in a clean monthly time series structure and indicates
# #### - No missing months
# #### - No duplicated time stamps
# #### - Uniform Spacing between observations
# 
# #### The bar chart itself does not reveal distributional characteristics because month is inherently unique identifier rather than a categorical variable with meaningful repetetion. However it validates the integrity of the timestamp index and confirms that the dataset is complete and chronologically consistent
# 

# In[23]:


for col in num_cols:
    if col == target:
        continue
    plt.figure(figsize=(6,5))
    plt.scatter(df[col], df[target], alpha=0.6)
    plt.title(f"{target} vs {col}")
    plt.xlabel(col)
    plt.ylabel(target)
    plt.tight_layout()
    plt.show()


# ### Scatter plot Analysis:
# ### Relationship between predictors and electricity sales(sales_mkwh)
# 
# #### scatter plots help visualize how each independent variable influences electricity demand. The following section provides a structured analysis across demand drivers, economic factors, weather conditions calendar features and time based effects
# 
# ### Demand & Economic Activity Drivers
# 
# ### Salse_mkwh Vs Customers
# #### - A broad positive trend is visible where more customers is more sales
# #### - However high scatter indicates customer count is a slow moving structural varialbe, not a month-to-month demand driver
# #### - Electricity consumption per customer to long term upward demand but short term variation is explained by other factors
# 
# #### Conclusion:
# #### Customers base growth contributes to long-term upward demand but short-term variation is explained by other factors
# 
# ### Sales_mkwh vs revenue_musd
# #### - Shows a strong linear positive relationship
# #### - Higher revenues correspond directly to higher electricity sales(as expected: Revenue = price x quantity)
# #### - Points cluster along an upward diagonal
# 
# #### Conclusion:
# #### Revenue and sales are highly correlated including both may create multicollinearity in models
# 
# ### Sales_mkwh vs energy_consumption_million_mmbtu
# #### - Very clear direct positive relationship
# #### - As total energy consumption increases, electricity sales rise proportionally
# #### -  One of the strongest patterns among all scatter plots
# 
# #### Conclusion
# #### This variable is high-power predictor of electricity demand
# 
# ### Sales_mkwh vs individual_production_index
# #### - Strong positive relationship: higher industrial activity indicating higher electricity consumption
# #### - Data shows multiple clear upward channels ( indicative of seasonal cycles)
# 
# #### Conclusion:
# #### Industrial activity is a major economic driver of electricity demand and strongly influences sales.
# 
# ### Price Sensitivity:
# ### Sales_mkwh vs price_cents_per_kwh
# #### - No negative correlation and instead weak or slightly positive
# #### - Electricity is a inelastic commodity, i.e. consumers do not react sharply to the price changes
# #### - Weather and industry overshadow pricing effects
# 
# #### Conclusion:
# #### Price does not significantly affect monthly demand. Demand price is inelastic in this dataset
# 
# ### Weather & Temperature related drivers
# #### Weather is usually ne of the strongest determinants of electricity use due to heating and cooling loads
# 
# ###  sales_mkwh vs temp_avg_f
# #### - Cleared U-shaped / non linear relationship
# #### - low-temps leads to higher heating usage that lead to higher sales
# #### - Higher temps leads to higher AC usage that leads to higher sales
# #### - Mid range temperatures (45-60F) show lower demand
# #### - Classic temperature electricity load curve
# 
# #### Conclusion:
# #### Temperature strongy influences electricity demand in a non-linear fashion. Polynomial or interaction terms may improve models
# 
# ### sales_mkwh vs heating_degree_days(HDD)
# #### - Sales increase gradually with more heating days
# #### - Points at HDD = 0 from a vertical cluster (months with no heating needed)
# #### - Beyond 20 HDD, a clear upward trend in sales
# 
# #### Conclusion:
# #### Heating demand meaningfully drives consumption during cold months
# 
# ### sales_mkwh vs cooling_degree_days (CDD)
# #### - Very strong upward trend
# #### - Higher CDD indicating higher electricity usage(air conditioning)
# #### - One of the strongest weather predictors
# 
# #### Conclusion:
# #### Cooling demand is a major contributor to peak electricity consumption
# 
# ### sales_mkwh vs precip_inches
# #### - No clear trend. points are widely scattered
# #### - Slight concentration of higher sales around 2.5 - 3 inches but not consistent
# 
# #### Conclusion:
# #### precipitation does not significantly impact electricity sales
# 
# ### Calendeer Variables(Holidays, Weekends):
# #### These variables have very low variance (fixed counts per month) limiting their predictive value
# 
# ### sales_mkwh vs public_holidays
# #### - Public holidays: 0 to 4 per month
# #### - No clear upward/downward trend
# #### - Sales distributed similarly across all holiday counts
# 
# #### Conclusion:
# #### Public holiday do not significantly affect monthly electricity demand
# 
# ### sales_mkwh vs weekend_days
# #### - Each variable takes only a few values (4,5 or 8,9,10)
# #### - Data forms vertical bands
# #### - No distinct trend
# 
# #### Conclusion:
# #### Weekend counts do not contribute to monthly sales variation
# 
# ### sales_mkwh vs total_off_days
# #### - A small drop in sales for months with 12-13 off days appear but consistent 
# #### - High noise with each category 
# 
# #### Conclusion:
# #### Total off days have minimal predictive value
# 
# 
# ### Time Based Features:
# 
# ### sales_mkwh vs year
# #### - A gentle upward trend indicates long-term growth in electricity demand
# #### - A strong seasonal fluctuation remain visible every year
# #### - High scatter within each year shows monthly variation dominates annual variation
# 
# #### Conclusion:
# #### - A trend component exists. Models should include time index or rolling features
# 
# ### sales_mkwh vs month_num:
# #### - Very clear seasonal pattern
# #### - Peaks around month 7-8 (summer cooling demand)
# #### - Secondary peaks around months 1-2 (winter heating)
# #### - Lowest around months 4-5 and 10-11 (mild weather)
# 
# #### Conclusion:
# #### Seasonality is one of the strongest structural drivers of elasticity demand
# 
# ### Summary of scatter plot insights
# 
# #### - Weather is strongest short term drivers
# #### - Years and months are strong cyclical and long term effects
# #### - Customers is long term but has weak month-to-month impact
# #### - Price shows demand is inelastic
# #### - Total off days is weak and noisy relationship
# #### - Precipitation shows no strong pattern

# In[24]:


# Ensure month column is in datetime format
df['month'] = pd.to_datetime(df['month'])

# Sort by date (important)
df = df.sort_values('month')

# Line plot
plt.figure(figsize=(14, 6))
plt.plot(df['month'], df['sales_mkwh'], marker='o', linestyle='-', alpha=0.7)

plt.title("Monthly Electricity Sales Over Time", fontsize=16)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Sales (M kWh)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


# #### The line plot illustrates the monthly electricity sales (in million kWh) from January 2001 to August 2025. This visualization is essential for understanding long-term consumption trends, seasonal behavior and structural shifts in demand
# 
# ### Long-Term upward Trend
# #### A clear upward trajectory is visible across 25 year period
# #### Electricity sales gradually rise from around 260000 - 280000 MkWh in the early 2000s to 380000 - 410000 MkWh by 2025
# 
# #### This long-term growth suggests:
# #### - Expansion in population and commercial activity
# #### - Higher appliance penetration
# #### - Increased industrialization
# #### - Growth in energy demand in both residential and commercial sectors
# 
# ### Strong Seasonal Pattern
# #### The plot exhibits a highly repetetive yearly cycle indicating strong seasonality:
# #### - Peak every summer and winter corresponding to:
# #### - Air conditioning loads (cooling degree days)
# #### - Heating loads (heating degree days)
# #### - Dips in spring and fall when weather driven demand in lowest
# 
# #### The seasonal behavior confirms the strong relationship between electricity sales and temperature variations aligning with patterns observed in the scatter plots
# 
# ### Increase seasonal Amplitude:
# #### The seasonal peaks appear to get higher over time meaning:
# #### - Summer and winter demand is rising faster than neutral months
# #### - Climate changes (warmer summers, colder winter or both)
# #### - Greater reliance on HVAC systems
# #### - Urban growth increasing temperature sensitive loads
# 
# ### Year to Year visibility:
# #### Some years show sharper peaks or dips compared to others. These fluctuations may reflect:
# #### - Extreme weather years
# #### - Economic conditions
# #### - Industrial activity cycles
# #### - Policy or tariff changes
# #### - Impact of event like Covid-19(noticable irregularities around 2020)
# 
# ### No abrupt structural Breaks:
# #### Despite variabiity the overall trend remains smooth and consistent indicating
# #### - No sudden regime shifts
# #### - Stable growth in electricity demands
# #### - Predictable seasonal cycles
# 
# ### Implication for modeling
# #### This plot strongly suggests that
# #### - Time independent features (lags, moving averages, rolling means) will help the model
# #### - Seasonality must be explicityly modeled
# #### - Weather variables like temps, HDD, CDD will significantly improve predictive accuracy
# #### - A model without trends would underperform
# 
# ### Summary:
# #### The monthyly electricity sales series displays a clear upward long term trend, strong seasonality and increasing seasonal amplitude over the 25 year period. Sales peak consistently during high temperature months(summer) and low-temperature months(winter) reflecting the influence of heating and cooling loads. The overall pattern indicates growing energy consumption driven by economic expansion, population growth and climate demand. These characteristics highlight the importance of incorporating temporal weather related variables when modeling electricity sales.

# In[25]:


# =========================
# 9. Time-series Plots (if month exists)
# =========================

if "month" in df.columns:
    ts_targets = [
        col for col in num_cols
        if col not in ["public_holidays", "saturdays", "sundays",
                       "weekend_days", "holiday_on_weekend", "total_off_days"]
    ]

    for col in ts_targets:
        plt.figure()
        plt.plot(df["month"], df[col])
        plt.title(f"{col} over time")
        plt.xlabel("Month")
        plt.ylabel(col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# In[26]:


if "month" in df.columns:
    plt.figure(figsize=(12,5))
    plt.plot(df["month"], df[target])
    plt.title("sales_mkwh over time")
    plt.xlabel("Month")
    plt.ylabel("sales_mkwh")
    plt.xticks(rotation=45)
    plt.show()


# ### Time Series Overview:
# #### This section summarizes the exploratory analysis of all time-dependent features based exclusively on the 12 time series plot provided. Each graph helps understnad how different variables evolve monthly from 2001 to 2025 and how they may influence electricity sales.
# 
# ### Target Variable: Monthly Electricity Sales(sales_mkwh)
# #### The line plot of sales_mkwh shows:
# #### - A clear upward trend over the 25 year period
# #### - Well defined seasonal cycles, repeating each year
# #### - Seasonal peaks correspond to high-demand months (summer cooling & winter heating)
# #### - Troughs correspond to moderate weather months
# #### This indicates that electricity sales are strongly seasonal and weather dependent with long term economic growth also contributing
# 
# ### Sales Vs. Month Scatter plot
# #### The month wise scatter plot exhibits:
# #### - Vertical stripes for each month, showing that monthly demand repeats cyclically
# #### - No abnormal spikes or sudden drops
# #### - Tight clustering within each month indicating stable yearly patterns
# #### This confirms strong seasonality and month-driven behavior
# 
# ### Year Progression over Time
# #### The line plot of year across the timeline is a smooth upward staircase
# #### - Every year progresses correctly without jumps or gaps
# #### - Confirms complete temporal coverage from 2001 to 2025
# #### - No missing year blocks
# #### This ensures the time series is chronologically consistent
# 
# ### Total Off days (Weekends + Public holidays):
# #### The total_off_days trend shows:
# #### - Consistent values between 9 and 13 days per month
# #### - Slight fluctuations dependign on that year's calendar
# #### - No long-term trend
# #### This feature is also stationary useful for demand prediction around low-usage periods
# 
# ### Industrial Production Index(IPI) over time:
# #### The industrial_production_index displays:
# #### - Long-term growth trend from 2001 to approx 2008
# #### - Fluctuations between 2008-2015 ( possible recession effects)
# #### - Gradual stabilization after 2015
# #### - No abrupt anomalies
# #### IPI correlates with overall economic activity and may influence commercial electricity consumption
# 
# ### Average Monthly Temperature
# #### The temp_avg_f plot shows:
# #### - Clear yearly seasonality
# #### - Repeating warm-cold-warm cycles
# #### - No long term temperature trend or drift
# #### Temperature is a major driver of electricity demand(AC/heating usage)
# 
# ### Heating Degree Days:
# #### The heating degree days plot shows:
# #### - Strong seasonality: high values during winter months
# #### - close to zero during summer months
# #### - Inter-year variabiity is mild but present
# #### HDD strongly impacts heating-driven electricity demands
# 
# ### Cooling Degree Days(CDD):
# #### The cooling degree days plot exhibits:
# #### - Opposite pattern of HDD indiciating high in summer
# #### - Almost zero during winter
# #### - Very strong seasonality, peaking consistently each year
# #### CDD is the primary driver of AC-driven electricity demand
# 
# ### Precipitation over Time:
# #### The precip_inches plot shows:
# #### - Mild variability month to month
# #### - No long-term trend
# #### - Seasonal rainfall peaks visible across years
# #### while precipitation is not a dominant factor in electricity sales, it might influency demand indirectly like cloud cover, strom etc.
# 
# ### Summary:
# ### Strong seasonal structure
# #### - Almost all weather related and demand related variables exhibit annual cycles
# 
# ### Mixture of trending and stationary features
# 
# #### Trending
# #### - sales_mkwh
# #### - industrial_production_index
# 
# #### Stationary/cyclic
# #### - temp_avg_f
# #### - heating_degree_days
# #### - cooling_degree_days
# #### - public_holidays
# #### - Weekend_days
# #### - total_off_days
# #### - precip_inches
# 
# ### Perfect temporal Integrity
# #### year and month plots confirm no missing periods
# 
# ### Suitability for Time-series forecasting
# #### The trends & seasonality suggests the following:
# #### - SARIMA
# #### - XGBoost with time features
# 
# ### Summary:
# #### The time series plots collectively reveal a dataset with strong annual seasonality, consistent monthly cycles and long term growth in electricity sales. Sales exhibit both trend and seasonal peaks while weather related variables such as temperature, HDD and CDD show highly structured yearly patterns. Calendar based variables including weekends, public holidays, and off days remain stationary over time with expectd cyclic behavior. Industrial production displays gradual economic trends that may influence commercial electricity consumption. All temporal plots confirm perfect chronological continuity from 2001 to 2025. Overall the time series is clean, well structured and shows clear seasonal and temporal dynamics making it highly suitable for forecasting models that incorporate trend and seasonality
# 
# 

# In[27]:


from statsmodels.tsa.seasonal import seasonal_decompose

df_ts = df.set_index("month")

result = seasonal_decompose(df_ts[target], model='additive', period=12)
result.plot()
plt.show()


# ### Seasonal Decomposition of Monthly Electricity Sales(sales_mkwh)
# #### Additive decomposition: Trend, Seasonality and Residuals
# #### To better understand the underlying structure of electricity sales over time, an additive time series decomposition was performed. The decomposition separates the observed series into three components. Trend, Seasonality and Residuals providign a clearer view of long term movements and repeating annual patterns.
# 
# ### Observed Series:
# #### The top panel displays the raw monthly electricity sales over the full period(2001-2025)
# #### Key Observations:
# #### - Sales exhibit a strong yearly seasonal cycle, with predictable peaks and troughs
# #### - There is a clear upward long-term growth, especially prominent after 2018
# #### - Short-term fluctuations are present but do not disrupt overall structure
# #### This confirms that the sereies is influenced by both economic growth and weather driven seasonality
# 
# ### Trend Component
# #### The second panel isolates the underlying longterm sales trajectory by smoothing out seasonal and irregular variations
# #### Insights:
# #### - The trend grows steadily from 280000M kWh in 2001 to 330000M kWh in 2025
# #### - A period of flattening or mild declide occurs around 2009-2011 which could be associated with global economic slow down
# #### - From 2015 onward, the trend rises consistently indicating higher energy consumption over time
# #### This component reflects overall economic expansion increasing population and industrial activity
# 
# ### Seasonal Component:
# #### The third panel shows the repetitive seasonal pattern that occurs every year.
# #### Interpretation:
# #### - The seasonal amplitude is significant with variation of +or- 50000M kWh around the mean
# #### - Clear peaks occur during summer months when cooling demand rises sharply
# #### - Secondary peaks in winter months corresponds to heating loads in colder periods
# #### - Troughs appear in mild-climater months(springs/autumn) when energy demand is naturally lower
# #### The seasonal pattern is highly regular confirming that electricity consumption is strongly driven by temperature cycls and season dependent usage behavior
# 
# ### Residual Component:
# #### The bottom panel shows the residuals and the portion of the series is not explained by trend or seasonality
# #### Observations:
# #### - Residuals fluctuate around zero with no long-term drift
# #### - A few spikes exist indicating occasional unusual consumption months, possibly due to
# #### -- Extreme weather events
# #### -- Unplanned outages
# #### -- Economic shocks
# #### -- Pandemics or sudden activity drop/rise
# #### Overall the residuals appear random and well-behaved suggesting the decomposition successfully captured the main structure of the time series
# 
# ### Summary of Findings:
# #### The decompositiion reveals that:
# #### Electricity sales have a strong, stable annual seasonal pattern driven by weather(heating/cooling)
# #### The long-term trend shows consistent growht, especially after 2015
# #### Residual fluctuations are small indicating that the data is regular, predictable and well suited for forecasting
# #### Both trend+seasonality together explain most of the demand variation which is ideal for time series models like SARIMA, Prophet, XGBoost with time features of LSTM

# ## Sales lag impact

# In[28]:


df["sales_lag1"] = df[target].shift(1)
df["sales_lag2"] = df[target].shift(2)

print(df[["sales_mkwh", "sales_lag1", "sales_lag2"]].corr())


# In[29]:


plt.figure(figsize=(7,5))
plt.scatter(df["sales_lag1"], df[target])
plt.title("sales_mkwh vs sales_lag1")
plt.xlabel("sales_lag1")
plt.ylabel("sales_mkwh")
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(df["sales_lag2"], df[target])
plt.title("sales_mkwh vs sales_lag2")
plt.xlabel("sales_lag2")
plt.ylabel("sales_mkwh")
plt.show()


# ### Lag Feature Analysis for sales_mkwh
# #### To understand temporal dependence in monthly electricity sales, lag features were created
# 
# #### sales_lag1 is sales from the previous month
# #### sales_lag2 is sales form two months before
# 
# #### A correlation matrix and scatter plots were used to analyse how strongly past sales values predict current sales
# 
# ### Key Insights:
# #### Strong positive Autocorrelation at Lag-1 (0.65)
# #### - The correlation between the current month's sales and the previous month's sales is 0.652 which indicates strong temporal dependence
# #### - This confims that sales data exhibits firt order auto correlation a common behavior in monthly time series data
# 
# #### Weak Relationship at Lag-3(0.067)
# #### - The correlation with two months ago is only 0.067 showing negligble predictive relationship between current sales and sales from 2 months prior
# 
# #### Lag1 and Lag2 are moderately correlated(0.643)
# #### - This is expected because monthly sales tend to move gradually resulting in adjacent lags being related
# 
# #### Conclusion from correlation matrix:
# #### - Lag1 is a useful predictor for foreasting models (AR, ARIMA, ML Regressors)
# #### -- Lag2 does not add much indepenedent predictive power when lag1 is already included
# 
# ### Scatter plot Analysis:
# #### A. sales_mkwh vs sales_lag1
# #### Observations:
# #### - Points form a clear upward sloping cloud, confirming strong positive dependence
# #### - When the previous month's sales increase the current month's sales also tend to increase
# #### - The relationship appears approximately linear with mild seasonal spread
# 
# #### Interpretation:
# #### - Monhly electricity demand changes gradually. High demand months cluster together
# #### - Lag1 is a strong leading indicator of upcoming months demand
# 
# #### B. Sales_mkwh vs sales_lag2
# #### Observations:
# #### - The scatter plot appears widely dispersed without a clear slope
# #### - Very weak clustering or pattern
# 
# #### Interpretation:
# #### - Sales from two months ago do not influence the current month meaningfully
# #### - This supports the low correlation observed in the matrix
# 
# ### Executive summary:
# #### - Monthly electricity sales show strong month-to-month continuity evidenced by a lag1 correlation of 0.652
# #### - Lag2 shows almost no predictive relationship, indicating demand patterns do not persist beyond one month
# #### - Scatter plots confirm thse relationships visually tight, upward sloping trend for lag1 and weak dispersion for lag2
# #### For forecasting models(ARIMA, SARIMA, LSTM, XGBoost) lag1 should be included as  a key feature while lag2 offers minimal benefit and my be excluded unless part of a seasonal autoregressive strucutre

# In[30]:


# =========================
# 10. Correlation Matrix (Numeric)
# =========================

if len(num_cols) >= 2:
    corr = df[num_cols].corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, interpolation="nearest")
    plt.xticks(range(len(num_cols)), num_cols, rotation=90)
    plt.yticks(range(len(num_cols)), num_cols)
    plt.title("Correlation matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    corr


# In[31]:


target_corr = df.corr()[target].sort_values(ascending=False)
print(target_corr)


# ### Correlation of Predictors with Electricity Sales(sales_mkwh)
# #### The table summarizes the Pearson correlation between monthly electricity sales (sales_mkwh) and all other numerical variables. These correlations help identify which predictors have the strongest linear relationships with electricity demand.
# 
# #### Below is a structured interpretation of the relationships, grouped into Strong, Moderate, and Weak/Negative correlations.
# 
# ### Strong Positive Correlations (Highly Predictive)
# #### These variables have correlation above 0.70, indicating they move closely with electricity sales and are likely to be key predictors.
# 
# #### - Revenue per Customer — 0.923
# #### This is the strongest correlate of sales. Higher electricity sales naturally increase total revenue per customer.
# 
# #### - kWh per Customer — 0.898
# #### Shows that customer-level consumption tracks very closely with total sales. Indicates strong demand-side alignment: when customers use more energy, system-wide sales rise proportionally.
# 
# #### - Cooling Degree Days (CDD) — 0.817
# #### High CDD values (hotter months) strongly increase electricity demand. Explains the summer peaks observed in the time series.
# 
# #### - Revenue (Total) — 0.802
# #### Expected strong relationship as electricity sold increases, revenue rises proportionally.
# 
# ### Moderate Positive Correlations
# #### These variables have correlation 0.40–0.70, indicating meaningful but not dominating influence.
# 
# #### - Energy Consumption (million MMBtu) — 0.655
# #### Reflects broader system energy usage. Higher fuel consumption for generation aligns with higher electricity output.
# 
# #### - Average Temperature (temp_avg_f) — 0.538
# #### Warmer temperatures increase air-conditioning load. Strong seasonal driver of electricity consumption.
# 
# #### - Electricity Price (price_cents_per_kwh) — 0.507
# #### Prices may increase during high-demand seasons. Regulatory/cost-push changes coincide with rising overall consumption.
# 
# #### - Revenue per kWh — 0.507
# #### Again reflects price × consumption dynamics.
# 
# #### - Industrial Production Index — 0.421
# #### Reflects industrial load contribution. Higher manufacturing output increases electricity demand.
# 
# ### Weak Positive Correlations (Small or Indirect Impact)
# #### These variables have correlation 0.10–0.35.
# 
# #### - Month — 0.339
# #### Captures seasonality to some extent, but limited on its own.
# 
# #### - Year — 0.335
# #### Long-term upward trend in sales over the years due to population, infrastructure, and economic growth.
# 
# #### - Customers — 0.252
# #### Customer count changes slowly over years, whereas sales fluctuate strongly with season.
# 
# #### - Precipitation (inches) — 0.150
# #### Minimal impact — rainfall doesn’t significantly influence electricity consumption.
# 
# #### - Month Number — 0.100
# #### Weak standalone indicator of seasonality.
# 
# ### Very Weak / Near-Zero Correlations
# #### These variables show correlation between −0.05 and +0.05, indicating almost no linear relationship:
# #### - Weekend_days — 0.033
# #### - Sundays — 0.032
# #### - Saturdays — 0.024
# #### Monthly electricity sales do not meaningfully depend on weekend count.
# 
# 
# ### Negative Correlations
# #### Variables with negative correlations reduce electricity sales when they increase.
# 
# #### - Heating Degree Days (HDD) — −0.413
# #### Cold-weather months have lower electricity sales.Indicates electricity usage is cooling-dominant rather than heating-dominant — typical in warmer regions of the US.
# 
# #### - Public Holidays — −0.232
# #### Electricity demand drops slightly during holiday-heavy months due to reduced industrial/commercial activity.
# 
# #### - Total Off Days — −0.134
# #### More off days reduce industrial and commercial load → lower sales.
# 
# #### - Holiday on Weekend — −0.067
# #### Minor effect.
# 
# ### Overall Interpretation
# 
# ### Final Summary:
# #### - The correlation analysis confirms that electricity demand is primarily driven by weather (especially heat), customer consumption patterns, and industrial activity. Calendar and holiday variables play a secondary role, while cold-weather indicators decrease sales.
# 
# #### These insights provide a strong foundation for feature engineering and model selection in the forecasting phase.
# 
# 
# 

# In[32]:


# =========================
# 11. Simple Feature Engineering (Optional)
#      – can be used later for modelling
# =========================

# Safely create ratios if columns exist
if set(["revenue_musd", "sales_mkwh"]).issubset(df.columns):
    df["revenue_per_kwh"] = df["revenue_musd"] / df["sales_mkwh"]

if set(["sales_mkwh", "customers"]).issubset(df.columns):
    df["kwh_per_customer"] = df["sales_mkwh"] / df["customers"]

if set(["revenue_musd", "customers"]).issubset(df.columns):
    df["revenue_per_customer"] = df["revenue_musd"] / df["customers"]

df.head()


# # Model

# In[33]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import pickle

import warnings
warnings.filterwarnings("ignore")

sns.set_theme()


# In[34]:


df = pd.read_csv("master_df.csv")
df.head()


# In[35]:


df.drop(columns="Unnamed: 0", inplace=True)


# In[36]:


df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
df["month_num"] = df["month_num"].astype('category')


# In[37]:


df.info()


# In[38]:


plt.figure(figsize = (12, 4))
sns.lineplot(data = df, x = "month", y = "sales_mkwh")
plt.xticks(rotation=90)
plt.title("Target Variable (Sales in million kilowatt hours)")
plt.show()


# In[39]:


corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(16, 6)) 
sns.heatmap(corr_matrix, 
            annot=True,     
            cmap='coolwarm',
            vmin=-1,        
            vmax=1,         
            center=0,       
            linewidths=.5   
            )
plt.title('Correlation Heatmap of Iris Dataset')
plt.show() 


# In[40]:


X = df.drop(columns = ["month", "customers", "sales_mkwh", "year", "revenue_musd"])
y = df["sales_mkwh"]


# In[41]:


X.columns


# ## Train Test Split

# In[42]:


# Splitting the data into training and testing sets
# Using the last 8 months (2025) as the test set

X_train = X.iloc[:-8]
y_train = y.iloc[:-8]

X_test = X.iloc[-8:]
y_test = y.iloc[-8:]


# ## Model

# In[43]:


model = xgb.XGBRegressor(
    objective='reg:squarederror', # Loss function for regression
    n_estimators=100,             # Number of boosting rounds (trees)
    learning_rate=0.1,            # Step size shrinkage
    max_depth=5,                  # Maximum depth of a tree
    enable_categorical=True, 
    tree_method='hist',
    random_state=42,
    n_jobs=-1                     # Use all available cores
)

model.fit(X_train, y_train)


# ## Evaluation

# In[44]:


y_pred = model.predict(X_test)


# In[45]:


print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"Mean Absolute Percentage Error: {mean_absolute_percentage_error(y_test, y_pred)*100}%")


# ## Hyperparameter Tuning

# In[46]:


# Define the hyperparameter search space
param_grid = {
    'n_estimators': [100, 200, 500, 1000],        # Number of trees
    'learning_rate': [0.01, 0.05, 0.1],# Step size shrinkage
    'max_depth': [3, 5, 7, 10],             # Depth of tree
    'subsample': [0.6, 0.8, 1.0],           # % of rows used per tree
    'colsample_bytree': [0.6, 0.8, 1.0],    # % of columns used per tree
    'gamma': [0, 0.1, 0.2]                  # Minimum loss reduction required to make a split
}


# In[47]:


grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_absolute_percentage_error', # Metric to optimize (negative MAPE)
    cv=3,                               # 3-fold Cross-Validation
    verbose=1,
    n_jobs=-1                           # Use all processors
)

grid_search.fit(X_train, y_train)


# In[48]:


# Best Parameters
grid_search.best_params_


# In[49]:


best_model = grid_search.best_estimator_


# ## Evaluation

# In[50]:


y_pred = best_model.predict(X_test)


# In[51]:


print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"Mean Absolute Percentage Error: {mean_absolute_percentage_error(y_test, y_pred)*100}%")


# In[52]:


y_pred = pd.Series(y_pred)
y_pred.index = y_test.index
y_pred


# In[53]:


# Plot the split
plt.figure(figsize=(10, 4))
plt.plot(y_train, label='Training Data')
plt.plot(y_test, label='Test Data', color='green')
plt.plot(y_pred, label='Test Data Forecasts', color='darkgrey')
plt.title("XGBoost Model Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()


# In[54]:


# Define the filename for your pickle file
filename = 'xgboost_model.pkl'

# Open the file in binary write mode ('wb') and dump the model
with open(filename, 'wb') as file:
    pickle.dump(best_model, file)

