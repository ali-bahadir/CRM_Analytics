##############################################################
# BG-NBD and Gamma-Gamma with CLTV Prediction
##############################################################

###############################################################
# Business Problem
###############################################################
# FLO wants to determine a roadmap for its sales and marketing activities.
# The company needs to predict the potential value that existing customers
# will bring to the company in the future for mid-to-long-term planning.

###############################################################
# Dataset Story
###############################################################

# The dataset includes information obtained from the past shopping behaviors
# of customers who made their last purchases in 2020 - 2021 through OmniChannel
# (both online and offline).

# master_id: Unique customer number
# order_channel: Platform used for shopping (Android, iOS, Desktop, Mobile, Offline)
# last_order_channel: The channel of the last purchase
# first_order_date: The date of the customer's first purchase
# last_order_date: The date of the customer's most recent purchase
# last_order_date_online: The date of the last purchase on the online platform
# last_order_date_offline: The date of the last purchase on the offline platform
# order_num_total_ever_online: Total number of purchases made online
# order_num_total_ever_offline: Total number of purchases made offline
# customer_value_total_ever_offline: Total spending in offline purchases
# customer_value_total_ever_online: Total spending in online purchases
# interested_in_categories_12: Categories the customer shopped in over the last 12 months

###############################################################
# TASKS
###############################################################

# TASK 1: Data Preparation

import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

# 1. Read the flo_data_20K.csv file and create a copy of the dataframe.

df_ = pd.read_csv("crmAnalytics/datasets/flo_data_20k.csv")
df = df_.copy()


# Step 2: Define the necessary functions for suppressing outliers
# Note: Frequency values must be integers when calculating CLTV.
# Therefore, round the upper and lower limits with the round() function.

def outlier_thresholds(dataframe, variable):  # Determines threshold values for the given variable
    quartile1 = dataframe[variable].quantile(0.01)  # Suppresses outliers in the dataframe
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):  # Replaces values exceeding thresholds
    low_limit, up_limit = outlier_thresholds(dataframe, variable)  # Adjusts values above or below thresholds
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)


# Step 3: Suppress outliers for the variables:
# "order_num_total_ever_online", "order_num_total_ever_offline",
# "customer_value_total_ever_offline", "customer_value_total_ever_online"

columns = ["order_num_total_ever_online", "order_num_total_ever_offline",
           "customer_value_total_ever_offline", "customer_value_total_ever_online"]

for col in columns:
    replace_with_thresholds(df, col)

# Step 4: Omnichannel indicates that customers shop from both online and offline platforms.
# Create new variables for each customer's total number of purchases and total spending.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Step 5: Check variable types. Convert variables representing dates to datetime.
df.info()
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

###############################################################
# TASK 2: Creating the CLTV Data Structure
###############################################################

# 1. Set the analysis date as two days after the last purchase in the dataset.
df["last_order_date"].max()  # 2021-05-30
analysis_date = dt.datetime(2021, 6, 1)

# 2. Create a new CLTV dataframe containing customer_id, recency_cltv_weekly,
# T_weekly, frequency, and monetary_cltv_avg values.

# recency: Time since the last purchase (in weeks, for each customer)
# T: Customer's age (in weeks, from the first purchase to the analysis date)
# frequency: Total number of repeat purchases (frequency > 1)
# monetary: Average profit per purchase

cltv_df = pd.DataFrame()
cltv_df['customer_id'] = df['master_id']
cltv_df['recency_cltv_weekly'] = ((df['last_order_date'] - df['first_order_date']).astype('timedelta64[D]')) / 7
cltv_df['T_weekly'] = ((analysis_date - df['first_order_date']).astype('timedelta64[D]')) / 7
cltv_df['frequency'] = df['order_num_total']
cltv_df['monetary_cltv_avg'] = df['customer_value_total'] / df['order_num_total']

cltv_df.head()

###############################################################
# TASK 3: Building BG/NBD and Gamma-Gamma Models, Calculating 6-Month CLTV
# Beta Geometric / Negative Binomial Distribution (models purchase count)
###############################################################

# 1. Fit the BG/NBD model.

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# Predict purchases expected within 3 months and add as exp_sales_3_month to the CLTV dataframe.

cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

# Predict purchases expected within 6 months and add as exp_sales_6_month to the CLTV dataframe.

cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

# 2. Fit the Gamma-Gamma model (models average profit).
# Predict the average value customers are expected to leave and add it as exp_average_value to the CLTV dataframe.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])

# 3. Calculate 6-month CLTV and add it as cltv to the dataframe.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv

# Observe the top 20 customers with the highest CLTV values.
cltv_df.sort_values("cltv", ascending=False)[:20]

###############################################################
# TASK 4: Creating Segments Based on CLTV
###############################################################

# 1. Divide all customers into 4 groups (segments) based on standardized 6-month CLTV.
# Assign segment names to the dataset as cltv_segment.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

cltv_df.groupby("cltv_segment").agg({"count", "mean", "sum"})
