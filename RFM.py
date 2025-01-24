###############################################################
# Customer Segmentation with RFM
###############################################################


# The dataset includes the historical shopping behaviors of customers who made their
# last purchases in 2020 - 2021 through OmniChannel (both online and offline).

# master_id: Unique customer ID
# order_channel : The platform used for shopping (Android, iOS, Desktop, Mobile, Offline)
# last_order_channel : The channel of the most recent purchase
# first_order_date : The date of the first purchase
# last_order_date : The date of the last purchase
# last_order_date_online : The date of the last online purchase
# last_order_date_offline : The date of the last offline purchase
# order_num_total_ever_online : Total number of online purchases
# order_num_total_ever_offline : Total number of offline purchases
# customer_value_total_ever_offline : Total amount spent on offline purchases
# customer_value_total_ever_online : Total amount spent on online purchases
# interested_in_categories_12 : List of categories the customer shopped in the last 12 months

###############################################################
# TASKS
###############################################################

# TASK 1: Data Understanding and Preparation

import pandas as pd
import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)

# 1. Read the flo_data_20K.csv file and create a copy of the dataframe.
df_ = pd.read_csv('crmAnalytics/datasets/flo_data_20k.csv')
df = df_.copy()

# 2. Analyze the dataset by checking:
# a. The first 10 observations,
# b. Variable names,
# c. Dimensions,
# d. Descriptive statistics,
# e. Missing values,
# f. Variable types.
df.head(10)
df.columns
df.shape
df.describe().T
df.isnull().sum()
df.info()

# 3. Omnichannel means customers shop from both online and offline platforms.
# Create new variables for the total number of purchases and total spending for each customer.
df['order_num_total'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['customer_value_total'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']

# 4. Check variable types. Convert variables representing dates to datetime.
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

# 5. Examine the distribution of the number of customers, total products purchased, and total spending across shopping channels.
df.groupby('order_channel').agg({'master_id': 'count',
                                 'order_num_total': 'sum',
                                 'customer_value_total': 'sum'})

# 6. List the top 10 customers bringing the highest revenue.
df.sort_values(by='customer_value_total', ascending=False)[:10]

# 7. List the top 10 customers with the most purchases.
df.sort_values("order_num_total", ascending=False)[:10]

# 8. Functionalize the data preprocessing steps.
def data_prep(dataframe):
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = (dataframe["customer_value_total_ever_offline"] +
                                         dataframe["customer_value_total_ever_online"])
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return df

###############################################################
# TASK 2: Calculating RFM Metrics
###############################################################

# Recency, Frequency, Monetary
# recency = analysis date - the date of the last purchase
# frequency = total purchases
# monetary = total revenue generated from purchases

# Set the analysis date as two days after the last purchase in the dataset.
df["last_order_date"].max()  # 2021-05-30
analysis_date = dt.datetime(2021, 6, 1)

rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (analysis_date - last_order_date.max()).days,
                                   'order_num_total': lambda order_num_total: order_num_total,
                                   'customer_value_total': lambda customer_value_total: customer_value_total.sum()})
rfm.head()

# Rename variables as desired
rfm.columns = ['recency', 'frequency', 'monetary']
rfm["customer_id"] = df["master_id"]

###############################################################
# TASK 3: Calculating RF and RFM Scores
###############################################################

# Convert Recency, Frequency, and Monetary metrics to scores between 1 and 5 using qcut.
# Save these scores as recency_score, frequency_score, and monetary_score.

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# Combine recency_score and frequency_score into a single variable and save it as RF_SCORE.
rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

# Combine recency_score, frequency_score, and monetary_score into a single variable and save it as RFM_SCORE.
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))

rfm.head()

###############################################################
# TASK 4: Defining Segments for RF Scores
###############################################################

# Define segments for RFM scores to make them more interpretable.
# Map RF_SCORE values to segment names using seg_map.
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm.head()

###############################################################
# TASK 5: Taking Action
###############################################################

# 1. Analyze the average recency, frequency, and monetary values of the segments.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# 2. Based on RFM analysis, find customers for the following two cases and save their IDs to CSV files.

# a. FLO is launching a new women's shoe brand with higher-than-average prices.
# Contact loyal (champions, loyal_customers) customers who shop in the women's category.
target_segments_customer_ids = rfm[rfm["segment"].isin(["champions", "loyal_customers"])]["customer_id"]

cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & (df["interested_in_categories_12"].
                                                                      str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("new_brand_target_customer_ids.csv", index=False)
cust_ids.shape

# b. A 40% discount is planned for men's and children's products.
# Target good customers from the past who haven't shopped recently and new customers.
target_segments_customer_ids = rfm[rfm["segment"].isin(["cant_loose", "hibernating", "new_customers"])]["customer_id"]

cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) &
              ((df["interested_in_categories_12"].str.contains("ERKEK")) | (df["interested_in_categories_12"].
                                                                            str.contains("COCUK")))]["master_id"]

cust_ids.to_csv("discount_target_customer_ids.csv", index=False)
