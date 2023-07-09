# Customer Personality Analysis
# Analysis of company's ideal customers

# About Dataset
# Context
# Problem Statement
#
# Customer Personality Analysis is a detailed analysis of a company’s ideal customers. It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviors and concerns of different types of customers.
#
# Customer personality analysis helps a business to modify its product based on its target customers from different types of customer segments. For example, instead of spending money to market a new product to every customer in the company’s database, a company can analyze which customer segment is most likely to buy the product and then market the product only on that particular segment.
#
# Content
# Attributes
#
# People
#
# ID: Customer's unique identifier
# Year_Birth: Customer's birth year
# Education: Customer's education level
# Marital_Status: Customer's marital status
# Income: Customer's yearly household income
# Kidhome: Number of children in customer's household
# Teenhome: Number of teenagers in customer's household
# Dt_Customer: Date of customer's enrollment with the company
# Recency: Number of days since customer's last purchase
# Complain: 1 if the customer complained in the last 2 years, 0 otherwise
# Products
#
# MntWines: Amount spent on wine in last 2 years
# MntFruits: Amount spent on fruits in last 2 years
# MntMeatProducts: Amount spent on meat in last 2 years
# MntFishProducts: Amount spent on fish in last 2 years
# MntSweetProducts: Amount spent on sweets in last 2 years
# MntGoldProds: Amount spent on gold in last 2 years
# Promotion
#
# NumDealsPurchases: Number of purchases made with a discount
# AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
# AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
# AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
# AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
# AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
# Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
# Place
#
# NumWebPurchases: Number of purchases made through the company’s website
# NumCatalogPurchases: Number of purchases made using a catalogue
# NumStorePurchases: Number of purchases made directly in stores
# NumWebVisitsMonth: Number of visits to company’s website in the last month


###############################################################
# Mission 1: Data Preparing
###############################################################
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

df_ = pd.read_csv("crmAnalytics/datasets/marketing_campaign.csv", sep="\t")
df = df_.copy()
df.head()


# Data Understanding

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)
df.columns
df = df.drop(
    labels=['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Z_CostContact',
            'Z_Revenue', 'Response'], axis=1)

df.head()

# We have null values in income,we will fill null values.
df["Income"].isnull().sum()

df["Income"].fillna((df["Income"].median()), inplace=True)

# We have Recency value,so we won't change type for date.

# Creating RFM Metrics
rfm = pd.DataFrame()
# ID
rfm["Customer_id"] = df["ID"]

# Frequency
rfm["frequency"] = df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]

# Monetary
rfm["monetary"] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df[
    'MntSweetProducts'] + df['MntGoldProds']

# Recency
rfm["recency"] = df["Recency"]

# RFM VALUES
# Recency

rfm["Recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])

# Frequency
rfm["Frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

# Monetary
rfm["Monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm.head()
rfm["Customer_id"].duplicated().unique()

# Now,We will generate the RF score.It is formed by the combination of Recency and Frequency scores.

rfm["RF_SCORE"] = (rfm["Recency_score"].astype(str) + rfm["Frequency_score"].astype(str))

rfm.describe().T

rfm[rfm["RF_SCORE"] == "11"].head()

rfm[rfm["RF_SCORE"] == "55"].head()

# Definition of RF Scores as Segments
# We will perform the segmentation process with seg_map.

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

rfm["Segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

rfm.head()

# We will examine the recency, frequency and monetary averages of the segments.

rfm[["Segment", "recency", "frequency", "monetary"]].groupby("Segment").agg(["mean", "count"])

rfm[rfm["Segment"] == "loyal_customers"].head()
rfm[rfm["Segment"] == "loyal_customers"].index

# We can do a analysis process according to income levels.

df["Income"].describe().T
intervals = [1730.00, 35538.75, 51381.50, 68289.75, 666666.00]
new_labels = ["low income", "middle income", "high-middle income", "high income"]
rfm["Income_level"] = pd.cut(df["Income"], intervals, labels=new_labels)
rfm.head()

sns.catplot(
    data=rfm, y="Income_level", hue="Segment", kind="count",
    palette="pastel", edgecolor=".6",
)

plt.show()

df.info()

# We can do another analysis process according to Age.
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
df["Dt_Customer"].max()  # Timestamp('2014-12-06 00:00:00')
df["Age"] = 2015 - df["Year_Birth"]

df["Age"].describe().T

intervals = [18, 25, 40, 67, 122]
new_labels = ["19_25", "26_40", "41_67", "67_100"]
rfm["Age"] = pd.cut(df["Age"], intervals, labels=new_labels)
rfm.head()

sns.catplot(
    data=rfm, y="Age", hue="Segment", kind="count",
    palette="pastel", edgecolor=".6",
)

plt.show()
