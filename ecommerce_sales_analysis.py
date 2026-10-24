"""
E-COMMERCE SALES DATA ANALYSIS
Dataset: ecommerce_sales_2025_2026.csv
Libraries: Pandas, NumPy, Matplotlib, Seaborn
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "ecommerce_sales_2025_2026.csv"
OUTPUT_DIR = BASE_DIR / "project_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

df = pd.read_csv(DATA_FILE)
print("=" * 60)
print("E-COMMERCE SALES DATA ANALYSIS")
print("=" * 60)
print("\nOriginal dataset shape:", df.shape)
print("Date range in source data:",
      pd.to_datetime(df["Order_Date"]).min().date(), "to",
      pd.to_datetime(df["Order_Date"]).max().date())

# ----------------------------------------------------------
# 2. DATA QUALITY CHECKS AND CLEANING
# ----------------------------------------------------------
print("\nDATA QUALITY CHECKS")
print("-" * 60)
print("Missing values in dataset:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
print("Duplicate Order_ID values:", int(df["Order_ID"].duplicated().sum()))

# Remove exact duplicate rows if any are present.
df = df.drop_duplicates().copy()

# Convert the date field to datetime.
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

# Standardise surrounding spaces and capitalisation in text fields.
text_cols = [
    "Product", "Category", "Region", "City",
    "Customer_Segment", "Payment_Method", "Order_Status"
]
for col in text_cols:
    df[col] = df[col].astype("string").str.strip()

print("Rows after removing duplicate rows:", len(df))
print("Invalid/missing dates after conversion:", int(df["Order_Date"].isna().sum()))
print("Order statuses found:", sorted(df["Order_Status"].dropna().unique().tolist()))
print("Payment methods found:", sorted(df["Payment_Method"].dropna().unique().tolist()))

# The supplied dataset has no missing values or duplicates. If that changes,
# investigate missing dates/IDs and other required fields before analysis.
# This check makes any future data-quality issue visible instead of silently
# filling values with assumptions.
if df["Order_Date"].isna().any():
    raise ValueError("Some Order_Date values could not be parsed. Please inspect the CSV.")
if df["Order_ID"].isna().any():
    raise ValueError("Some Order_ID values are missing. Please inspect the CSV.")

# ----------------------------------------------------------
# 3. FILTER DELIVERED ORDERS AND CALCULATE KPIs
# ----------------------------------------------------------
sales = df[df["Order_Status"] == "Delivered"].copy()
if sales.empty:
    raise ValueError("No delivered orders were found; KPI calculations cannot continue.")

total_orders = len(df)
delivered_orders = len(sales)
cancelled_orders = int((df["Order_Status"] == "Cancelled").sum())
returned_orders = int((df["Order_Status"] == "Returned").sum())
total_quantity = int(sales["Quantity"].sum())
total_revenue = sales["Realized_Revenue"].sum()
total_profit = sales["Realized_Profit"].sum()
average_order_value = total_revenue / delivered_orders
profit_margin = (total_profit / total_revenue) * 100

print("\nKEY PERFORMANCE INDICATORS")
print("-" * 60)
print(f"Total orders: {total_orders:,}")
print(f"Delivered orders: {delivered_orders:,}")
print(f"Cancelled orders: {cancelled_orders:,}")
print(f"Returned orders: {returned_orders:,}")
print(f"Products sold in delivered orders: {total_quantity:,}")
print(f"Total realised revenue: ₹{total_revenue:,.2f}")
print(f"Total realised profit: ₹{total_profit:,.2f}")
print(f"Average order value: ₹{average_order_value:,.2f}")
print(f"Profit margin: {profit_margin:.2f}%")

print("\nREVENUE STATISTICS")
print("-" * 60)
print(f"Mean: ₹{sales['Realized_Revenue'].mean():,.2f}")
print(f"Median: ₹{sales['Realized_Revenue'].median():,.2f}")
print(f"Standard deviation: ₹{sales['Realized_Revenue'].std():,.2f}")
print(f"Skewness: {sales['Realized_Revenue'].skew():.2f}")

# ----------------------------------------------------------
# 4. GROUPED ANALYSIS
# ----------------------------------------------------------
monthly_revenue = sales.groupby(
    sales["Order_Date"].dt.to_period("M")
)["Realized_Revenue"].sum()

top_products = sales.groupby("Product")["Realized_Revenue"].sum().sort_values(
    ascending=False
).head(10)

category_revenue = sales.groupby("Category")["Realized_Revenue"].sum().sort_values(
    ascending=False
)
category_profit = sales.groupby("Category")["Realized_Profit"].sum().sort_values(
    ascending=False
)
region_sales = sales.groupby("Region")["Realized_Revenue"].sum().sort_values(
    ascending=False
)
city_sales = sales.groupby("City")["Realized_Revenue"].sum().sort_values(
    ascending=False
).head(10)
customer_sales = sales.groupby("Customer_Segment")["Realized_Revenue"].sum().sort_values(
    ascending=False
)
payment_methods = sales["Payment_Method"].value_counts()

category_data = sales.groupby("Category").agg(
    Revenue=("Realized_Revenue", "sum"),
    Profit=("Realized_Profit", "sum"),
    Delivered_Orders=("Order_ID", "count")
)
category_data["Profit_Margin_Percent"] = (
    category_data["Profit"] / category_data["Revenue"] * 100
)

print("\nREVENUE BY CATEGORY")
print(category_data.round(2).to_string())
print("\nTOP 10 PRODUCTS BY REVENUE")
print(top_products.round(2).to_string())
print("\nREVENUE BY REGION")
print(region_sales.round(2).to_string())
print("\nTOP 10 CITIES BY REVENUE")
print(city_sales.round(2).to_string())
print("\nORDERS BY PAYMENT METHOD")
print(payment_methods.to_string())

# Helper: consistently save every chart and display it.
chart_count = 0
def finish_chart(filename):
    global chart_count
    chart_count += 1
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

# ----------------------------------------------------------
# 5. DATA VISUALISATION: 19 CHARTS
# ----------------------------------------------------------

# Figure 1: Monthly revenue trend
plt.figure(figsize=(12, 6))
plt.plot(monthly_revenue.index.astype(str), monthly_revenue.values,
         marker="o", linewidth=2)
plt.title("Monthly Revenue Trend", fontsize=15, fontweight="bold")
plt.xlabel("Month")
plt.ylabel("Revenue (₹)")
plt.xticks(rotation=60)
plt.grid(True, alpha=0.3)
finish_chart("01_monthly_revenue.png")

# Figure 2: Top 10 products by revenue
plt.figure(figsize=(10, 6))
sns.barplot(color="steelblue", x=top_products.values, y=top_products.index)
plt.title("Top 10 Products by Revenue", fontsize=15, fontweight="bold")
plt.xlabel("Revenue (₹)")
plt.ylabel("Product")
finish_chart("02_top_products.png")

# Figure 3: Revenue by category
plt.figure(figsize=(9, 6))
sns.barplot(color="steelblue", x=category_revenue.index, y=category_revenue.values)
plt.title("Revenue by Product Category", fontsize=15, fontweight="bold")
plt.xlabel("Category")
plt.ylabel("Revenue (₹)")
plt.xticks(rotation=20)
finish_chart("03_revenue_by_category.png")

# Figure 4: Profit by category
plt.figure(figsize=(9, 6))
sns.barplot(color="steelblue", x=category_profit.index, y=category_profit.values)
plt.title("Profit by Product Category", fontsize=15, fontweight="bold")
plt.xlabel("Category")
plt.ylabel("Profit (₹)")
plt.xticks(rotation=20)
finish_chart("04_profit_by_category.png")

# Figure 5: Revenue by region
plt.figure(figsize=(9, 6))
sns.barplot(color="steelblue", x=region_sales.index, y=region_sales.values)
plt.title("Revenue by Region", fontsize=15, fontweight="bold")
plt.xlabel("Region")
plt.ylabel("Revenue (₹)")
finish_chart("05_revenue_by_region.png")

# Figure 6: Top 10 cities by revenue
plt.figure(figsize=(10, 6))
sns.barplot(color="steelblue", x=city_sales.values, y=city_sales.index)
plt.title("Top 10 Cities by Revenue", fontsize=15, fontweight="bold")
plt.xlabel("Revenue (₹)")
plt.ylabel("City")
finish_chart("06_top_cities.png")

# Figure 7: Revenue by customer segment
plt.figure(figsize=(9, 6))
sns.barplot(color="steelblue", x=customer_sales.index, y=customer_sales.values)
plt.title("Revenue by Customer Segment", fontsize=15, fontweight="bold")
plt.xlabel("Customer Segment")
plt.ylabel("Revenue (₹)")
finish_chart("07_customer_segments.png")

# Figure 8: Orders by payment method
plt.figure(figsize=(10, 6))
sns.barplot(color="steelblue", x=payment_methods.index, y=payment_methods.values)
plt.title("Orders by Payment Method (Delivered Orders)", fontsize=15, fontweight="bold")
plt.xlabel("Payment Method")
plt.ylabel("Number of Orders")
plt.xticks(rotation=20)
finish_chart("08_payment_methods.png")

# Figure 9: Overall order status distribution
order_status = df["Order_Status"].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(order_status.values, labels=order_status.index, autopct="%1.1f%%",
        startangle=90)
plt.title("Order Status Distribution", fontsize=15, fontweight="bold")
finish_chart("09_order_status.png")

# Figure 10: Quantity vs revenue
plt.figure(figsize=(10, 6))
sns.scatterplot(data=sales, x="Quantity", y="Realized_Revenue", hue="Category")
plt.title("Quantity vs Revenue", fontsize=15, fontweight="bold")
plt.xlabel("Quantity Sold")
plt.ylabel("Realised Revenue (₹)")
finish_chart("10_quantity_vs_revenue.png")

# Figure 11: Discount vs revenue
plt.figure(figsize=(10, 6))
sns.scatterplot(data=sales, x="Discount", y="Realized_Revenue", hue="Category")
plt.title("Discount vs Revenue", fontsize=15, fontweight="bold")
plt.xlabel("Discount Rate")
plt.ylabel("Realised Revenue (₹)")
finish_chart("11_discount_vs_revenue.png")

# Figure 12: Revenue distribution
plt.figure(figsize=(10, 6))
sns.histplot(sales["Realized_Revenue"], bins=30, kde=True, color="steelblue")
plt.title("Revenue Distribution", fontsize=15, fontweight="bold")
plt.xlabel("Realised Revenue per Order (₹)")
plt.ylabel("Number of Orders")
finish_chart("12_revenue_distribution.png")

# Figure 13: Correlation heatmap
numeric_data = sales[
    ["Quantity", "Unit_Price", "Discount", "Revenue", "Cost", "Profit"]
]
plt.figure(figsize=(10, 7))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap", fontsize=15, fontweight="bold")
finish_chart("13_correlation_heatmap.png")

# Figure 14: Monthly revenue by category
monthly_category = sales.groupby(
    [sales["Order_Date"].dt.to_period("M"), "Category"]
)["Realized_Revenue"].sum().unstack()
plt.figure(figsize=(13, 7))
for category in monthly_category.columns:
    plt.plot(monthly_category.index.astype(str), monthly_category[category],
             marker="o", label=category)
plt.title("Monthly Revenue by Category", fontsize=15, fontweight="bold")
plt.xlabel("Month")
plt.ylabel("Revenue (₹)")
plt.xticks(rotation=60)
plt.legend()
finish_chart("14_monthly_category_revenue.png")

# Figure 15: Profit margin by category
plt.figure(figsize=(9, 6))
sns.barplot(color="steelblue", x=category_data.index, y=category_data["Profit_Margin_Percent"])
plt.title("Profit Margin by Category", fontsize=15, fontweight="bold")
plt.xlabel("Category")
plt.ylabel("Profit Margin (%)")
plt.xticks(rotation=20)
finish_chart("15_profit_margin_category.png")

# Figure 16: Revenue distribution by category (box plot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=sales, x="Category", y="Realized_Revenue", color="steelblue")
plt.title("Revenue Spread by Category", fontsize=15, fontweight="bold")
plt.xlabel("Category")
plt.ylabel("Realised Revenue per Order (₹)")
plt.xticks(rotation=20)
finish_chart("16_revenue_boxplot.png")

# Figure 17: Revenue distribution by customer segment (violin plot)
plt.figure(figsize=(10, 6))
sns.violinplot(data=sales, x="Customer_Segment", y="Realized_Revenue",
               inner="quartile", cut=0, color="steelblue")
plt.title("Revenue Distribution by Customer Segment", fontsize=15, fontweight="bold")
plt.xlabel("Customer Segment")
plt.ylabel("Realised Revenue per Order (₹)")
finish_chart("17_revenue_violinplot.png")

# Figure 18: Number of delivered orders by region
plt.figure(figsize=(9, 6))
sns.countplot(data=sales, x="Region", order=sales["Region"].value_counts().index, color="steelblue")
plt.title("Number of Delivered Orders by Region", fontsize=15, fontweight="bold")
plt.xlabel("Region")
plt.ylabel("Number of Delivered Orders")
finish_chart("18_orders_by_region.png")

# Figure 19: Pair plot of selected numeric variables.
# Sampling keeps the pair plot readable and quick to render.
pair_data = sales[
    ["Quantity", "Unit_Price", "Discount", "Realized_Revenue", "Realized_Profit"]
].sample(n=min(700, len(sales)), random_state=42)
pair_grid = sns.pairplot(pair_data, corner=True, diag_kind="hist",
                         diag_kws={"color": "steelblue"},
                         plot_kws={"alpha": 0.45, "s": 14})
pair_grid.fig.suptitle("Pair Plot of Selected Numeric Variables", y=1.02,
                       fontsize=15, fontweight="bold")
pair_grid.fig.tight_layout()
pair_grid.fig.savefig(OUTPUT_DIR / "19_numeric_pairplot.png",
                      dpi=160, bbox_inches="tight")
plt.show()
plt.close(pair_grid.fig)
chart_count += 1

# ----------------------------------------------------------
# 6. BUSINESS INSIGHTS
# ----------------------------------------------------------
best_product = top_products.index[0]
best_category = category_revenue.index[0]
best_region = region_sales.index[0]
best_city = city_sales.index[0]
best_customer = customer_sales.index[0]
best_payment = payment_methods.index[0]

print("\nBUSINESS INSIGHTS")
print("-" * 60)
print("Top product by revenue:", best_product)
print("Top category by revenue:", best_category)
print("Top region by revenue:", best_region)
print("Top city by revenue:", best_city)
print("Customer segment with highest revenue:", best_customer)
print("Most used payment method among delivered orders:", best_payment)
print(f"Charts generated and saved: {chart_count}")
print("Chart folder:", OUTPUT_DIR)
print("\nAnalysis completed successfully.")
