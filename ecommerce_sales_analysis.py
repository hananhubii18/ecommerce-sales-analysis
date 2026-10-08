import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# E-COMMERCE SALES DATA ANALYSIS
# ==========================================

# Load dataset
df = pd.read_csv("ecommerce_sales_2025_2026.csv")

# Convert date column
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Select delivered orders
sales = df[df["Order_Status"] == "Delivered"]


# ==========================================
# 1. BASIC INFORMATION
# ==========================================

print("\n======================================")
print("      E-COMMERCE SALES ANALYSIS")
print("======================================")

print("\nDataset Information")
print("-------------------")

print("Total Orders:", len(df))
print("Delivered Orders:", len(sales))
print("Total Products Sold:", sales["Quantity"].sum())

print("Start Date:", df["Order_Date"].min().date())
print("End Date:", df["Order_Date"].max().date())


# ==========================================
# 2. SALES CALCULATIONS
# ==========================================

total_revenue = sales["Realized_Revenue"].sum()

total_profit = sales["Realized_Profit"].sum()

total_quantity = sales["Quantity"].sum()

average_order_value = total_revenue / len(sales)

profit_margin = (total_profit / total_revenue) * 100


print("\nKey Performance Indicators")
print("--------------------------")

print("Total Revenue: ₹", round(total_revenue, 2))

print("Total Profit: ₹", round(total_profit, 2))

print("Products Sold:", total_quantity)

print("Average Order Value: ₹",
      round(average_order_value, 2))

print("Profit Margin:",
      round(profit_margin, 2), "%")


# ==========================================
# 3. MONTHLY REVENUE
# ==========================================

monthly_revenue = sales.groupby(
    sales["Order_Date"].dt.to_period("M")
)["Realized_Revenue"].sum()


plt.figure(figsize=(12, 6))

plt.plot(
    monthly_revenue.index.astype(str),
    monthly_revenue.values,
    marker="o",
    linewidth=2
)

plt.title(
    "Monthly Revenue Trend",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Month")
plt.ylabel("Revenue (₹)")

plt.xticks(rotation=60)

plt.grid(True)

plt.tight_layout()

plt.show()


# ==========================================
# 4. TOP 10 PRODUCTS
# ==========================================

top_products = sales.groupby(
    "Product"
)["Realized_Revenue"].sum()

top_products = top_products.sort_values(
    ascending=False
).head(10)


plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_products.values,
    y=top_products.index
)

plt.title(
    "Top 10 Products by Revenue",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Revenue (₹)")
plt.ylabel("Product")

plt.tight_layout()

plt.show()


# ==========================================
# 5. REVENUE BY CATEGORY
# ==========================================

category_revenue = sales.groupby(
    "Category"
)["Realized_Revenue"].sum()

category_revenue = category_revenue.sort_values(
    ascending=False
)


plt.figure(figsize=(9, 6))

sns.barplot(
    x=category_revenue.index,
    y=category_revenue.values
)

plt.title(
    "Revenue by Product Category",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Category")
plt.ylabel("Revenue (₹)")

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()


# ==========================================
# 6. PROFIT BY CATEGORY
# ==========================================

category_profit = sales.groupby(
    "Category"
)["Realized_Profit"].sum()

category_profit = category_profit.sort_values(
    ascending=False
)


plt.figure(figsize=(9, 6))

sns.barplot(
    x=category_profit.index,
    y=category_profit.values
)

plt.title(
    "Profit by Product Category",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Category")
plt.ylabel("Profit (₹)")

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()


# ==========================================
# 7. SALES BY REGION
# ==========================================

region_sales = sales.groupby(
    "Region"
)["Realized_Revenue"].sum()

region_sales = region_sales.sort_values(
    ascending=False
)


plt.figure(figsize=(9, 6))

sns.barplot(
    x=region_sales.index,
    y=region_sales.values
)

plt.title(
    "Revenue by Region",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Region")
plt.ylabel("Revenue (₹)")

plt.tight_layout()

plt.show()


# ==========================================
# 8. TOP 10 CITIES
# ==========================================

city_sales = sales.groupby(
    "City"
)["Realized_Revenue"].sum()

city_sales = city_sales.sort_values(
    ascending=False
).head(10)


plt.figure(figsize=(10, 6))

sns.barplot(
    x=city_sales.values,
    y=city_sales.index
)

plt.title(
    "Top 10 Cities by Revenue",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Revenue (₹)")
plt.ylabel("City")

plt.tight_layout()

plt.show()


# ==========================================
# 9. CUSTOMER SEGMENTS
# ==========================================

customer_sales = sales.groupby(
    "Customer_Segment"
)["Realized_Revenue"].sum()

customer_sales = customer_sales.sort_values(
    ascending=False
)


plt.figure(figsize=(9, 6))

sns.barplot(
    x=customer_sales.index,
    y=customer_sales.values
)

plt.title(
    "Revenue by Customer Segment",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Customer Segment")
plt.ylabel("Revenue (₹)")

plt.tight_layout()

plt.show()


# ==========================================
# 10. PAYMENT METHODS
# ==========================================

payment_methods = sales[
    "Payment_Method"
].value_counts()


plt.figure(figsize=(10, 6))

sns.barplot(
    x=payment_methods.index,
    y=payment_methods.values
)

plt.title(
    "Orders by Payment Method",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Payment Method")
plt.ylabel("Number of Orders")

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()


# ==========================================
# 11. ORDER STATUS
# ==========================================

order_status = df[
    "Order_Status"
].value_counts()


plt.figure(figsize=(8, 6))

plt.pie(
    order_status.values,
    labels=order_status.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title(
    "Order Status Distribution",
    fontsize=16,
    fontweight="bold"
)

plt.show()


# ==========================================
# 12. QUANTITY VS REVENUE
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=sales,
    x="Quantity",
    y="Realized_Revenue",
    hue="Category"
)

plt.title(
    "Quantity vs Revenue",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Quantity Sold")
plt.ylabel("Revenue (₹)")

plt.tight_layout()

plt.show()


# ==========================================
# 13. DISCOUNT VS REVENUE
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=sales,
    x="Discount",
    y="Realized_Revenue",
    hue="Category"
)

plt.title(
    "Discount vs Revenue",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Discount")
plt.ylabel("Revenue (₹)")

plt.tight_layout()

plt.show()


# ==========================================
# 14. REVENUE DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    sales["Realized_Revenue"],
    bins=30,
    kde=True
)

plt.title(
    "Revenue Distribution",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Revenue per Order (₹)")
plt.ylabel("Number of Orders")

plt.tight_layout()

plt.show()


# ==========================================
# 15. CORRELATION HEATMAP
# ==========================================

numeric_data = sales[
    [
        "Quantity",
        "Unit_Price",
        "Discount",
        "Revenue",
        "Cost",
        "Profit"
    ]
]


plt.figure(figsize=(10, 7))

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Correlation Heatmap",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.show()


# ==========================================
# 16. MONTHLY CATEGORY SALES
# ==========================================

monthly_category = sales.groupby(
    [
        sales["Order_Date"].dt.to_period("M"),
        "Category"
    ]
)["Realized_Revenue"].sum().unstack()


plt.figure(figsize=(13, 7))

for category in monthly_category.columns:

    plt.plot(
        monthly_category.index.astype(str),
        monthly_category[category],
        marker="o",
        label=category
    )


plt.title(
    "Monthly Revenue by Category",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Month")
plt.ylabel("Revenue (₹)")

plt.xticks(rotation=60)

plt.legend()

plt.tight_layout()

plt.show()


# ==========================================
# 17. PROFIT MARGIN BY CATEGORY
# ==========================================

category_data = sales.groupby(
    "Category"
).agg(
    Revenue=("Realized_Revenue", "sum"),
    Profit=("Realized_Profit", "sum")
)

category_data["Profit_Margin"] = (
    category_data["Profit"] /
    category_data["Revenue"]
) * 100


plt.figure(figsize=(9, 6))

sns.barplot(
    x=category_data.index,
    y=category_data["Profit_Margin"]
)

plt.title(
    "Profit Margin by Category",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Category")
plt.ylabel("Profit Margin (%)")

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()


# ==========================================
# 18. FINAL BUSINESS INSIGHTS
# ==========================================

best_product = top_products.index[0]

best_category = category_revenue.index[0]

best_region = region_sales.index[0]

best_city = city_sales.index[0]

best_customer = customer_sales.index[0]

best_payment = payment_methods.index[0]


print("\n======================================")
print("          BUSINESS INSIGHTS")
print("======================================")

print("Best Selling Product :", best_product)

print("Best Category        :", best_category)

print("Best Region          :", best_region)

print("Best City            :", best_city)

print("Best Customer Segment:", best_customer)

print("Most Used Payment    :", best_payment)

print("\nAnalysis Completed Successfully!")

print("Total Graphs Created: 18")