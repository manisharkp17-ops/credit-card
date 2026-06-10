import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import plotly.express as px
import plotly.graph_objects as go

import kagglehub
import pandas as pd
import os

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
df = pd.read_csv(os.path.join(path, "creditcard.csv"))

print("Dataset Loaded Successfully")
print(df.shape)

print("Dataset Shape:", df.shape)
print(df.head())


print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())


df = df.drop_duplicates()


print("\nStatistical Summary")
print(df.describe())

print("\nFraud Distribution")
print(df["Class"].value_counts())

print("\nAverage Transaction Amount")
print(df["Amount"].mean())


plt.figure(figsize=(6,4))
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Normal Transactions")
plt.show()


plt.figure(figsize=(8,5))
plt.hist(df["Amount"], bins=50)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(8,5))
plt.hist(df["Time"], bins=50)
plt.title("Transaction Time Distribution")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(8,5))
sns.boxplot(x="Class", y="Amount", data=df)
plt.title("Amount vs Fraud")
plt.show()


plt.figure(figsize=(8,5))
plt.scatter(df["Amount"], df["Class"], alpha=0.3)
plt.title("Transaction Amount vs Fraud")
plt.xlabel("Amount")
plt.ylabel("Class")
plt.show()


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

def risk_level(amount):
    if amount < 100:
        return "Low Risk"
    elif amount < 1000:
        return "Medium Risk"
    else:
        return "High Risk"

df["Risk_Level"] = df["Amount"].apply(risk_level)

print(df["Risk_Level"].value_counts())

print(
    df.groupby("Class")["Amount"]
    .agg(["count", "mean", "min", "max"])
)

print(
    df.groupby("Risk_Level")["Amount"]
    .agg(["count", "mean"])
)



X = df[["Time"]]
y = df["Amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nR2 Score:", r2)
print("Mean Squared Error:", mse)


plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Amount")
plt.ylabel("Predicted Amount")
plt.title("Actual vs Predicted")
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8,5))
plt.hist(residuals, bins=50)
plt.title("Residual Distribution")
plt.show()



fig1 = px.histogram(
    df,
    x="Class",
    title="Fraud vs Normal Transactions"
)
fig1.show()


fig2 = px.histogram(
    df,
    x="Amount",
    nbins=50,
    title="Transaction Amount Distribution"
)
fig2.show()

fig3 = px.histogram(
    df,
    x="Risk_Level",
    title="Risk Level Distribution"
)
fig3.show()


fig4 = px.box(
    df,
    x="Class",
    y="Amount",
    title="Amount vs Fraud"
)
fig4.show()


fig5 = px.scatter(
    df,
    x="Time",
    y="Amount",
    color="Class",
    title="Time vs Amount"
)
fig5.show()


corr_matrix = df.corr(numeric_only=True)

fig6 = go.Figure(
    data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns
    )
)

fig6.update_layout(title="Correlation Heatmap")
fig6.show()


print("\nProject Completed Successfully")
print("EDA + Linear Regression + Dashboard Generated")
