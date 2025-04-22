import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
geoo
# Load and clean the data
df = pd.read_csv("D:\\crime.csv")
df.columns = df.columns.str.strip()  # Clean column names

# Data Inspection & Cleaning
print("Data Shape:", df.shape)
print("Data Info:")
df.info()
print("Summary Statistics:")
print(df.describe(include='all'))

# Remove Duplicates
df = df.drop_duplicates()

# NaN Count per Column
print("NaN Count per Column:")
print(df.isna().sum())

# Data Aggregation & Transformation
# Top Crime Types
top_crimes = df['Crm Cd Desc'].value_counts().head(10)
print("Top 10 Crime Types:")
print(top_crimes)

# Top Premises
top_premises = df['Premis Desc'].value_counts().head(10)
print("Top 10 Crime Premises:")
print(top_premises)

# Victim Demographics
print("Victim Sex Distribution:")
print(df['Vict Sex'].value_counts())

print("Victim Descent Distribution:")
print(df['Vict Descent'].value_counts())

print("Victim Age Stats:")
print(df['Vict Age'].describe())

# Weapon Usage Frequency
top_weapons = df['Weapon Desc'].value_counts().head(10)
print("Top Weapon Types:")
print(top_weapons)

# Crime Status Summary
print("Crime Status Summary:")
print(df['Status Desc'].value_counts())

# Visual Analysis
# Crime Type Distribution - Bar
plt.figure(figsize=(10,6))
plt.barh(top_crimes.index, top_crimes.values, color=sns.color_palette("Set2", len(top_crimes)))
plt.title("Top 10 Crime Types")
plt.xlabel("Count")
plt.gca().invert_yaxis()
plt.show()

# Crime Type Distribution - Pie
plt.figure(figsize=(8,8))
plt.pie(top_crimes.values, labels=top_crimes.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
plt.title("Top 10 Crime Types (Pie)")
plt.show()

# Age Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['Vict Age'].dropna(), bins=30, kde=True, color='skyblue')
plt.title("Victim Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Crimes by Victim Sex
plt.figure(figsize=(6,4))
sns.boxplot(x='Vict Sex', y='Vict Age', data=df, hue='Vict Sex', palette='pastel', legend=False)
plt.title("Crimes by Victim Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Scatter Map of LAT/LON
location_data = df[['LAT', 'LON']].dropna()
plt.figure(figsize=(8, 6))
sns.scatterplot(x='LON', y='LAT', data=location_data, alpha=0.6)
plt.title("Crime Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()


# Box Plot: Victim Age by Sex
plt.figure(figsize=(8, 6))
sns.boxplot(x='Vict Sex', y='Vict Age', data=df, hue='Vict Sex', palette='pastel', legend=False)
plt.title("Victim Age Distribution by Sex")
plt.xlabel("Sex")
plt.ylabel("Age")
plt.show()

# Violin Plot: Victim Age by Sex
plt.figure(figsize=(8, 6))
sns.violinplot(x='Vict Sex', y='Vict Age', data=df, hue='Vict Sex', palette='pastel', split=True)
plt.title("Victim Age Distribution by Sex (Violin Plot)")
plt.xlabel("Sex")
plt.ylabel("Age")
plt.show()



# Advanced Aggregations
# Most Dangerous Locations (by crime count)
dangerous_locations = df['LOCATION'].value_counts().head(10)
print("Most Dangerous Locations:")
print(dangerous_locations)

# Top Crimes Per Descent
top_crimes_per_descent = df.groupby('Vict Descent')['Crm Cd Desc'].agg(lambda x: x.value_counts().idxmax())
print("Top Crime Type per Victim Descent:")
print(top_crimes_per_descent)

# Cross-tab of Crime Type vs Weapon
crosstab = pd.crosstab(df['Crm Cd Desc'], df['Weapon Desc'])
print("Cross-tab: Crime Type vs Weapon")
print(crosstab.head(10))
