import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
df = pd.read_csv("dataset/Data Export.csv")

# Show first 5 rows
print(df.head())

print(df.info())

print(df.head())

print(df.isnull().sum())

print(df['pollutant_id'].unique())

pivot_df = df.pivot_table(
    index=['city', 'station', 'last_update'],
    columns='pollutant_id',
    values='pollutant_avg'
).reset_index()

print(pivot_df.head())

pivot_df.fillna(
    pivot_df.mean(numeric_only=True),
    inplace=True
)

print(pivot_df.isnull().sum())

pivot_df.to_csv(
    "dataset/clean_aqi_data.csv",
    index=False
)

print("Clean dataset saved successfully!")

# Load cleaned dataset
clean_df = pd.read_csv("dataset/clean_aqi_data.csv")

print(clean_df.head())

# Correlation Heatmap
plt.figure(figsize=(12,8))

sns.heatmap(
    clean_df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Matrix")
plt.show()

# Histogram Distribution
clean_df.hist(
    figsize=(12,10),
    bins=20
)

plt.show()

# Boxplot for Outlier Detection
plt.figure(figsize=(12,6))

sns.boxplot(
    data=clean_df.select_dtypes(include='number')
)

plt.xticks(rotation=45)
plt.title("Outlier Detection")
plt.show()


# Feature Selection

X = clean_df[['CO', 'NH3', 'NO2', 'OZONE', 'PM10', 'SO2']]

y = clean_df['PM2.5']

print("Features:")
print(X.head())

print("Target:")
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

