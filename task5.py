import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  

# Load the dataset
df = pd.read_csv('C:\\Users\\Vasudev\\Downloads\\heart.csv')

# Display the first and last 5 rows
print(df.head())
print(df.tail())

# Display column names
print(df.columns.values)

# Check for missing values
print(df.isna().sum())

# Information about the dataset
print(df.info())

# Histogram of numerical values
df.hist(bins=50, grid=False, figsize=(20, 15))
plt.show()

# Statistical summary
print(df.describe())

# Count of people with and without heart disease
print(df['target'].value_counts())

# Bar chart of heart disease values
df['target'].value_counts().plot(kind='bar', color=["orchid", "salmon"])
plt.title("Heart Disease Values")
plt.xlabel("1 = Heart disease, 0 = No heart disease")
plt.ylabel("Count")
plt.show()

# Pie chart of heart disease values
df['target'].value_counts().plot(kind='pie', figsize=(8, 6), autopct='%1.1f%%', colors=["salmon", "orchid"])
plt.title("Heart Disease Proportion")
plt.show()

# Count of males and females
sex_counts = df['sex'].value_counts() # 0 is female, 1 is male
print(sex_counts)

# Pie chart of male-female ratio
sex_counts.plot(kind='pie', figsize=(8, 6), autopct='%1.1f%%', colors=["lightblue", "salmon"])
plt.legend(["Male", "Female"])
plt.title("Male-Female Ratio")
plt.show()

# Crosstab of heart disease by sex
heart_sex_crosstab = pd.crosstab(df['target'], df['sex'])
print(heart_sex_crosstab)

# Count plot of heart disease frequency by sex
sns.countplot(x='target', data=df, hue='sex')
plt.title("Heart Disease Frequency by Sex")
plt.xlabel("0 = No Heart Disease, 1 = Heart Disease")
plt.show()

# Counting values for different chest pain types
chest_pain_counts = df['cp'].value_counts()
print(chest_pain_counts)

# Bar chart of chest pain types
df['cp'].value_counts().plot(kind='bar', color=['salmon', 'lightskyblue', 'springgreen', 'khaki'])
plt.title('Chest Pain Type vs Count')
plt.show()

# Crosstab of sex by chest pain type
sex_cp_crosstab = pd.crosstab(df['sex'], df['cp'])
print(sex_cp_crosstab)

# Bar chart of chest pain type by sex
sex_cp_crosstab.plot(kind='bar', color=['coral', 'lightskyblue', 'plum', 'khaki'])
plt.title("Type of Chest Pain by Sex")
plt.xlabel('0 = Female, 1 = Male')
plt.show()

# Crosstab of chest pain type by heart disease
cp_target_crosstab = pd.crosstab(df['cp'], df['target'])
print(cp_target_crosstab)

# Count plot of chest pain type by heart disease
sns.countplot(x='cp', data=df, hue='target')
plt.title("Chest Pain Type vs Heart Disease")
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.show()

# Distribution plot for age with KDE
sns.displot(x='age', data=df, bins=30, kde=True)
plt.title("Age Distribution with KDE")
plt.show()

# Distribution plot for heart rate (thalach) with KDE
sns.displot(x='thalach', data=df, bins=30, kde=True, color='chocolate')
plt.title("Heart Rate (Thalach) Distribution with KDE")
plt.show()
