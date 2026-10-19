# Module 5 - Student Starter Code 

# This code is provided as a starting point for the assignment.
# You are expected to complete the code and implement the required functionality. 
# TODO: Import libraries for whatever you need to complete the assignment (the libraries below are for reference and are incomplete).
import numpy as np
import pandas as pd

# Import libraries for visualization from matplotlib and seaborn and inspection from sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: Import libraries for preprocessing from sklearn

# Import libraries for imputation from sklearn
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Import libraries for model model selection and metrics from sklearn
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# Import libraries for regression from sklearn
from sklearn.linear_model import LinearRegression

# Function to test dataset on a linear regression model
# Use this function to compare how different imputation methods affect the model performance
def get_score(df, imputer=None): 

       # Define features (X) and target (y)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    regressor = LinearRegression()
    if imputer is not None:
        estimator = make_pipeline(imputer, regressor)
    else:
        estimator = regressor
    scores = cross_val_score(
        estimator, X, y, scoring="neg_mean_squared_error", cv=4
    )
    return scores.mean(), scores.std()


#----------------------------------------------------------------
# Step 1: Load the CSV file into a pandas DataFrame
#----------------------------------------------------------------
from pathlib import Path

folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / "Module5Dataset.csv.csv") 

#----------------------------------------------------------------
# Step 2: Data Exploration and Visualization
#         Does the data contain NAN or nan values (implying incomplete data)?
#         What type of data is in the dataset?
#----------------------------------------------------------------

    # TODO: Display information about the dataset
print("\nFirst four samples:")
print(df.head(4).to_string(index=False))

print("\nNumber of samples:", df.shape[0])
print("Number of input features:", df.shape[1] - 1)

print("\nDataset information:")
df.info()

print("\nCount, unique, top, and frequency:")
print(df.describe(include="all").to_string())

print("\nMissing values:")
print(df.isna().sum())
    # TODO: Dispay the unique values in each feature

    # TODO: Visualize the distribution of 'Features'
print("\nUnique values:")
for column in df.columns:
    print(f"{column}: {df[column].unique()}")

# Visualize the distribution of features and class
fig, axes = plt.subplots(3, 3, figsize=(12, 8))

for ax, column in zip(axes.flat, df.columns):
    counts = df[column].fillna("Missing").value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")

# Hide unused panels
for ax in axes.flat[len(df.columns):]:
    ax.set_visible(False)

plt.tight_layout(h_pad=3, w_pad=2)
plt.show()


#----------------------------------------------------------------
# Step 3: Attempt to score the raw dataset before preprocessing

print("\nMODEL SCORE BEFORE PREPROCESSING")

try:
    raw_mean, raw_std = get_score(df)
    print("Mean negative MSE:", raw_mean)
    print("Standard deviation:", raw_std)
except ValueError:
    print("Raw dataset could not be scored.")
    print("The data contains text categories and missing values.")
    print("LinearRegression requires numeric inputs without missing values.")

x_labels = []
mses = []
stds = []
#----------------------------------------------------------------
# Step 4: Transform categories into numeric values

category_maps = {
    "Feature1": {"low": 0, "med": 1, "high": 2, "vhigh": 3},
    "Feature2": {"low": 0, "med": 1, "high": 2, "vhigh": 3},
    "Feature3": {"2": 0, "3": 1, "4": 2, "5more": 3},
    "Feature4": {"2": 0, "4": 1, "more": 2},
    "Feature5": {"small": 0, "med": 1, "big": 2},
    "Feature6": {"low": 0, "med": 1, "high": 2},
    "Class": {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
}

df_numeric = df.copy()

for column, mapping in category_maps.items():
    df_numeric[column] = df[column].map(mapping)

print("\nTRANSFORMED DATA: FIRST FOUR SAMPLES")
print(df_numeric.head(4).to_string(index=False))

print("\nMISSING VALUES AFTER TRANSFORMATION")
print(df_numeric.isna().sum())

# Generate the correlation matrix before imputation
plt.figure(figsize=(9, 6))
sns.heatmap(
    df_numeric.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)
plt.title("Correlation Matrix Before Imputation")
plt.tight_layout()
plt.show()
#----------------------------------------------------------------

#----------------------------------------------------------------
# Step 5: Compare model scores before and after imputation

print("\nNUMERIC DATA WITHOUT IMPUTATION")

try:
    mean_score, std_score = get_score(df_numeric)
    print("Mean negative MSE:", mean_score)
    print("Standard deviation:", std_score)
except ValueError:
    print("Cannot score: numeric input features still contain NaN values.")

# Reset the results lists
x_labels = []
mses = []
stds = []

# Imputers are fitted inside each cross-validation training fold
imputation_methods = [
    ("Zero Imputation",
     SimpleImputer(strategy="constant", fill_value=0)),
    ("Mean Imputation",
     SimpleImputer(strategy="mean")),
    ("KNN Imputation",
     KNNImputer()),
    ("Iterative Imputation",
     IterativeImputer(max_iter=10, random_state=42))
]

for name, imputer in imputation_methods:
    mean_score, std_score = get_score(df_numeric, imputer)

    x_labels.append(name)
    mses.append(mean_score)
    stds.append(std_score)

results = pd.DataFrame({
    "Method": x_labels,
    "Mean Negative MSE": mses,
    "Standard Deviation": stds
})

print("\nIMPUTATION COMPARISON: FOUR-FOLD CROSS-VALIDATION")
print(results.to_string(index=False))
print("\nA negative MSE closer to zero indicates better performance.")

#----------------------------------------------------------------
# Step 6: Visualize the imputed data and compare model scores

X = df_numeric.drop(columns=["Class"])
imputed_datasets = {}

# Fit on the full dataset for descriptive visualizations only.
# Model scores above use imputation within each training fold.
for name, imputer in imputation_methods:
    filled = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    imputed_datasets[name] = filled

    print(f"\n{name}: missing values remaining")
    print(filled.isna().sum().to_string())

# Compare feature distributions after each imputation method
fig, axes = plt.subplots(4, 6, figsize=(16, 10))

for row, (name, filled) in enumerate(imputed_datasets.items()):
    for col, feature in enumerate(X.columns):
        ax = axes[row, col]
        ax.hist(
            filled[feature],
            bins=np.linspace(-0.5, 3.5, 17),
            color="steelblue",
            edgecolor="white"
        )
        if row == 0:
            ax.set_title(feature)

        if col == 0:
            ax.set_ylabel(f"{name}\nCount")

        if row == 3:
            ax.set_xlabel("Encoded value")

fig.suptitle("Feature Distributions After Imputation", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Compare correlations after imputation
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, (name, filled) in zip(axes.flat, imputed_datasets.items()):
    plot_data = filled.copy()
    plot_data["Class"] = df_numeric["Class"]

    sns.heatmap(
        plot_data.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title(name)

plt.tight_layout()
plt.show()

# Plot mean scores with standard deviations as error bars
plt.figure(figsize=(10, 6))
plt.bar(x_labels, mses, yerr=stds, capsize=5, color="steelblue")
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Imputation Method")
plt.ylabel("Mean Cross-Validation Negative MSE")
plt.title("Imputation Comparison: Scores Closer to Zero Are Better")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()