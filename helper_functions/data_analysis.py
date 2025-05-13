import matplotlib.pyplot as plt
from preprocessing.preproc import dataframe_copy
import numpy as np
from scipy.stats import pointbiserialr
import seaborn as sns
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
# List of specific columns to select
columns_to_select = ['time_in_hospital', 'num_lab_procedures',
                     'num_procedures', 'number_inpatient', 'number_diagnoses', 'num_medications', 'readmitted' ]

numeric_df = dataframe_copy[columns_to_select]
categorical_df = dataframe_copy.drop(columns=columns_to_select + ['readmitted'])
target = dataframe_copy['readmitted']

####### num features with eachother  ##########
# corr = numeric_df.corr(numeric_only=True)
# plt.figure(figsize=(12, 8))
#sns.heatmap(corr, cmap='coolwarm', annot=False)
# plt.title("Numerical features correlation with target")
# plt.show()
########num features with target ############


# Example data
from scipy.stats import pointbiserialr

# Ensure numeric_df and target have the same length
if len(numeric_df) == len(target):
    # Loop through each column in numeric_df
    for column in numeric_df.columns:
        numerical_feature = numeric_df[column].to_numpy().flatten()  # Convert to 1D array
        binary_target = target.to_numpy().flatten()  # Convert target to 1D array

        # Point-Biserial Correlation
        correlation, p_value = pointbiserialr(numerical_feature, binary_target)

        print(f"Point-Biserial Correlation for {column}: {correlation}")
        print(f"P-Value: {p_value}\n")
else:
    print("Error: The feature and target must have the same length.")

############# categorical features with target ################ using chi
#
# chi_selector = SelectKBest(chi2, k='all')
# chi_selector.fit(categorical_df, target)
# chi_scores = pd.Series(chi_selector.scores_, index=categorical_df.columns).sort_values(ascending=False)
# print("chi_score values:")
# print(chi_scores)
#
# ##############using mutual######################
# X = categorical_df
# y = dataframe_copy['readmitted']  # Your binary outcome
# # Calculate mutual information
# # Since your categorical variables are already encoded, we can specify which ones are discrete
# # Assuming all columns in X are categorical (already encoded)
# discrete_features = np.ones(len(X.columns), dtype=bool)  # All features are discrete/categorical
#
# # Calculate mutual information
# mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
#
# # Create a dataframe of scores
# mi_results = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi_scores})
# mi_results = mi_results.sort_values('Mutual Information', ascending=False)
#
# # Display the top features by mutual information
# print(mi_results)
#
# # Optionally, create a visualization
#
#
# plt.figure(figsize=(10, 8))
# plt.barh(mi_results['Feature'][:15], mi_results['Mutual Information'][:15])
# plt.xlabel('Mutual Information')
# plt.title('Top 15 categorical features by Mutual Information')
# plt.tight_layout()
# plt.show()
#
