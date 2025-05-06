import string
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings as wr

data = pd.read_csv("C:/Users/DELL/University/Ai Project/dataset_diabetes/diabetic_data.csv")
mappingFile22 = pd.read_csv("C:/Users/DELL/University/Ai Project/dataset_diabetes/IDs_mapping.csv")
dataframe = pd.DataFrame(data)
columnsToDrop=[
    'metformin-pioglitazone',
    'metformin-rosiglitazone',
    'glipizide-metformin',
    'glimepiride-pioglitazone',
    'examide',
    'citoglipton',
    'max_glu_serum',
    'patient_nbr',
    'A1Cresult', 'weight',
    'encounter_id', 'number_outpatient',
    'number_emergency'
]
dataframe.drop(columns=columnsToDrop, axis=1, inplace=True)

# converting age column as numerical
def convert_age_range(age_range):
    numbers = age_range.strip("[]()").split('-') # convert age range values with its meadian value
    return (int(numbers[0]) + int(numbers[1])) // 2
dataframe['age'] = dataframe['age'].apply(convert_age_range)

# converting columns with unkown values into null
columnToConvert=['diag_1','diag_2', 'diag_3']
dataframe[columnToConvert] = dataframe[columnToConvert].apply(
    lambda col: pd.to_numeric(col, errors='coerce'))

# filling categorical nulls with mode
dataframe.replace(['Unknown/Invalid','?', 'NA', 'N/A', 'None', ''], np.nan, inplace=True)
strNullColumns = [col for col in dataframe.columns if dataframe[col].dtype in ['object'] and dataframe[col].isnull().any()]
dataframe[strNullColumns]=dataframe[strNullColumns].fillna(dataframe[strNullColumns].mode().iloc[0])

# mapping columns
mappingColumn1 = mappingFile22.iloc[0:8, [0,1]] # reading rows from mapping
mappingColumn2 = mappingFile22.iloc[10:40 , [0,1]]
mappingColumn3 = mappingFile22.iloc[42:67 , [0,1]]

mappingColumn1.columns = ['admission_type_id', 'admission_type']
mappingColumn2.columns = ['discharge_disposition_id', 'discharge_disposition']
mappingColumn3.columns = ['admission_source_id', 'admission_source']

dataframe['admission_type_id'] = dataframe['admission_type_id'].astype(str) # convert into string to apply join
dataframe['discharge_disposition_id'] = dataframe['discharge_disposition_id'].astype(str)
dataframe['admission_source_id'] = dataframe['admission_source_id'].astype(str)

admission_groups={'6': '5', '8': '5'}
disposition_groups={
    '6': '1', '8': '1',  # Home
    '2': '9', '30': '9','10': '9', '27': '9', '29': '9',  # Short-Term Care
    '5': '4', '15': '4', '22': '4', '23': '4', '24': '4',  # Long-Term
    '19': '11', '20': '11', '21': '11',  # Expired
    '14': '13',  # Hospice
    '16': '12', '17': '12',  # Outpatient
    '25': '26', '18': '26'  # Unknown
}
source_groups={
    '2': '1', '3': '1',                  # Referral
    '10': '4', '22': '4',                # Hospital Transfer
    '5': '6', '18': '6', '19': '6', '25': '6', '26': '6',  # Facility Transfer
    '12': '11', '13': '11', '14': '11', '23': '11', '24': '11',  # Birth-related → map to '11'
    '15': '9', '17': '9', '20': '9', '21': '9'   # Unknown
}
dataframe['admission_type_id'] = dataframe['admission_type_id'].replace(admission_groups) # grouping columns with common feature
dataframe['discharge_disposition_id']=dataframe['discharge_disposition_id'].replace(disposition_groups)
dataframe['admission_source_id']=dataframe['admission_source_id'].replace(source_groups)

dataframe = dataframe.merge(mappingColumn1, how='left', on='admission_type_id') # applying join using id column
dataframe = dataframe.merge(mappingColumn2, how='left', on='discharge_disposition_id')
dataframe = dataframe.merge(mappingColumn3, how='left', on='admission_source_id')
mappedColumns = ['admission_type', 'discharge_disposition','admission_source']
columnsdrop = [
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id'
]
dataframe.drop(columns=columnsdrop, inplace=True)

# encoding categorical columns
# encoding target column
dataframe['readmitted'] = dataframe['readmitted'].apply(lambda x: 1 if x == '<30' else 0) # encoding target column
y = dataframe['readmitted']

# one-hot encoding for mapped columns & renaming them
dataframe=pd.get_dummies(dataframe,columns=mappedColumns,dtype=int)
column_renames = {
    'discharge_disposition_Discharged to home': 'discharge_home',
    'discharge_disposition_Discharged/transferred to SNF': 'discharge_snf',
    'discharge_disposition_Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital': 'discharge_psych_hosp',
    'discharge_disposition_Discharged/transferred to ICF':'discharge_ICF',
    'discharge_disposition_Expired': 'discharge_expired',
    'discharge_disposition_Hospice / home': 'discharge_hospice',
    'discharge_disposition_Left AMA': 'discharge_left_ama',
    'discharge_disposition_Short-Term discharge': 'discharge_short_term',
    'discharge_disposition_Admitted as an inpatient to this hospital':'discharge_inpatiesnt',
    'discharge_disposition_Unknown/Invalid': 'discharge_unknown',
    'discharge_disposition_Still patient or expected to return for outpatient services':'discharge_expected_outpatient',
    'discharge_disposition_expected_outpatient': 'discharge_outpatient',
    'discharge_disposition_long-Term discharge': 'discharge_long_term',

    'admission_source_ Court/Law Enforcement': 'admit_law',
    'admission_source_ Emergency Room': 'admit_Emroom',
    'admission_source_ Not Available': 'admit_notAvailable',
    'admission_source_Normal Delivery':'admit_birth_related',
    'admission_source_ Physician Referral': 'admit_physician',
    'admission_source_ Transfer from another health care facility': 'admit_transfer_other',
    'admission_source_Birth-related': 'admit_birth_related',
    'admission_source_Transfer from a hospital': 'admit_hosp_transfer'
}
dataframe.rename(columns=column_renames, inplace=True)

# encoding rest of categorical columns
categorical_cols = dataframe.select_dtypes(include='object').columns
categorical_cols= dataframe[categorical_cols].drop('medical_specialty', axis=1)
le = LabelEncoder()
for col in categorical_cols:
 dataframe[col] = le.fit_transform(dataframe[col])

# clipping outliers
dataframe=dataframe.drop('readmitted', axis=1)
numericalColumns = dataframe.select_dtypes(include=['int64','float64']).columns
Q1 = dataframe[numericalColumns].quantile(0.25)
Q3 = dataframe[numericalColumns].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
for col in numericalColumns:
    dataframe[col] = dataframe[col].clip(lower=lower_bound[col], upper=upper_bound[col]) # clip normal data without the outliers

# splitting data
X = dataframe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# filling numerical columns nulls with mean
numNullColumns = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
X_train[numNullColumns]=X_train[numNullColumns].apply(pd.to_numeric, errors='coerce')
X_train[numNullColumns]=X_train[numNullColumns].fillna(X_train[numNullColumns].mean())
X_test[numNullColumns]=X_test[numNullColumns].fillna(X_train[numNullColumns].mean())

# target encoding for medical_specialty column
encoder=TargetEncoder(cols=['medical_specialty'])
encoder.fit(X_train[['medical_specialty']],y_train) # calculate target means with train set & target column
X_train['medical_specialty']= encoder.transform(X_train[['medical_specialty']]) # applying encoding using .fit means value
X_test['medical_specialty']= encoder.transform(X_test[['medical_specialty']])
encoded_med_specialty = pd.concat([X_train['medical_specialty'], X_test['medical_specialty']]).sort_index()
dataframe['medical_specialty'] = encoded_med_specialty # load the new value of encoded column back into dataframe

print(X_train.dtypes)
print(y_train.dtypes)
print(dataframe.columns.shape[0])
print(dataframe.columns)
