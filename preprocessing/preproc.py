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

data = pd.read_csv("C:/Users/lilac/PycharmProjects/pythonProject1/dataset_diabetes/diabetic_data.csv")
dataframe = pd.DataFrame(data)
mappingFile22 = pd.read_csv("C:/Users/lilac/PycharmProjects/pythonProject1/dataset_diabetes/IDs_mapping.csv")
columnsToDrop = [  # write why we dropped here
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
mappingColumn1 = mappingFile22.iloc[0:8, [0, 1]]
mappingColumn2 = mappingFile22.iloc[10:40, [0, 1]]
mappingColumn3 = mappingFile22.iloc[42:67, [0, 1]]

dataframe.replace(['Unknown/Invalid', '?', 'NA', 'N/A', 'None', ''], np.nan, inplace=True)
columnsWithNull1 = [col for col in dataframe.columns if
                    dataframe[col].dtype in ['object'] and dataframe[col].isnull().any()]
columnsWithNull2 = [col for col in dataframe.columns if
                    dataframe[col].dtype in ['int64', 'float64'] and dataframe[col].isnull().any()]

dataframe[columnsWithNull1] = dataframe[columnsWithNull1].fillna(dataframe[columnsWithNull1].mode().iloc[0])
dataframe[columnsWithNull2] = dataframe[columnsWithNull2].apply(pd.to_numeric, errors='coerce')
dataframe[columnsWithNull2] = dataframe[columnsWithNull2].fillna(dataframe[columnsWithNull2].mean())


# عايزين نحول ال age لي num
def convert_age_range(age_range):
    numbers = age_range.strip("[]()").split('-')  # convert age range values with its median value
    return (int(numbers[0]) + int(numbers[1])) // 2


dataframe['age'] = dataframe['age'].apply(convert_age_range)
#print(dataframe.columns)
# encoding categorical columns
dataframe['readmitted'] = dataframe['readmitted'].apply(
    lambda x: 1 if x == '<30' else 0)  # encode target column into 0, 1
le = LabelEncoder()
#dataframe = dataframe.drop(
#columns=['diag_1', 'diag_2', 'diag_3'])  # drop these columns until finding a way to convert them as int
#categorical_cols = dataframe.select_dtypes(
 #   include='object').columns  # بحدد الكولم النوعا categoricalعشان اعمل ليهم encod


# target encoding
encoder = TargetEncoder(cols=['medical_specialty'])
encoder.fit(dataframe['medical_specialty'], dataframe['readmitted'])
dataframe['medical_specialty'] = encoder.transform(dataframe['medical_specialty'])
#print(dataframe['medical_specialty'].unique())
mean_encoded = dataframe.groupby('medical_specialty')['readmitted'].mean()
#print(mean_encoded)

# splitting data
#X = dataframe.drop('readmitted', axis=1)
#y = dataframe['readmitted']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #should include stratify

mappingColumn1.columns = ['admission_type_id', 'admission_type']
mappingColumn2.columns = ['discharge_disposition_id', 'discharge_disposition']
mappingColumn3.columns = ['admission_source_id', 'admission_source']

dataframe['admission_type_id'] = dataframe['admission_type_id'].astype(str)  # convert into string to apply join
dataframe['discharge_disposition_id'] = dataframe['discharge_disposition_id'].astype(str)
dataframe['admission_source_id'] = dataframe['admission_source_id'].astype(str)

#################### here we must first combine columns to mapp into categories

dataframe['admission_type_id'] = dataframe['admission_type_id'].replace({'6': '5', '8': '5'})
dataframe['discharge_disposition_id'] = dataframe['discharge_disposition_id'].replace({
    '6': '1', '8': '1',  # Home
    '30': '9', '27': '9', '29': '9',  # Short-Term Care
    '5': '4', '15': '4', '22': '4', '23': '4', '24': '4',  # Long-Term
    '19': '11', '20': '11', '21': '11',  # Expired
    '14': '13',  # Hospice
    '16': '12', '17': '12',  # Outpatient
    '25': '18', '26': '18'  # Unknown
})
# dataframe['discharge_disposition_id']=dataframe['discharge_disposition_id'].replace({})

dataframe = dataframe.merge(mappingColumn1, how='left', on='admission_type_id')
dataframe = dataframe.merge(mappingColumn2, how='left', on='discharge_disposition_id')
dataframe = dataframe.merge(mappingColumn3, how='left', on='admission_source_id')
mappedColumns = ['admission_type', 'discharge_disposition', 'admission_source']
#print(dataframe.columns.shape[0])
# dataframe[col]=dataframe[col].fillna(dataframe[col].mode().iloc[0])
# print(dataframe['admission_type'].unique())
columnsdrop = [
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id'
]
dataframe.drop(columns=columnsdrop, inplace=True)  # dropping columns with many constant or nulls
# hot encoding
dataframe = pd.get_dummies(dataframe, columns=['admission_type', 'discharge_disposition', 'admission_source'],
                           dtype=int)  # اول نجمعهم في محموعات وبعداك نسوي ال hot


# clipping outlires ######
numericalColumns = dataframe.select_dtypes(include=['int64', 'float64']).columns
Q1 = dataframe[numericalColumns].quantile(0.25)
Q3 = dataframe[numericalColumns].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
for col in numericalColumns:  ######
    dataframe[col] = dataframe[col].clip(lower=lower_bound[col],
                                         upper=upper_bound[col])  # clip normal data without the outliers
#dataframe.to_csv('processed_data.csv', index=False)

#################################################### end
print(dataframe.info())

# tasks to do:
# EDA analysis again to know relations between target and which columns to drop
# mapping other coulmns بعد ما نجمعهم مجموعات
# نحل مشكلة انو لو الكولم ارقام بقراهو سترينج
# نشوف وين مقروض نقسم الداتا : قبل ال encodig و بعد ال  mapping
# adding loop to fill nulls correctly
# Target encoding should always be done after splitting your data
