#!/usr/bin/env python
# coding: utf-8

# #### Import Libraries

# In[194]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# #### Read and Understand the Data

# In[195]:


train = pd.read_csv(r"C:\Users\AKHIL SURESH\Downloads\train_ctrUa4K.csv")


# In[196]:


train.head()


# In[197]:


train.shape


# In[198]:


train.info()


# In[199]:


train.isna().sum()


# In[200]:


train.describe()


# In[201]:


test = pd.read_csv(r"C:\Users\AKHIL SURESH\Downloads\test_lAUu6dG.csv")


# In[202]:


test.head()


# In[203]:


test.shape


# In[204]:


test.describe()


# In[205]:


test.isna().sum()


# #### Handle the missing values

# In[206]:


test_ids = test['Loan_ID'].copy()


# In[207]:


X = train.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = train['Loan_Status'].copy()


# In[208]:


X_test = test.drop('Loan_ID', axis=1)


# In[209]:


categorical_cols = ['Gender','Married','Dependents','Self_Employed']


# In[210]:


numerical_missing_cols = ['LoanAmount','Loan_Amount_Term','Credit_History']


# In[211]:


for col in categorical_cols:
    mode_value = X[col].mode()[0]
    X[col] = X[col].fillna(mode_value)
    X_test[col] = X_test[col].fillna(mode_value)


# In[212]:


for col in numerical_missing_cols:
    median_value = X[col].median()
    X[col] = X[col].fillna(median_value)
    X_test[col] = X_test[col].fillna(median_value)


# In[213]:


X.isna().sum()


# In[214]:


X_test.isna().sum()


# #### Encode

# In[215]:


from sklearn.preprocessing import LabelEncoder


# In[216]:


le = LabelEncoder()


# In[217]:


X['Education'] = le.fit_transform(X['Education'])
X_test['Education'] = le.transform(X_test['Education'])


# In[218]:


one_hot_cols = ['Gender','Married','Dependents','Self_Employed','Property_Area']


# In[219]:


X = pd.get_dummies(X, columns=one_hot_cols, dtype=int)


# In[220]:


X_test = pd.get_dummies(X_test, columns=one_hot_cols, dtype=int)


# In[221]:


X_test = X_test.reindex(columns=X.columns, fill_value=0)


# In[222]:


X.shape


# In[223]:


X_test.shape


# In[224]:


X.head()


# In[225]:


X_test.head()


# In[226]:


y = y.map({'Y': 1, 'N': 0})


# In[227]:


y.value_counts()


# #### Split Training Data

# In[228]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[229]:


scale_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']


# #### Scale the train val of train_data and test of test_data

# In[230]:


from sklearn.preprocessing import StandardScaler


# In[231]:


scaler = StandardScaler()


# In[232]:


X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()


# In[233]:


X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])


# In[234]:


X_val_scaled[scale_cols] = scaler.transform(X_val[scale_cols])


# In[235]:


X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])


# #### Logistic Regression

# In[236]:


from sklearn.linear_model import LogisticRegression


# In[237]:


model = LogisticRegression(max_iter=100)


# In[238]:


model.fit(X_train_scaled, y_train)


# In[239]:


y_pred = model.predict(X_val_scaled)


# In[240]:


confusion_matrix(y_val, y_pred)


# In[241]:


lr_accuracy = accuracy_score(y_val, y_pred)
lr_precision = precision_score(y_val, y_pred)
lr_recall = recall_score(y_val, y_pred)
lr_f1 = f1_score(y_val, y_pred)


# In[242]:


print("Metrics Report")
print("Accuracy: ", round(lr_accuracy, 3))
print("Precision: ", round(lr_precision, 3))
print("Recall score: ", round(lr_recall, 3))
print("F1 score: ", round(lr_f1, 3))


# In[243]:


from sklearn.metrics import classification_report


# In[244]:


print(classification_report(y_val, y_pred))


# #### Decision Tree

# In[245]:


from sklearn.tree import DecisionTreeClassifier


# In[246]:


dt = DecisionTreeClassifier(random_state=42)


# In[247]:


dt.fit(X_train_scaled, y_train)


# In[248]:


dt_pred = dt.predict(X_val_scaled)


# In[249]:


dt_accuracy = accuracy_score(y_val, dt_pred)
dt_precision = precision_score(y_val, dt_pred)
dt_recall = recall_score(y_val, dt_pred)
dt_f1 = f1_score(y_val, dt_pred)


# In[250]:


print("Metrics Report")
print("Accuracy: ", round(dt_accuracy, 3))
print("Precision: ", round(dt_precision, 3))
print("Recall score: ", round(dt_recall, 3))
print("F1 score: ", round(dt_f1, 3))


# In[251]:


print(classification_report(y_val, dt_pred))


# #### Random Forest

# In[252]:


from sklearn.ensemble import RandomForestClassifier


# In[253]:


rf = RandomForestClassifier(n_estimators=100, random_state=42)


# In[254]:


rf.fit(X_train_scaled, y_train)


# In[255]:


rf_pred = rf.predict(X_val_scaled)


# In[256]:


rf_accuracy = accuracy_score(y_val, rf_pred)
rf_precision = precision_score(y_val, rf_pred)
rf_recall = recall_score(y_val, rf_pred)
rf_f1 = f1_score(y_val, rf_pred)


# In[257]:


print("Random-Forest Metrics Report")
print("Accuracy: ", round(rf_accuracy, 3))
print("Precision: ", round(rf_precision, 3))
print("Recall score: ", round(rf_recall, 3))
print("F1 score: ", round(rf_f1, 3))


# In[258]:


print(classification_report(y_val,rf_pred))


# #### Predicting using the test dataset and transferring the predicted results to a csv file

# In[259]:


test_prediction = model.predict(X_test_scaled)


# In[260]:


test_prediction = np.where(test_prediction == 1,'Y','N')


# In[261]:


submission = pd.DataFrame({'Loan_ID': test_ids, 'Loan_Status': test_prediction})


# In[262]:


submission.head()


# In[263]:


submission.shape


# In[264]:


submission.to_csv('submission.csv',index=False)


# #### Model Comparison Report and Visualisation

# In[265]:


results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    ],

    "Accuracy": [
        lr_accuracy,
        dt_accuracy,
        rf_accuracy,
    ],

    "Precision": [
        lr_precision,
        dt_precision,
        rf_precision,
    ],

    "Recall Score": [
        lr_recall,
        dt_recall,
        rf_recall,
    ],

    "F1 Score": [
        lr_f1,
        dt_f1,
        rf_f1,
    ]
})

results


# In[266]:


plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
plt.bar(
    results["Model"],
    results["Accuracy"]
)
plt.title("Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=20)

plt.subplot(2,2,2)
plt.bar(
    results["Model"],
    results["Precision"]
)
plt.title("Precision Comparison")
plt.xlabel("Model")
plt.ylabel("Precision")
plt.ylim(0, 1)
plt.xticks(rotation=20)

plt.subplots_adjust(hspace=0.5)

plt.subplot(2,2,3)
plt.bar(
    results["Model"],
    results["Recall Score"]
)
plt.title("Recall Score Comparison")
plt.xlabel("Model")
plt.ylabel("Recall Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)

plt.subplot(2,2,4)
plt.bar(
    results["Model"],
    results["F1 Score"]
)
plt.title("F1 Score Comparison")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)

plt.show()

