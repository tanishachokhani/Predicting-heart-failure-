# Predicting-heart-failure-
# Importing libraries and Raw data 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
df= pd.read_csv('heart_failure_clinical_records_dataset-2.csv')
df.head()

# Profiling of variables
import pandas_profiling as pp
pp.ProfileReport(df)


# Finding Missing Values 
msno.matrix(df)

![image](https://user-images.githubusercontent.com/82114061/118257297-185ce000-b4cc-11eb-9230-c9f937df319e.png)

# Replacing missing Values with column means 
df.anaemia = df.anaemia.fillna('1')
df['creatinine_phosphokinase'].median()
df.creatinine_phosphokinase = df.creatinine_phosphokinase.fillna(250)
df.diabetes = df.diabetes.fillna('1')
df['ejection_fraction'].median()
df.ejection_fraction = df.ejection_fraction.fillna(38)
df['platelets'].median()
df.platelets = df.platelets.fillna(259500.0)
df['serum_sodium'].median()
df.serum_sodium = df.serum_sodium.fillna(137.0)
df.smoking = df.smoking.fillna('No')
df.head()

# Dummy Coding
df['sex'].replace('Male',1)
df['sex'] = np.where(df['sex'].str.contains('Male'), 1, 0)
df['smoking'] = np.where(df['smoking'].str.contains('Yes I smoke'), 1, 0)
df['DEATH_EVENT'] = np.where(df['DEATH_EVENT'].str.contains("Death"), 1, 0)
df.head()


# Counting Values for Data Exploration visualizations 
df1 = df.groupby(["sex","diabetes","smoking","anaemia","high_blood_pressure","DEATH_EVENT"])["age"].count().reset_index()
df1.columns= ["sex","diabetes","smoking","anaemia","high_blood_pressure","DEATH_EVENT", "count"]


# Descriptive statistics-Plotting a sun burst chart 
df1.loc[df1["sex"]== 0 , "sex"] = "female"
df1.loc[df1["sex"]== 1, "sex"] = "male"

df1.loc[df1["diabetes"]== 0 , "diabetes"] = "no diabetes"
df1.loc[df1["diabetes"]== 1, "diabetes"] = "diabetes"

df1.loc[df1['DEATH_EVENT'] == 0,'DEATH_EVENT'] = "LIVE"
df1.loc[df1['DEATH_EVENT'] == 1, 'DEATH_EVENT'] = 'DEATH'

fig = px.sunburst(df1, 
                  path=["sex","diabetes","DEATH_EVENT"],
                  values="count",
                  title="Gender & Diabetes Sunburst Chart ",
                  width=600,
                  height=600)

fig.show()
df1.loc[df1["smoking"]== 0 , "smoking"] = "non smoking"
df1.loc[df1["smoking"]== 1, "smoking"] = "smoking"

fig = px.sunburst(df1, 
                  path=["sex","smoking","DEATH_EVENT"],
                  values="count",
                  title="Gender & Smoking Sunburst Chart ",
                  width=600,
                  height=600)

fig.show()
df1.loc[df1["anaemia"]== 0 , "anaemia"] = "no anaemia"
df1.loc[df1["anaemia"]== 1, "anaemia"] = "anaemia"

fig = px.sunburst(df1, 
                  path=["sex","anaemia","DEATH_EVENT"],
                  values="count",
                  title="Gender & Anaemia  Sunburst Chart ",
                  width=600,
                  height=600)

fig.show()
df1.loc[df1["high_blood_pressure"]== 0 , "high_blood_pressure"] = "no high_blood_pressure"
df1.loc[df1["high_blood_pressure"]== 1, "high_blood_pressure"] = "high_blood_pressure"

fig = px.sunburst(df1, 
                  path=["sex","high_blood_pressure","DEATH_EVENT"],
                  values="count",
                  title="Gender & High Blood Pressure Sunburst Chart ",
                  width=600,
                  height=600)

fig.show()

![newplot (6)](https://user-images.githubusercontent.com/82114061/118256655-5dccdd80-b4cb-11eb-9b61-c8fe585eec3d.png)
![newplot (7)](https://user-images.githubusercontent.com/82114061/118256664-60c7ce00-b4cb-11eb-97da-88cdf2639a5c.png)
![newplot (8)](https://user-images.githubusercontent.com/82114061/118256673-62919180-b4cb-11eb-9606-d083e814776e.png)
![newplot (9)](https://user-images.githubusercontent.com/82114061/118256686-64f3eb80-b4cb-11eb-983f-d44cc8ee59c6.png)


# Distribution of data to see dependency of variables 
sex_mortality = []
sex_mortality.append(len(df[(df['DEATH_EVENT']==1)&(df['sex']==1)]))
sex_mortality.append(len(df[(df['DEATH_EVENT']==0)&(df['sex']==1)]))
sex_mortality.append(len(df[(df['DEATH_EVENT']==1)&(df['sex']==0)]))
sex_mortality.append(len(df[(df['DEATH_EVENT']==0)&(df['sex']==0)]))
sex_labels = ['male_died','male_survived','female_died','female_survived']
plt.pie(x=sex_mortality,autopct='%.1f',labels=sex_labels);

smoking_died = len(df[(df['DEATH_EVENT']==1)&(df['smoking']==1)])
smoking_survived = len(df[(df['DEATH_EVENT']==0)&(df['smoking']==1)])
non_smoking_died = len(df[(df['DEATH_EVENT']==1)&(df['smoking']==0)])
non_smoking_survived = len(df[(df['DEATH_EVENT']==0)&(df['smoking']==0)])

smoking_mortality = []
smoking_mortality.append(len(df[(df['DEATH_EVENT']==1)&(df['smoking']==1)]))
smoking_mortality.append(len(df[(df['DEATH_EVENT']==0)&(df['smoking']==1)]))
smoking_mortality.append(len(df[(df['DEATH_EVENT']==1)&(df['smoking']==0)]))
smoking_mortality.append(len(df[(df['DEATH_EVENT']==0)&(df['smoking']==0)]))
smoking_labels = ['smoking_died','smoking_survived','non_smoking_died','non_smoking_survived']
plt.pie(x=smoking_mortality,autopct='%.1f',labels=smoking_labels);

diabetes_died = len(df[(df['DEATH_EVENT']==1)&(df['diabetes']==1)])
diabetes_survived = len(df[(df['DEATH_EVENT']==0)&(df['diabetes']==1)])
non_diabetes_died = len(df[(df['DEATH_EVENT']==1)&(df['diabetes']==0)])
non_diabetes_survived = len(df[(df['DEATH_EVENT']==0)&(df['diabetes']==0)])

diabetes_mortality = []
diabetes_mortality.append(len(df[(df['DEATH_EVENT']==1)&(df['diabetes']==1)]))
diabetes_mortality.append(len(df[(df['DEATH_EVENT']==0)&(df['diabetes']==1)]))
diabetes_mortality.append(len(df[(df['DEATH_EVENT']==1)&(df['diabetes']==0)]))
diabetes_mortality.append(len(df[(df['DEATH_EVENT']==0)&(df['diabetes']==0)]))
diabetes_labels = ['diabetes_died','diabetes_survived','non_diabetes_died','non_diabetes_survived']
plt.pie(x=diabetes_mortality,autopct='%.1f',labels=diabetes_labels);

anaemia_died = len(df[(df['DEATH_EVENT']==1)&(df['anaemia']==1)])
anaemia_survived = len(df[(df['DEATH_EVENT']==0)&(df['anaemia']==1)])
non_anaemia_died = len(df[(df['DEATH_EVENT']==1)&(df['anaemia']==0)])
non_anaemia_survived = len(df[(df['DEATH_EVENT']==0)&(df['anaemia']==0)])

anaemia_mortality = []
anaemia_mortality.append(len(df[(df['DEATH_EVENT']==1)&(df['anaemia']==1)]))
anaemia_mortality.append(len(df[(df['DEATH_EVENT']==0)&(df['anaemia']==1)]))
anaemia_mortality.append(len(df[(df['DEATH_EVENT']==1)&(df['anaemia']==0)]))
anaemia_mortality.append(len(df[(df['DEATH_EVENT']==0)&(df['anaemia']==0)]))
anaemia_labels = ['anaemia_died','anaemia_survived','non_anaemia_died','non_anaemia_survived']
plt.pie(x=anaemia_mortality,autopct='%.1f',labels=anaemia_labels);

high_blood_pressure_died = len(df[(df['DEATH_EVENT']==1)&(df['high_blood_pressure']==1)])
high_blood_pressure_survived = len(df[(df['DEATH_EVENT']==0)&(df['high_blood_pressure']==1)])
no_high_blood_pressure_died = len(df[(df['DEATH_EVENT']==1)&(df['high_blood_pressure']==0)])
no_high_blood_pressure_survived = len(df[(df['DEATH_EVENT']==0)&(df['high_blood_pressure']==0)])

high_blood_pressure_mortality = []
high_blood_pressure_mortality.append(len(df[(df['DEATH_EVENT']==1)&(df['high_blood_pressure']==1)]))
high_blood_pressure_mortality.append(len(df[(df['DEATH_EVENT']==0)&(df['high_blood_pressure']==1)]))
high_blood_pressure_mortality.append(len(df[(df['DEATH_EVENT']==1)&(df['high_blood_pressure']==0)]))
high_blood_pressure_mortality.append(len(df[(df['DEATH_EVENT']==0)&(df['high_blood_pressure']==0)]))
high_blood_pressure_labels = ['high_blood_pressure_died','high_blood_pressure_survived','no_high_blood_pressure_died','no_high_blood_pressure_survived']
plt.pie(x=high_blood_pressure_mortality,autopct='%.1f',labels=high_blood_pressure_labels);

![image](https://user-images.githubusercontent.com/82114061/118257360-2b6fb000-b4cc-11eb-8875-aa8ae84b3695.png)
![image](https://user-images.githubusercontent.com/82114061/118257377-30ccfa80-b4cc-11eb-9d09-c34428b0d040.png)
![image](https://user-images.githubusercontent.com/82114061/118257386-3591ae80-b4cc-11eb-9b90-95522e9bc3a6.png)
![image](https://user-images.githubusercontent.com/82114061/118257404-3a566280-b4cc-11eb-9acf-3ce67ab47a9d.png)
![image](https://user-images.githubusercontent.com/82114061/118257428-417d7080-b4cc-11eb-8096-9d5c2813bdd2.png)


# Correlation mapping 
df1=df.drop(["smoking"], axis = 1, inplace = True)
corr_matrix = df.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlaation btw features")
threshold = 0.2 
filtre = np.abs(corr_matrix["DEATH_EVENT"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(df[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation Between Features w Corr Theshold 0.75")
plt.show()

![image](https://user-images.githubusercontent.com/82114061/118257496-535f1380-b4cc-11eb-9872-1c8e25fbae20.png)
![image](https://user-images.githubusercontent.com/82114061/118257530-5c4fe500-b4cc-11eb-93af-6a46dd6d90af.png)


# Detecting Outliers in the data
plt.figure(figsize=(15, 12))

plt.subplot(2,3,1)
sns.boxplot(x='DEATH_EVENT', y='age', data=df)
plt.title('Distribution of Age')

plt.subplot(2,3,2)
sns.boxplot(x='DEATH_EVENT', y='creatinine_phosphokinase', data=df)
plt.title('Distribution of creatinine_phosphokinase')

plt.subplot(2,3,3)
sns.boxplot(x='DEATH_EVENT', y='ejection_fraction', data=df)
plt.title('Distribution of ejection_fraction')

plt.subplot(2,3,4)
sns.boxplot(x='DEATH_EVENT', y='platelets', data=df)
plt.title('Distribution of platelets')

plt.subplot(2,3,5)
sns.boxplot(x='DEATH_EVENT', y='serum_creatinine', data=df)
plt.title('Distribution of serum_creatinine')

plt.subplot(2,3,6)
sns.boxplot(x='DEATH_EVENT', y='serum_sodium', data=df)
plt.title('Distribution of serum_sodium');

![image](https://user-images.githubusercontent.com/82114061/118257595-7093e200-b4cc-11eb-889d-0d2dea9074f9.png)


# Removing Outliers
df2=df2[df['creatinine_phosphokinase']<1300]
df2=df2[df['ejection_fraction']<60]
df2=df2[(df['platelets']>100000) & (df['platelets']<420000)]
df2=df2[df['serum_creatinine']<1.5]
df2=df2[df['serum_sodium']>126]

# Checking the data after removing outliers 
def displot_numeric_features(feature):#code to visualize distribution, scatterplot and boxplot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=110)
    
    sns.distplot(df2[feature], ax=ax1)
    sns.scatterplot(df2[feature], df2["DEATH_EVENT"], ax=ax2)
    sns.boxplot(df2[feature],orient='h', ax=ax3, width=0.2)

    print(f"Skewness Coefficient of {feature} is {df2[feature].skew():.2f}")
    ax1.set_yticks([])
    
    return plt
    
displot_numeric_features("creatinine_phosphokinase").show()
displot_numeric_features("ejection_fraction").show()
displot_numeric_features("platelets").show()
displot_numeric_features("serum_creatinine").show()
displot_numeric_features("serum_sodium").show()

![image](https://user-images.githubusercontent.com/82114061/118257640-7c7fa400-b4cc-11eb-8943-fe548a4322d8.png)
![image](https://user-images.githubusercontent.com/82114061/118257664-83a6b200-b4cc-11eb-80a2-16a3972df551.png)
![image](https://user-images.githubusercontent.com/82114061/118257681-899c9300-b4cc-11eb-9565-474fa4287b7b.png)
![image](https://user-images.githubusercontent.com/82114061/118257700-8f927400-b4cc-11eb-8b8c-efc1d3ca3cc1.png)
![image](https://user-images.githubusercontent.com/82114061/118257719-94efbe80-b4cc-11eb-98dc-d3e58c4e141c.png)


# Separating features
x = df[[c for c in df.columns if c != 'DEATH_EVENT']] 
y = df['DEATH_EVENT']

# Balancing the data
from imblearn.over_sampling import SMOTE
from collections import Counter
smote = SMOTE()
x_smote, y_smote = smote.fit_resample(x, y)
print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_smote))

# Splitting data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size=0.4, random_state=0)
X_train.shape

# Scaling the data 
from sklearn.preprocessing import MinMaxScaler #scaling all the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model building 
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LogisticRegression 
logmodel = LogisticRegression()
logmodel.fit(X_train_scaled,y_train)
predictions = logmodel.predict(X_test_scaled)

# Printing classification report 
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

![Picture10](https://user-images.githubusercontent.com/82114061/118258210-3a0a9700-b4cd-11eb-9e8b-10800cf3e4a2.png)



# Plotting the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(y_test,predictions)
sns.heatmap(confusion_matrix, annot=True)

![image](https://user-images.githubusercontent.com/82114061/118257885-cbc5d480-b4cc-11eb-97e0-1319bd657364.png)


# Conclusive data visualisations 
fig = px.histogram(df, x="time", color="DEATH_EVENT", marginal="violin", hover_data=df.columns, 
                   title ="Distribution of TIME Vs DEATH_EVENT", 
                   labels={"time": "TIME"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                  )
fig.show()

fig = px.histogram(df, x="serum_creatinine", color="DEATH_EVENT", marginal="violin", hover_data=df.columns, 
                   title ="Distribution of SERUM CREATININE Vs DEATH_EVENT", 
                   labels={"serum_creatinine": "SERUM CREATININE"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                  )
fig.show()

fig = px.histogram(df, x="ejection_fraction", color="DEATH_EVENT", marginal="violin", hover_data=df.columns, 
                   title ="Distribution of EJECTION FRACTION Vs DEATH_EVENT", 
                   labels={"ejection_fraction": "EJECTION FRACTION"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                  )
fig.show()

fig = px.histogram(df, x="age", color="DEATH_EVENT", marginal="violin", hover_data=df.columns, 
                   title ="Distribution of AGE Vs DEATH_EVENT", 
                   labels={"age": "AGE"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                  )
fig.show()

fig = px.histogram(df, x="creatinine_phosphokinase", color="DEATH_EVENT", marginal="violin", hover_data=df.columns, 
                   title ="Distribution of CREATININE PHOSPHOKINASE Vs DEATH_EVENT", 
                   labels={"creatinine_phosphokinase": "CREATININE PHOSPHOKINASE"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                  )
fig.show()

![newplot (14)](https://user-images.githubusercontent.com/82114061/118258307-53134800-b4cd-11eb-9a51-544210c60284.png)
![newplot (15)](https://user-images.githubusercontent.com/82114061/118258529-98377a00-b4cd-11eb-9e7c-5c5241b9615f.png)
![newplot (16)](https://user-images.githubusercontent.com/82114061/118258540-9bcb0100-b4cd-11eb-9984-52a49d9c7d30.png)
![newplot (17)](https://user-images.githubusercontent.com/82114061/118258542-9ec5f180-b4cd-11eb-8d09-e8e3ee899eb3.png)
![newplot (18)](https://user-images.githubusercontent.com/82114061/118258549-a1284b80-b4cd-11eb-9524-7c16e3ccf246.png)

# Observations
DIABETES: There is no relation between Diabetes and death event. The mortality rate for both diabetic and non diabetic is observed to be approximately equal to each other. 


AGE: The patients have a higher chance of survival between the ages of 40 and 70. The survival rate is high for both male and female between 50 to 65. Patients having higher age are more prone to Cardiovascular Diseases
 

SMOKING: The patients who are non smoking have a higher chance of survival between the ages of 50 and 65. On the other hand, the patients who are smoking have change of survival between the ages 50 to 60. As per the data, the mortality rate is not remarkably dependent on the smoking or non smoking habits of males and females
 

SERUM CREATININE: When the level of serum creatinine in the blood is between 0 and 2 mg/dL, it is observed that the patient's survival rates is higher than the death rate. High value of serum creatine increase the probability to die. 
 
 
 BLOOD PRESSURE-It is observed that the difference of mortality rate for people who are having Blood Pressure and the mortality rate for people who aren’t are insignificant. Therefore it can be stated that there is no direct relation between the blood pressure levels and CVD deaths.


EJECTION FRACTION-It is found that lower the levels of ejection fraction below 50% are fatal and leads to higher death rate which reduces with gradual increase in the ejection fraction. Therefore it can be stated that there is a direct relation between the ejection fraction levels and CVD deaths.


CREATININE PHOSPHOKINASE- It is observed that a lower level of creatinine phosphokinase is directly related to a higher death rate which gradually reduces with an increase in the creatinine phosphokinase levels in the sample population. Therefore it can be stated that there is a direct relation between the creatinine phosphokinase levels and CVD deaths.


ANAEMIA: The mortality rate for people who are infected by Anaemia are 36% whereas mortality rate for people who are not infected by Anaemia is observed to be 30%. There is  no strong relation between the death event and Anaemia. Hence, Anaemia does not affect Cardiovascular arrest strongly.


PLATELETS: The count of platelets in the study depicts weak correlation with death event. death rate is not highly affected by the lower platelet count in the population


SERUM SODIUM: With increase in Serum Sodium level the mortality rate decreases, but there is weak influence of level of Serum Sodium on death in the population







