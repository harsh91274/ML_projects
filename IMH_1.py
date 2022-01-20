#Intermountain Healthcare Case Study by Harsh Biren Vora
#The objective is to develop a model to predict patient no-shows given patient history

#%% Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from sklearn.model_selection import train_test_split; 
from sklearn.linear_model import LogisticRegression;
from sklearn.preprocessing import MinMaxScaler, StandardScaler;
from sklearn.metrics import classification_report, roc_curve, roc_auc_score;
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#import functions
from IMH_1_supporting import predict_p, unique_vals, circular_transform, prevalence, get_fi, auc_roc

# figure size in inches
rcParams['figure.figsize'] = 10,10;

#%% READ IN DATA
#read data dictionary
data_dict=pd.read_csv('Data_ref.csv')

#read data
data_orig=pd.read_csv('Medical_No_Shows.csv')

p_var=int(input('Remake Plots (1=YES)? '));
h_var=int(input('Try Hyperparameter Tuning (1=YES)? '));
s_var=int(input('Try Oversampling (1=YES)? '));
#%% PRELIMINARY DATA FEASIBILITY

print(data_orig.shape);
#Data has approx 110k records

#check formats of the columns
data_formats=data_orig.dtypes

#check null values in columns 
data_nulls=data_orig.isna().sum(axis=0);
# NO nulls in dataset is strange (or the data is unusually clean) - need to check if nulls are replaced with a value

#check unique values
data_unique=unique_vals(data_orig)

#Column AppointmentID is primary key - Number of unique values = Number of rows
#Columns No-show and Gender need label encoding
#Column locationID is assumes ordinality in its current form - needs processing 
#Column Disability has four unique values - needs one-hot encoding
#Columns ScheduledDay and AppointmentDay need to converted to datetime format for feature engineering
#%% DATASET CLEANING 
data=data_orig.copy();
#list of columns to drop for machine learning post data-cleanup
drop_cols=['AppointmentID'];
#list of columns for encoding
ohe_cols=['Disability'];
#-----------------------------------------------------------------------------
#Modify column gender column to binary 
data['gender_mod']=np.where(data['Gender']=='F', 0, 1);
drop_cols.append('Gender');

#-----------------------------------------------------------------------------
#convert columns to datetime
data['scheduled_day_mod']=pd.to_datetime(data['ScheduledDay']);
drop_cols.append('ScheduledDay');

#-----------------------------------------------------------------------------
data['appointment_day_mod']=pd.to_datetime(data['AppointmentDay']);
drop_cols.append('AppointmentDay');

#drop the old columns in favor of the newly created columns
#data.drop(columns=drop_cols, inplace=True);

#%% STATISTICAL ANALYSIS OF THE DATASET
#basic statistics of the data
data_describe=data.describe();
#-----------------------------------------------------------------------------
#age has a minimum value of -1, which is incorrect
#check to see the impact of this incorrect date
age_check=data[data['Age']<0];

#only one record with (age=-1) so we can drop
data=data[~data['Age']<0];

#%% FEATURE ENGINEERING
# New Feature to calculate time between appointment and scheduling date
data['days_to_appt']=(data['appointment_day_mod']-data['scheduled_day_mod']).dt.days;

f1=plt.figure(figsize=(12,8))
plt.hist(data['days_to_appt']);
f1.savefig('schedule_to_appt_days_before.jpeg');
#-----------------------------------------------------------------------------
#check for negative values of the calculated column
date_check=data[data['days_to_appt']<0]
#~38k rows with negative values - thats too many to drop
#Assuming this reflects same-day or on-the-spot appointments with doctors, will change values to zero
data.loc[data['days_to_appt']<0, 'days_to_appt']=0;
#-----------------------------------------------------------------------------
f2=plt.figure(figsize=(12,8))
plt.hist(data['days_to_appt']);
f2.savefig('schedule_to_appt_days_after.jpeg');

#add columns to drop
#drop_cols.append('scheduled_day_mod');

#%% New features to calculate month number, week number and day of the week for appointment
data['appointment_month']=data['appointment_day_mod'].dt.month;
data['appointment_week']=data['appointment_day_mod'].dt.weekofyear;
data['appointment_day']=data['appointment_day_mod'].dt.dayofweek;

#add appointment_day column to encoding list
ohe_cols.append('appointment_day');
#-----------------------------------------------------------------------------
#Modify No Show target column to binary
data['no_show_output']=np.where(data['No-show']=='No', 0, 1);
drop_cols.append('No-show');

#%% CREATE FEATURE TO COUNT PREVIOUSLY MISSED appointments by patient
data=data.sort_values(by=['appointment_day_mod', 'scheduled_day_mod'], ascending=True).reset_index(drop=True);
data['n_prev_appt_missed']=data.groupby('PatientID')['no_show_output'].apply(lambda x: x.cumsum()-1)
data.loc[data['n_prev_appt_missed']<0,'n_prev_appt_missed']=0;

#-----------------------------------------------------------------------------
#add modified columns to drop list
drop_cols.append('appointment_day_mod');
drop_cols.append('scheduled_day_mod');
drop_cols.append('PatientID');
#drop_cols.append('AppointmentDay');

#%% CREATE TOTAL CONDITIONS FEATURE
data['total_conditions']=data['Hypertension']+data['Diabetes']+data['Alcoholism']+data['Disability'].astype(bool).map({True:1, False:0})

#%% CHANGE TO CIRCULAR FEATURES IF THE DATA EXISTS ACROSS MULTIPLE YEARS?
#change month and week number from ordinal features to circular features
#data=circular_transform(data, 'appointment_week', 'appointment_month');
#drop_cols.append('appointment_month');
#drop_cols.append('appointment_week');

#%%List of columns for encoding
#ohe_cols=['Disability'], 'appointment_day'];
data_ohe=pd.get_dummies(data[ohe_cols].astype(str),drop_first = False);
drop_cols.append('Disability')
drop_cols.append('appointment_day');

#Add LocationID to drop columns since it assumes ordinality and cannot encode 81 unique values meaningfully
drop_cols.append('LocationID');

#%% VISUALIZATIONS

if p_var==1:
    data['LocationID'].value_counts().plot(kind="bar",figsize=(20,5));
    plt.xlabel('LocationID', fontsize=12); plt.ylabel('Frequency', fontsize=12);
    plt.savefig('LocationID_distribution.jpeg');
    plt.close();
    
    data['appointment_day'].value_counts().plot(kind="bar",figsize=(20,5));
    plt.xlabel('Appointment Day (0 is Monday)', fontsize=12); plt.ylabel('Frequency', fontsize=12);
    plt.savefig('appointmentday_distribution.jpeg');
    plt.close();
    
    data['days_to_appt'].value_counts().plot(kind="bar",figsize=(20,5));
    plt.xlabel('Days to Appointment', fontsize=12); plt.ylabel('Frequency', fontsize=12);
    plt.savefig('waittoappt_distribution.jpeg');
    plt.close();
    
    plt.figure(figsize=(16,4))
    plt.xticks(rotation=90, fontsize=12)
    ax = sns.countplot(x=data['Age'])
    ax.set_title("No of appointments by age");
    plt.savefig('age_distribution.jpeg');
    plt.close()
    
    #%%#PLOT CLASS DISTRIBUTION OF NUMERICAL VARIABLES
    
    plt.figure()
    data['days_to_appt'][data['No-show']=="No"].hist(alpha=0.8, bins=20);
    data['days_to_appt'][data['No-show']=="Yes"].hist(alpha=0.8, bins=20);
    plt.xlabel('Days to Appointment', fontsize=12); plt.ylabel('Frequency', fontsize=12);
    plt.legend()
    plt.savefig('waittoappt_class_distribution.jpeg');
    plt.close();
    
    plt.figure()
    data['Age'][data['No-show']=="No"].hist(alpha=0.8, bins=20);
    data['Age'][data['No-show']=="Yes"].hist(alpha=0.8, bins=20);
    plt.xlabel('Age', fontsize=12); plt.ylabel('Frequency', fontsize=12);
    plt.legend()
    plt.savefig('age_class_distribution.jpeg');
    plt.close();
    
    plt.figure()
    data['n_prev_appt_missed'][data['No-show']=="No"].hist(alpha=0.8, bins=20);
    data['n_prev_appt_missed'][data['No-show']=="Yes"].hist(alpha=0.8, bins=20);
    plt.xlabel('Previous Appointments Missed', fontsize=12); plt.ylabel('Frequency', fontsize=12);
    plt.legend()
    plt.savefig('prevapptsmissed_class_distribution.jpeg');
    plt.close();
    
    #%% PLOT CATEGORICAL VARIABLES CLASS DISTRIBUTION
    categorical_vars = ['appointment_day', 'Hypertension', 'Diabetes', 'Alcoholism', 'Disability', 'SMS_received', 'MedicaidIND']
    
    fig = plt.figure(figsize=(16, 11))
    for i, var in enumerate(categorical_vars):
        ax = fig.add_subplot(3, 3, i+1)
        data.groupby([var, 'No-show'])[var].count().unstack('No-show').plot(ax=ax, kind='bar', stacked=True)
    
    plt.savefig('categorical_class_dist.jpeg');
    plt.close()
    
    #%% pairplot
    pp_cols=['Gender', 'Age', 'MedicaidIND', 'Hypertension', 'Diabetes', 'Alcoholism', 'Disability', 'SMS_received', 'days_to_appt', 'appointment_month', 'appointment_week', 'appointment_day', 'n_prev_appt_missed', 'total_conditions', 'No-show'];
    plt.figure(figsize=(20,20));
    sns.color_palette("vlag", as_cmap=True)
    sns.pairplot(data[pp_cols], hue='No-show', kind='hist');
    plt.savefig('pairplot_all.jpeg');
    plt.close()

#%% DROP COLUMNS PRIOR TO MACHINE LEARNING
#add encoded columns
data_ml=pd.concat([data, data_ohe], axis=1);
#-----------------------------------------------------------------------------
#drop columns 
data_ml=data_ml.drop(columns=drop_cols);
#-----------------------------------------------------------------------------
# CHECK THE MACHINE LEARNING DATASET
data_ml_describe=unique_vals(data_ml);


#%% ANALYZE CLASS SEPARATION OF TARGET VARIABLE
#plot shows weak linear correlation to final output column, maybe due to class imalance
#check for class distribution of target variable

ml_output=data_ml['no_show_output'].value_counts();
#0: showed for appointment
#1: did not show for appointment
n_noshow=data[data_ml['no_show_output']==1].shape[0];
n_show=data[data_ml['no_show_output']==0].shape[0];

print('No Show Percentage: ' + str(((n_noshow*100)/(n_show+n_noshow))))
#approximately 20% no shows - target variable is imbalanced

#%% FIND CORRELATION 
#create correlation matrix to analyze relationship between features and with output variable
corr = data_ml.corr()
# plot the heatmap
f4=plt.figure()
sns.heatmap(corr, cmap='vlag', xticklabels=corr.columns, yticklabels=corr.columns)
f4.savefig('ml_input_correlation.jpeg')

#%% SPLIT DATASET FOR MACHINE LEARNING

#define input dataframe
X_orig=data_ml.drop(columns=['no_show_output']);
#define target dataframe
y=data_ml['no_show_output'];

#SCALE THE DATASET FOR regression
scaler=MinMaxScaler();
X=scaler.fit_transform(X_orig);
#SPLIT dataset stratified by classes in output variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

print('Minority class testing set prevalence(n = %d):%.3f'%(len(y_test),prevalence(y_test.values)))
print('Minority class training set prevalence(n = %d):%.3f'%(len(y_train),prevalence(y_train.values)))

#%% LOGISTIC REGRESSION
lr = LogisticRegression(solver='newton-cg')
lr.fit(X_train, y_train)
print(lr.score(X_train,y_train))
y_pred_lr = lr.predict(X_test)
lr_report = classification_report(y_test, y_pred_lr)
print(f"Logistic Regression Classification Report : \n{lr_report}")

#%% RANDOM FOREST
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
rf_report = classification_report(y_test, y_pred_rf)

print(f"Random Forest Classification Report : \n{rf_report}")

#%% GRADIENT BOOSTING CLASSIFIER
gbm = GradientBoostingClassifier()
gbm.fit(X_train, y_train)

y_pred_gbm = gbm.predict(X_test)
gbm_report = classification_report(y_test, y_pred_gbm)

print(f"Gradient Boosting Classification Report : \n{gbm_report}")
#Gradient Boosting model is best model based on F-1 score
best_model=gbm;
#%% PREDICT PROBABILITIES
y_train_preds, y_test_preds=predict_p(best_model, X_train, X_test);

#%% GET AUC, ROC scores and curve
auc_train, auc_test=auc_roc(y_train, y_train_preds, y_test, y_test_preds, 'GBM');

#%%-FEATURE IMPORTANCES FOR GRADIENT BOOSTIN
#----------------------------------------------------------------------------
feature_importances_gbm=get_fi(gbm, X_orig, 'gbm_feature_importances');

#%% HYPERPARAMETER SEARCH
#SEARCH hyperparameters

if h_var==1:
    from sklearn.model_selection import GridSearchCV
    #p_test = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750]}
    
    p_test = {'learning_rate':[0.01, 0.05, 0.1, 0.5, 1],  
                  'min_samples_split':[2,5,10,20], 
                  'max_depth':[2,3,5,10]}
    
    tuning = GridSearchCV(estimator =GradientBoostingClassifier(), param_grid = p_test, scoring='f1',n_jobs=-1, cv=5)
    tuning.fit(X_train,y_train)
    tuning.best_params_ 
    tuning.best_score_
    
    #get predictions1
    gbm_opt = GradientBoostingClassifier(**tuning.best_params_)
    gbm_opt.fit(X_train, y_train)
    #y_pred_gbm = gbm.predict(X_test)
    y_pred_gbm_opt = gbm_opt.predict(X_test)
    gbm_opt_report = classification_report(y_test, y_pred_gbm_opt)
    
    print(f"Gradient Boosting Classification Hyperameter Optimized Report : \n{gbm_opt_report}")
    
    #PREDICT PROBABILITIES
    y_train_opt_preds, y_test_opt_preds=predict_p(gbm_opt, X_train, X_test);
    
    # GET AUC, ROC scores and curve
    auc_opt_train, auc_opt_test=auc_roc(y_train, y_train_opt_preds, y_test, y_test_opt_preds, 'GBM_OPT');
    
    #FEATURE IMPORTANCES FOR GRADIENT BOOSTING
    #----------------------------------------------------------------------------
    feature_importances_gbm_opt=get_fi(gbm_opt, X_orig, 'gbm_OPT_feature_importances');

#%% OVERSAMPLING
if s_var==1:
    from imblearn.over_sampling import SMOTENC, RandomOverSampler 
    
    print('OVERSAMPLING')
    ros = RandomOverSampler(random_state=42);
    Xs, ys = ros.fit_resample(X, y);
    print('OVERSAMPLING COMPLETE')
    #%%
    #SPLIT dataset stratified by classes in output variable
    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.3, random_state=0, stratify=ys)
    
    print('Minority class testing set prevalence(n = %d):%.3f'%(len(y_test),prevalence(y_test.values)))
    print('Minority class training set prevalence(n = %d):%.3f'%(len(y_train),prevalence(y_train.values)))

    gbm = GradientBoostingClassifier()
    gbm.fit(X_train, y_train)
    
    y_pred_gbm = gbm.predict(X_test)
    gbm_report = classification_report(y_test, y_pred_gbm)
    
    print(f"SMOTE Gradient Boosting Classification Report : \n{gbm_report}")
    #Gradient Boosting model is best model based on F-1 score
    best_model=gbm;
    # SMOTE PREDICT PROBABILITIES
    y_train_preds, y_test_preds=predict_p(best_model, X_train, X_test);
    
    #SMOTE GET AUC, ROC scores and curve
    auc_train, auc_test=auc_roc(y_train, y_train_preds, y_test, y_test_preds, 'GBM');
    
    #SMOTE FEATURE IMPORTANCES FOR GRADIENT BOOSTIN
    #----------------------------------------------------------------------------
    feature_importances_gbm=get_fi(gbm, X_orig, 'SMOTE_gbm_feature_importances');
