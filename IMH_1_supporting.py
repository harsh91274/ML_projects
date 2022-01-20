#supporting functions for IMH case study by Harsh Biren Vora
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


from sklearn.model_selection import train_test_split; 
from sklearn.linear_model import LogisticRegression;
from sklearn.preprocessing import MinMaxScaler, StandardScaler;
from sklearn.metrics import classification_report, roc_curve, roc_auc_score;
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

rcParams['figure.figsize'] = 10,10;

#%% UNIQUE VALUE FUNCTION
def unique_vals(df):
    
    df_cols=pd.DataFrame(df.dtypes, index=df.columns);
    df_cols['unique_entries']=df.nunique().values;
    df_cols['unique_fraction']=(df_cols['unique_entries']/df.shape[0]).round(2);

    df_cols['null_entries']=df.isna().sum();
    df_cols['null_fraction']=(df_cols['null_entries']/df.shape[0]).round(2);
    
    for c in list(df.columns):
        # get a list of unique values
        n = df[c].unique()
        
        if len(n)<30:
            print(c)
            print(n)
        else:
            print(c + ': ' +str(len(n)) + ' unique values');
    
    return df_cols;

#%% ORDINAL TO CIRCULAR FEATURE FUNCTION

def circular_transform(wd, week_col, month_col):
    wd['week_sin'] = np.sin(wd[week_col]*(2.*np.pi/max(wd[week_col])))
    wd['week_cos'] = np.cos(wd[week_col]*(2.*np.pi/max(wd[week_col])))
    wd['month_sin'] = np.sin((wd[month_col]-1)*(2.*np.pi/12))
    wd['month_cos'] = np.cos((wd[month_col]-1)*(2.*np.pi/12))

    return wd

#%%
def prevalence(y_actual):
    # this function calculates the prevalence of the positive class (label = 1)
    return (sum(y_actual)/len(y_actual));

#%%
def get_fi(gbm, X_orig, fname):
    feature_importances = pd.DataFrame(gbm.feature_importances_, index = X_orig.columns,columns=['importance']).sort_values('importance', ascending=False);
    #feature_importances.head()
    
    #f6=plt.figure()
    #plt.hist(feature_importances);
    feature_importances.plot(kind='barh');
    plt.xlabel('Feature');  
    #plt.xticks(feature_importances['index'])
    #plt.xticks([col for col in X_orig.columns])
    plt.ylabel('Feature Importance');
    plt.savefig(fname+'.jpeg');
    
    return feature_importances;

#%%
def predict_p(best_model, X_train, X_test):
    y_train_preds = best_model.predict_proba(X_train)[:,1];
    y_test_preds = best_model.predict_proba(X_test)[:,1];

    return y_train_preds, y_test_preds;
#%%
def auc_roc(y_train, y_train_preds, y_test, y_test_preds, model='GBM'):
    
    fpr_train, tpr_train, thresholds_train=roc_curve(y_train, y_train_preds);
    fpr_test, tpr_test, thresholds_test=roc_curve(y_test, y_test_preds);
    
    auc_train=roc_auc_score(y_train, y_train_preds);
    auc_test=roc_auc_score(y_test, y_test_preds);
    
    print('TRAINING ' + model +' ROC score: ' + str(roc_auc_score(y_train, y_train_preds)));
    print('TESTING '+ model + ' ROC score: '+str(roc_auc_score(y_test, y_test_preds)));
    
    f9=plt.figure()
    plt.plot(fpr_train, tpr_train, 'r-',label ='Train AUC:%.3f'%auc_train)
    plt.plot(fpr_test, tpr_test, 'g-',label ='Test AUC:%.3f'%auc_test)
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(fontsize=14)
    plt.show();
    f9.savefig('ROC_AUC_'+model+'.jpeg')
    
    return auc_train, auc_test;


