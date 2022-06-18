'''
StateFarm Classification
Homework: GLM vs Non-GLM
========================
written by:
Bryan Caraway
========================
Code Repostory:
https://github.com/Bryancaraway/Statefarm_hw
'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, metrics
from tensorflow.keras.losses import BinaryFocalCrossentropy as bfc

tf.random.set_seed(0) # reproducibility 

class StateFarm_Classification:

    def __init__(self, test_fname, train_fname, explore_data=False, save_pred=False):
        # open up files
        self.trainXY = pd.read_csv(train_fname)
        self.testX  = pd.read_csv(test_fname)
        self.explore_data = explore_data
        self.save_pred = save_pred

    def do_statefarm_hw(self):
        # pipeline start
        self.step_1() # clean and prepare data
        self.step_2() # initialize classification models: log_reg,
        self.step_3() # apply models to test dataset
        self.step_4() # compare model performance

    def step_1(self): # clean and prepare data
        '''
        For data cleaning:
            Handle: strings, NaNs, catagorical features, yes/no
            Then: seperate X and Y, standardize features
        '''
        if self.explore_data:
            self.do_data_exploration()
            
        # drop bad feature x39, only one value
        self.trainXY.drop(columns=['x39'], inplace=True)
        self.testX.drop(columns=['x39'],  inplace=True)
        
        # clean up day of the week
        day_conv = {'Monday':'Mon','Tuesday':'Tue','Wednesday':'Wed','Thursday':'Thur','Friday':'Fri','Saturday':'Sat','Sunday':'Sun',}
        self.trainXY['x3'] = self.trainXY['x3'].apply(lambda d_ : d_ if d_ not in day_conv else day_conv[d_]).str.lower()
        self.testX['x3'] = self.testX['x3'].apply(lambda d_ : d_ if d_ not in day_conv else day_conv[d_]).str.lower()
        
        # convert % and $ to floats
        self.trainXY['x7'] = self.trainXY['x7'].str.replace('%','').astype(float)
        self.testX['x7'] = self.testX['x7'].str.replace('%','').astype(float)
        self.trainXY['x19'] = self.trainXY['x19'].apply(lambda s_ : s_[:s_.find('.')+3].replace('$','') ).astype(float)
        self.testX['x19'] = self.testX['x19'].apply(lambda s_ : s_[:s_.find('.')+3].replace('$','') ).astype(float)
        
        # convert binary to 0/1
        bin_conv = {'female':0 ,'male':1, 'no':0, 'yes':1 }
        bin_cols = ['x24','x31','x93','x99'] 
        for bin_col in bin_cols:
            self.trainXY[bin_col] = self.trainXY[bin_col].apply(lambda b_ : b_ if b_ not in bin_conv else bin_conv[b_])
            self.testX[bin_col] = self.testX[bin_col].apply(lambda b_ : b_ if b_ not in bin_conv else bin_conv[b_])
            
        # nan to num treatment for cat, special nan treatment
        self.trainXY.fillna(value={'x24':.5, 'x33': 'None', 'x77': 'None', 'x99':0,}, inplace=True) 
        self.testX.fillna(value={ 'x24':.5, 'x33': 'None', 'x77': 'None', 'x99':0,}, inplace=True)
        
        # encode catagorical data : x3 (7), x33 (51+1, has nans), x60 (12), x65 (5), x77 (7+1, has nans) 
        ohe = OneHotEncoder(sparse=False, categories='auto', drop='first')
        cat_cols = ['x3','x33','x60','x65','x77']
        train_cat_ohe = ohe.fit_transform(self.trainXY[cat_cols])
        test_cat_ohe =  ohe.transform(self.testX[cat_cols])
        train_cats = pd.DataFrame(train_cat_ohe, columns=ohe.get_feature_names_out(cat_cols))
        test_cats = pd.DataFrame(test_cat_ohe, columns=ohe.get_feature_names_out(cat_cols))
        self.trainXY = pd.concat([self.trainXY,train_cats], axis=1)
        self.testX = pd.concat([self.testX,test_cats], axis=1)
        self.trainXY.drop(columns=cat_cols, inplace=True)
        self.testX.drop(columns=cat_cols, inplace=True)
        
        # clean up numerical data
        # for "almost exclusively one value" continuous variables, set non mode to 0 and mode to 1
        excl_cols = ['x58','x67','x71','x84']
        for excl_col in excl_cols:
            train_mode = self.trainXY[excl_col].mode()[0]
            self.trainXY[excl_col] = self.trainXY[excl_col].apply(lambda v_ : 1. if v_ == train_mode else (0 if ~np.isnan(v_) else v_))
            self.testX[excl_col] = self.testX[excl_col].apply(lambda v_ : 1. if v_ == train_mode else (0 if ~np.isnan(v_) else v_))
            
        # first add extra nan-indicator variables for variables with nans
        nan_cols = self.trainXY.columns[self.trainXY.isnull().any()]
        for nan_col in nan_cols:
            self.trainXY[nan_col+'_nan'] = self.trainXY[nan_col].isnull().astype(int)
            self.testX[nan_col+'_nan'] = self.testX[nan_col].isnull().astype(int)
            
        # impute nan with training median (numerical) or mode (binary), and add new nan indicator column
        for bin_col in excl_cols+bin_cols:
            train_mode = self.trainXY[bin_col].mode()[0]
            self.trainXY[bin_col].fillna(train_mode, inplace=True)
            self.testX[bin_col].fillna(train_mode, inplace=True)
        for num_nan_col in self.trainXY.columns[self.trainXY.isnull().any()]:
            train_med = self.trainXY[num_nan_col].median()
            self.trainXY[num_nan_col].fillna(train_med, inplace=True)
            self.testX[num_nan_col].fillna(train_med, inplace=True)
            
        # split labels from training variables
        self.trainY, self.trainX = self.trainXY['y'].values, self.trainXY.drop(columns=['y'])
        self.trainX, self.testX = self.trainX.align(self.testX, join='left', axis=1) # mach sure columns are aligned
        
        # stanardize the data
        stand = StandardScaler()
        self.trainX = stand.fit_transform(self.trainX.values)
        self.testX = stand.transform(self.testX.values)
        
    def step_2(self): # initialize the log_reg and nn models
        kf = KFold(n_splits=5)
        self.fold_splits = list(kf.split(self.trainX, self.trainY))
        # use cross validation, training 5 seperate models of each mlax
        self.glm = []
        self.non_glm = []
        for train_i, val_i in self.fold_splits:
            glm     = LogisticRegression(solver='lbfgs', C=0.1, max_iter=500)
            glm.fit(self.trainX[train_i], self.trainY[train_i])
            self.glm.append(glm)
            
            non_glm = self.build_nn_model() 
            non_glm.fit(self.trainX[train_i], self.trainY[train_i], epochs=100, batch_size=1024,
                        callbacks=[callbacks.EarlyStopping(monitor='val_loss',patience=2,restore_best_weights=True)],
                        validation_data = (self.trainX[val_i], self.trainY[val_i]),
                        shuffle=True, verbose=0)
            self.non_glm.append(non_glm)
                
    def step_3(self): # get predicitons of the test dataset
        # need to output 2 csv files : 'glmresults.csv', 'nonglmresults.csv' : no header label or index column
        self.glm_pred     = np.mean([glm.predict_proba(self.testX)[:,1] for glm in self.glm], axis=0)
        self.non_glm_pred = np.mean([non_glm.predict(self.testX, verbose=0) for non_glm in self.non_glm], axis=0)
        if self.save_pred:
            pd.DataFrame(self.glm_pred).to_csv('glmresults.csv',        header=None, index=None)
            pd.DataFrame(self.non_glm_pred).to_csv('nonglmresults.csv', header=None, index=None)
        
    def step_4(self): # get AUC (4 decimal places) for each model
        # estimate the AUC for test using Kfold cross validation
        glm_auc_scores = []
        non_glm_auc_scores = []

        for i, (train_i, val_i) in enumerate(self.fold_splits):
            glm_pred     = self.glm[i].predict_proba(self.trainX[val_i])[:,1]
            glm_auc_scores.append(round(roc_auc_score(self.trainY[val_i], glm_pred),4))
            
            non_glm_pred     = self.non_glm[i].predict(self.trainX[val_i], verbose=0)
            non_glm_auc_scores.append(round(roc_auc_score(self.trainY[val_i], non_glm_pred),4))
            
        glm_auc     = round(np.mean(glm_auc_scores),4)
        non_glm_auc = round(np.mean(non_glm_auc_scores),4)
        print(f'GLM Test AUC     : {glm_auc:.4f}')
        print(f'Non GLM Test AUC : {non_glm_auc:.4f}')

    def do_data_exploration(self):
        '''
        Notes on data exploration:
        1) Several non-float/int data: 3, 7, 19, 24, 31, 33, 39, 60, 65, 77, 93, 99
        x3 (day of the week), x7 (%), x19 ($), x24 (fem/male), x31 (y/n), x33 (state), x39 (miles), x60 (month), x65 (ins), x77 (car), x93 (y/n), x99 (y/n)
        2) x39 (miles) all have the same values (SAFELY DROP)
        3) x99 is all yes, will change nan to 0
        4) Several features with > 30% NAN entries: 30, 44, 49, 52, 54, 55, 57, 74, 95, 99
        x30 (80%), x44 (86%), x49 (32%), x52 (40%), x54 (32%), x55 (44%), x57 (81%), x74 (33%), x95 (32%), x99 (32%)
        '''
        df = self.trainXY
        print(df.loc[:,df.columns[df.dtypes == 'O'].values]) # which variables are not numbers
        print(df.isnull().sum()[df.isnull().sum()>0][(df.isnull().sum()[df.isnull().sum()>0] / len(df) > 0.30)] / len(df)) # percent is null if over 30%
        #
        import matplotlib.pyplot as plt # v3.4.2
        corr = df.corr() 
        print(abs(corr['y']).sort_values(ascending=False).head(10)) # no feature is strongly correlated, max(|corr|) = .12
        for i in range(0, len(df.columns), 16): # only 20 plots at a time
            df.iloc[:,i:i+16].hist(bins=50, figsize=(12,8))
            plt.show()
            plt.close()
        sus_num_features = ['x38', 'x58', 'x59', 'x67', 'x71', 'x75', 'x79', 'x84', 'x98']
        print(df.loc[:,sus_num_features].describe())
        ax = df.loc[df['y']==1,sus_num_features].hist(bins=50, figsize=(12,8), histtype='step', density=True, label='1') # these look suspicious
        df.loc[df['y']==0,sus_num_features].hist(     bins=50, figsize=(12,8), histtype='step', xdensity=True, label='0', ax=ax) # these look suspicious
        '''
        Notes on suspicious features:
        x38 : min capped near 0 (standardize should fix)
        x58 : almost exclusively set to ~300
        x59 : 1/0 , may be fine
        x67 : almost exclusively set to ~14
        x71 : almost exclusively set to ~0
        x75 : very high tail (standardize should fix)
        x79 : 1/0, may be fine
        x84 : almost exclusively set to ~2
        x98 : 1/0, may be fine
        '''
        plt.show()
        plt.close()


    def build_nn_model(self):
        # using deep learning (2 hideen layers), feed-forward NN binary classifier
        # binary focal loss, to better handle label imabalance
        main_input = layers.Input(shape=[self.trainX.shape[1]], name='input')
        layer  = layers.BatchNormalization()(main_input) # standard practice, regularization
        layer    = layers.Dense(256,  activation='relu')(layer)
        layer    = layers.Dense(64,  activation='relu')(layer)
        layer     = layers.Dropout(.5)(layer) # regularization
        output = layers.Dense(1, activation='sigmoid', name='output')(layer) # binary
        model      = models.Model(inputs=main_input, outputs=output, name='model')
        optimizer  = optimizers.Adam(learning_rate=0.001) # hard-coded learning rate 
        model.compile(loss=bfc(gamma=.20),optimizer=optimizer,metrics=['accuracy', metrics.AUC()])
        return model

    

if __name__=='__main__':
    statefarm_config = {
        'test_fname'  : 'exercise_40_test.csv',
        'train_fname' : 'exercise_40_train.csv',
        'explore_data': False,
        'save_pred'   : True, 
    }
    sf_hw = StateFarm_Classification(**statefarm_config)    
    sf_hw.do_statefarm_hw()
