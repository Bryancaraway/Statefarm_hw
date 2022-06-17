'''
StateFarm Classification
Homework: GLM vs Non-GLM
========================
written by:
Bryan Caraway
'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras import backend as K

class StateFarm_Classification:

    def __init__(self, test_fname, train_fname, explore_data=False, save_models=False, save_pred=False):
        # open up files
        self.trainXY = pd.read_csv(train_fname)
        self.testXY  = pd.read_csv(test_fname)
        self.explore_data = explore_data
        self.save_models = save_models
        self.save_pred = save_pred

    def do_statefarm_hw(self):
        # pipeline start
        step_1() # clean and prepare data
        step_2() # initialize classification models: log_reg,
        step_3() # apply models to test dataset
        step_4() # compare model performance
        pass

    def step_1(self): # clean and prepare data
        '''
        For data cleaning:
            Handle: NaNs, catagorical features, yes/no, outliers
            Then: seperate X and Y, standardize features, PCA (dimensionality reduction)
        '''
        if self.explore_data:
            self.do_data_exploration()
        # nan to num treatment
        
        # transform "object" type data
        
        # split labels from training variables
        self.trainX, self.trainY = self.trainXY.filter(like='x').values, self.trainXY.filter(like='y').values
        self.testX, self.testY = self.testXY.filter(like='x').values, self.testXY.filter(like='y').values
        
        
        pass
    def step_2(self): # initialize the log_reg and [] modles
        self.glm     = LogisticRegression()
        self.non_glm = self.build_nn_model()
        # train here
        self.glm.fit(self.trainX, self.trainY)
        self.non_glm.fit(self.trainX, self.trainY, epochs=100, batch_seize=2056,
                         callbacks=[callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)],
                         shuffle=True, verbose=0)
        # add save model stuff?
        
    def step_3(self): # get predicitons of the test dataset
        # need to output 2 csv files : 'glmresults.csv', 'nonglmresults.csv' : no header label or index column
        self.glm_pred     = self.glm.predict(self.testX)
        self.non_glm_pred = self.non_glm.predict(self.testX)
        if self.save_pred:
            pd.DataFrame(self.glm_pred).to_csv('glmresults.csv',        header=None, index=None)
            pd.DataFrame(self.non_glm_pred).to_csv('nonglmresults.csv', header=None, index=None)
        
    def step_4(self): # get AUC (4 decimal places) for each model
        '''
        Please write an executive summary that includes a comparison of the two modeling approaches, with emphasis on  relative strengths and weaknesses of the algorithms. 
        To receive maximum points on the executive summary, at least one strength and one weakness for each algorithm should be described.

        Additionally, your executive summary should include which algorithm you think will perform better on the test set, and your support for that decision. 
        Based on your model development process, include estimates for the test AUCs for each model. The estimates should be in a table and rounded to four decimal places. 
        Finally, describe how you would demonstrate to a business partner that one model is better than the other without using a scoring metric.
        '''
        glm_auc     = round(roc_auc_score(self.testY, self.glm_pred),4)
        non_glm_auc = round(roc_auc_score(self.testY, self.non_glm_pred),4)
        print('GLM Test AUC     : {self.glm_acu:.4f}')
        print('Non GLM Test AUC : {self.non_glm_acu:.4f}')

    def do_data_exploration(self):
        '''
        Notes on data exploration:
        1) Several non-float/int data: 3, 7, 19, 24, 31, 33, 39, 60, 77, 93, 99
        x3 (day), x7 (%), x19 ($), x24 (fem/male), x31 (y/n), x33 (state), x39 (miles), x60 (month), x77 (car), x93 (y/n), x99 (y/n)
        2) x39 (miles) all have the same values
        3) Several features with > 30% NAN entries: 30, 44, 49, 52, 54, 55, 57, 74, 95, 99
        x30 (80%), x44 (86%), x49 (32%), x52 (40%), x54 (32%), x55 (44%), x57 (81%), x74 (33%), x95 (32%), x99 (32%)
        '''
        df = self.trainXY
        print(df.loc[:,df.columns[df.dtypes == 'O'].values]) # which variables are not numbers
        print(df.isnull().sum()[df.isnull().sum()>0][(df.isnull().sum()[df.isnull().sum()>0] / len(df) > 0.30)] / len(df)) # percent is null if over 30%
        
        pass

    def build_nn_model(self):
        # using deep learning (2 hideen layers), feed-forward NN binary classifier
        main_input = layers.Input(shape=[len(self.trainX.columns)], name='input')
        bn_layer  = layers.BatchNormalization()(main_input) # standard practice, regularization
        hl1    = layers.Dense(128, activatoin='relu')(layer)
        hl2    = layers.Dense(64,  activatoin='relu')(hl1)
        #do     = layers.Dropout()(hl2) # regularization
        output = layers.Dense(1, activation='sigmoid', name='output')(hl2)
        #
        # add load model?
        model      = models.Model(inputs=main_input, outputs=output, name='model')
        optimizer  = optimizers.Adam(learning_rate=0.001) # hard-coded learning rate
        model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy',tf.keras.metrics.AUC()])
        return model

    

def __name__=='__main__':
    statefarm_config = {
        'test_fname'  : 'exercise_40_test.csv',
        'train_fname' : 'exercise_40_train.csv',
        'explore_data': True,
        'save_models' : False,
        'save_pred'   : False, 
    }
    sf_hw = StateFarm_Classification(**statefarm_config)    
    sf_hw.do_statefarm_hw()
