#!/usr/bin/env python
# coding: utf-8

# # HW1
# 
# ## Overview
# 
# Preparing the data, computing basic statistics and constructing simple models are essential steps for data science practice. In this homework, you will use clinical data as raw input to perform **Heart Failure Prediction**. For this homework, **Python** programming will be required. See the attached skeleton code as a start-point for the programming questions.
# 
# This homework assumes familiarity with Pandas. If you need a Pandas crash course, we recommend working through [100 Pandas Puzzles](https://github.com/ajcr/100-pandas-puzzles), the solutions are also available at that link. 

# In[1]:


import os
import sys

DATA_PATH = "../HW1-lib/data/"
TRAIN_DATA_PATH = DATA_PATH + "train/"
VAL_DATA_PATH = DATA_PATH + "val/"
    
sys.path.append("../HW1-lib")


# ## About Raw Data
# 
# For this homework, we will be using a clinical dataset synthesized from [MIMIC-III](https://www.nature.com/articles/sdata201635).
# 
# Navigate to `TRAIN_DATA_PATH`. There are three CSV files which will be the input data in this homework. 

# In[2]:


get_ipython().system('ls $TRAIN_DATA_PATH')


# **events.csv**
# 
# The data provided in *events.csv* are event sequences. Each line of this file consists of a tuple with the format *(pid, event_id, vid, value)*. 
# 
# For example, 
# 
# ```
# 33,DIAG_244,0,1
# 33,DIAG_414,0,1
# 33,DIAG_427,0,1
# 33,LAB_50971,0,1
# 33,LAB_50931,0,1
# 33,LAB_50812,1,1
# 33,DIAG_425,1,1
# 33,DIAG_427,1,1
# 33,DRUG_0,1,1
# 33,DRUG_3,1,1
# ```
# 
# - **pid**: De-identified patient identier. For example, the patient in the example above has pid 33. 
# - **event_id**: Clinical event identifier. For example, DIAG_244 means the patient was diagnosed of disease with ICD9 code [244](http://www.icd9data.com/2013/Volume1/240-279/240-246/244/244.htm); LAB_50971 means that the laboratory test with code 50971 was conducted on the patient; and DRUG_0 means that a drug with code 0 was prescribed to the patient. Corresponding lab (drug) names can be found in `{DATA_PATH}/lab_list.txt` (`{DATA_PATH}/drug_list.txt`).
# - **vid**: Visit identifier. For example, the patient has two visits in total. Note that vid is ordinal. That is, visits with bigger vid occour after that with smaller vid.
# - **value**: Contains the value associated to an event (always 1 in the synthesized dataset).

# **hf_events.csv**
# 
# The data provided in *hf_events.csv* contains pid of patients who have been diagnosed with heart failure (i.e., DIAG_398, DIAG_402, DIAG_404, DIAG_428) in at least one visit. They are in the form of a tuple with the format *(pid, vid, label)*. For example,
# 
# ```
# 156,0,1
# 181,1,1
# ```
# 
# The vid indicates the index of the first visit with heart failure of that patient and a label of 1 indicates the presence of heart failure. **Note that only patients with heart failure are included in this file. Patients who are not mentioned in this file have never been diagnosed with heart failure.**

# **event_feature_map.csv**
# 
# The *event_feature_map.csv* is a map from an event_id to an integer index. This file contains *(idx, event_id)* pairs for all event ids.

# ## 1 Descriptive Statistics [20 points]
# 
# Before starting analytic modeling, it is a good practice to get descriptive statistics of the input raw data. In this question, you need to write code that computes various metrics on the data described previously. A skeleton code is provided to you as a starting point.
# 
# The definition of terms used in the result table are described below:
# 
# - **Event count**: Number of events recorded for a given patient.
# - **Encounter count**: Number of visits recorded for a given patient.
# 
# Note that every line in the input file is an event, while each visit consists of multiple events.

# **Complete the following code cell to implement the required statistics.**
# 
# Please be aware that **you are NOT allowed to change the filename and any existing function declarations.** Only `numpy`, `scipy`, `scikit-learn`, `pandas` and other built-in modules of python will be available for you to use. The use of `pandas` library is suggested. 

# In[3]:


import time
import pandas as pd
import numpy as np
import datetime

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT.

def read_csv(filepath=TRAIN_DATA_PATH):

    '''
    Read the events.csv and hf_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    
    NOTE: remember to use `filepath` whose default value is `TRAIN_DATA_PATH`.
    '''
    
    events = pd.read_csv(filepath + 'events.csv')
    hf = pd.read_csv(filepath + 'hf_events.csv')

    return events, hf

def event_count_metrics(events, hf):

    '''
    TODO : Implement this function to return the event count metrics.
    
    Event count is defined as the number of events recorded for a given patient.
    '''
    
    avg_hf_event_count = None
    max_hf_event_count = None
    min_hf_event_count = None
    avg_norm_event_count = None
    max_norm_event_count = None
    min_norm_event_count = None
    
    # your code here
    df_events = events[['pid','vid']]
    event_count_groups_pid = events.groupby('pid').size()
    uniq_pids = set(hf['pid']) # unique pids
    hf_counts = event_count_groups_pid[event_count_groups_pid.index.isin(uniq_pids)]
    normal_counts = event_count_groups_pid[~event_count_groups_pid.index.isin(uniq_pids)]
    
    avg_hf_event_count = hf_counts.mean()
    max_hf_event_count = hf_counts.max()
    min_hf_event_count = hf_counts.min()
    avg_norm_event_count = normal_counts.mean()
    max_norm_event_count = normal_counts.max()
    min_norm_event_count = normal_counts.min()

    return avg_hf_event_count, max_hf_event_count, min_hf_event_count,            avg_norm_event_count, max_norm_event_count, min_norm_event_count

def encounter_count_metrics(events, hf):

    '''
    TODO : Implement this function to return the encounter count metrics.
    
    Encounter count is defined as the number of visits recorded for a given patient. 
    '''
    
    avg_hf_encounter_count = None
    max_hf_encounter_count = None
    min_hf_encounter_count = None
    avg_norm_encounter_count = None
    max_norm_encounter_count = None
    min_norm_encounter_count = None
    
    # your code here
    group_encounter_counts = events.groupby('pid')['vid'].nunique()
    uniq_pids = set(hf['pid'])
    hf_counts = group_encounter_counts[group_encounter_counts.index.isin(uniq_pids)]
    norm_counts = group_encounter_counts[~group_encounter_counts.index.isin(uniq_pids)]

    avg_hf_encounter_count = hf_counts.mean()
    max_hf_encounter_count = hf_counts.max()
    min_hf_encounter_count = hf_counts.min()
    avg_norm_encounter_count = norm_counts.mean()
    max_norm_encounter_count = norm_counts.max()
    min_norm_encounter_count = norm_counts.min()
    return avg_hf_encounter_count, max_hf_encounter_count, min_hf_encounter_count,            avg_norm_encounter_count, max_norm_encounter_count, min_norm_encounter_count


# In[4]:


'''
DO NOT MODIFY THIS.
'''

events, hf = read_csv(TRAIN_DATA_PATH)

#Compute the event count metrics
start_time = time.time()
event_count = event_count_metrics(events, hf)
end_time = time.time()
print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
print(event_count)

#Compute the encounter count metrics
start_time = time.time()
encounter_count = encounter_count_metrics(events, hf)
end_time = time.time()
print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
print(encounter_count)


# In[5]:


'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''

events, hf = read_csv(TRAIN_DATA_PATH)
event_count = event_count_metrics(events, hf)
assert event_count == (188.9375, 2046, 28, 118.64423076923077, 1014, 6), "event_count failed!"


# In[6]:


'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''

events, hf = read_csv(TRAIN_DATA_PATH)
encounter_count = encounter_count_metrics(events, hf)
assert encounter_count == (2.8060810810810812, 34, 2, 2.189423076923077, 11, 1), "encounter_count failed!"


# ## 2 Feature construction [40 points] 
# 
# It is a common practice to convert raw data into a standard data format before running real machine learning models. In this question, you will implement the necessary python functions in this script. You will work with *events.csv*, *hf_events.csv* and *event_feature_map.csv* files provided in **TRAIN_DATA_PATH** folder. The use of `pandas` library in this question is recommended. 
# 
# Listed below are a few concepts you need to know before beginning feature construction (for details please refer to lectures). 
# 
# <img src="img/window.jpg" width="600"/>
# 
# - **Index vid**: Index vid is evaluated as follows:
#   - For heart failure patients: Index vid is the vid of the first visit with heart failure for that patient (i.e., vid field in *hf_events.csv*). 
#   - For normal patients: Index vid is the vid of the last visit for that patient (i.e., vid field in *events.csv*). 
# - **Observation Window**: The time interval you will use to identify relevant events. Only events present in this window should be included while constructing feature vectors.
# - **Prediction Window**: A fixed time interval that is to be used to make the prediction.
# 
# In the example above, the index vid is 3. Visits with vid 0, 1, 2 are within the observation window. The prediction window is between visit 2 and 3.

# ### 2.1 Compute the index vid [10 points]
# 
# Use the definition provided above to compute the index vid for all patients. Complete the method `read_csv` and `calculate_index_vid` provided in the following code cell. 

# In[7]:


import pandas as pd
import datetime


def read_csv(filepath=TRAIN_DATA_PATH):
    
    '''
    Read the events.csv, hf_events.csv and event_feature_map.csv files.
    
    NOTE: remember to use `filepath` whose default value is `TRAIN_DATA_PATH`.
    '''

    events = pd.read_csv(filepath + 'events.csv')
    hf = pd.read_csv(filepath + 'hf_events.csv')
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, hf, feature_map


def calculate_index_vid(events, hf):
    
    '''
    TODO: This function needs to be completed.

    Suggested steps:
        1. Create list of normal patients (hf_events.csv only contains information about heart failure patients).
        2. Split events into two groups based on whether the patient has heart failure or not.
        3. Calculate index vid for each patient.
    
    IMPORTANT:
        `indx_vid` should be a pd dataframe with header `['pid', 'indx_vid']`.
    '''

     # your code here
    indx_vid = ''
    indx_vid = events[['pid', 'vid']].drop_duplicates(subset=['pid'], keep = 'last', ignore_index=True)
    hf_pid_dict = hf.set_index('pid')['vid'].to_dict()
    indx_vid['vid'] = indx_vid['pid'].map(hf_pid_dict).fillna(indx_vid['vid'])
    indx_vid.rename(columns={"vid": "indx_vid"}, inplace=True)
    return indx_vid


# In[8]:


'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''

events, hf, feature_map = read_csv(TRAIN_DATA_PATH)
indx_vid_df = calculate_index_vid(events, hf)
assert indx_vid_df.shape == (4000, 2), "calculate_index_vid failed!"

indx_vid = dict(list(zip(indx_vid_df.pid, indx_vid_df.indx_vid)))
assert indx_vid[78] == 1, "calculate_index_vid failed!"
assert indx_vid[1230] == 5, "calculate_index_vid failed!"



# ### 2.2 Filter events [10 points]
# 
# Remove the events that occur outside the observation window. That is, only keep events in visits before index vid. Complete the method *filter_events* provided in the following code cell.

# In[9]:


def filter_events(events, indx_vid):
    
    '''
    TODO: This function needs to be completed.

    Suggested steps:
        1. Join indx_vid with events on pid.
        2. Filter events occuring in the observation window [:, index vid) (Note that the right side is OPEN).
    
    
    IMPORTANT:
        `filtered_events` should be a pd dataframe withe header  `['pid', 'event_id', 'value']`.
    '''

    filtered_events = None

    events_indx_vid = events.merge(indx_vid, on='pid', how='left')
    filtered_events = events_indx_vid[events_indx_vid.vid < events_indx_vid.indx_vid][['pid', 'event_id', 'value']].reset_index(drop=True)
    #print(filtered_events[filtered_events.pid == 78])
    return filtered_events


# In[10]:


'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''

events, hf, feature_map = read_csv(TRAIN_DATA_PATH)
indx_vid = calculate_index_vid(events, hf)
filtered_events = filter_events(events, indx_vid)
assert filtered_events[filtered_events.pid == 78].shape == (128, 3), "filter_events failed!"



# ### 2.3 Aggregate events [10 points]
# 
# To create features suitable for machine learning, we will need to aggregate the events for each patient as follows:
# 
# - **count** occurences for each event.
# 
# Each event type will become a feature and we will directly use event_id as feature name. For example, given below raw event sequence for a patient,
# 
# ```
# 33,DIAG_244,0,1
# 33,LAB_50971,0,1
# 33,LAB_50931,0,1
# 33,LAB_50931,0,1
# 33,DIAG_244,1,1
# 33,DIAG_427,1,1
# 33,DRUG_0,1,1
# 33,DRUG_3,1,1
# 33,DRUG_3,1,1
# ```
# 
# We can get feature value pairs *(event_id, value)* for this patient with ID *33* as
# ```
# (DIAG_244, 2.0)
# (LAB_50971, 1.0)
# (LAB_50931, 2.0)
# (DIAG_427, 1.0)
# (DRUG_0, 1.0)
# (DRUG_3, 2.0)
# ```
# 
# Next, replace each *event_id* with the *feature_id* provided in *event_feature_map.csv*.
# 
# ```
# (146, 2.0)
# (1434, 1.0)
# (1429, 2.0)
# (304, 1.0)
# (898, 1.0)
# (1119, 2.0)
# ```
# 
# Lastly, in machine learning algorithm like logistic regression, it is important to normalize different features into the same scale. We will use the [min-max normalization](http://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range) approach. (Note: we define $min(x)$ is always 0, i.e. the scale equation become $x$/$max(x)$).
# 
# Complete the method *aggregate_events* provided in the following code cell.

# In[11]:


def aggregate_events(filtered_events_df, hf_df, feature_map_df):
    
    '''
    TODO: This function needs to be completed.

    Suggested steps:
        1. Replace event_id's with index available in event_feature_map.csv.
        2. Aggregate events using count to calculate feature value.
        3. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios).
    
    
    IMPORTANT:
        `aggregated_events` should be a pd dataframe with header `['pid', 'feature_id', 'feature_value']`.
    '''
    
    aggregated_events = None
    # your code here
    events_feature_df = filtered_events_df.merge(feature_map, how="left", on=["event_id"])
    events_feature_df = events_feature_df.rename(columns={'idx':'feature_id'})
    aggregated_events = events_feature_df.groupby(['pid', 'feature_id']).size().reset_index().rename(columns={0:'feature_value'})
    feature_max = aggregated_events.groupby(['feature_id'])['feature_value'].max().reset_index().rename(columns={'feature_value':'max_value'})
    aggregated_events = aggregated_events.merge(feature_max, how = 'left', on = 'feature_id')
    aggregated_events.feature_value = aggregated_events.feature_value/aggregated_events.max_value
    aggregated_events.drop(columns = ['max_value'], inplace = True)
    return aggregated_events


# In[12]:


'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''

events, hf, feature_map = read_csv(TRAIN_DATA_PATH)
index_vid = calculate_index_vid(events, hf)
filtered_events = filter_events(events, index_vid)
aggregated_events = aggregate_events(filtered_events, hf, feature_map)
assert aggregated_events[aggregated_events.pid == 88037].shape == (29, 3), "aggregate_events failed!"


# ### 2.4 Save in  SVMLight format [10 points]
# 
# If the dimensionality of a feature vector is large but the feature vector is sparse (i.e. it has only a few nonzero elements), sparse representation should be employed. In this problem you will use the provided data for each patient to construct a feature vector and represent the feature vector in SVMLight format.
# 
# ```
# <line> .=. <target> <feature>:<value> <feature>:<value>
# <target> .=. 1 | 0
# <feature> .=. <integer>
# <value> .=. <float>
# ```
# 
# The target value and each of the feature/value pairs are separated by a space character. Feature/value pairs MUST be ordered by increasing feature number. **(Please do this in `save_svmlight()`.)** Features with value zero can be skipped. For example, the feature vector in SVMLight format will look like: 
# 
# ```
# 1 2:0.5 3:0.12 10:0.9 2000:0.3
# 0 4:1.0 78:0.6 1009:0.2
# 1 33:0.1 34:0.98 1000:0.8 3300:0.2
# 1 34:0.1 389:0.32
# ```
# 
# where, 1 or 0 will indicate whether the patient has heart failure or not (i.e. the label) and it will be followed by a series of feature-value pairs **sorted** by the feature index (idx) value.
# 
# You may find *utils.py* useful. You can review the code by running `%load utils.py`.

# In[13]:


# %load   ../HW1-lib/utils.py


# In[14]:


import utils
import collections

def create_features(events_in, hf_in, feature_map_in):

    indx_vid = calculate_index_vid(events_in, hf_in)

    #Filter events in the observation window
    filtered_events = filter_events(events_in, indx_vid)

    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, hf_in, feature_map_in)

    '''
    TODO: Complete the code below by creating two dictionaries.
        1. patient_features : Key is pid and value is array of tuples(feature_id, feature_value). 
                              Note that pid should be integer.
        2. hf : Key is pid and value is heart failure label. 
                Only include patients with heart failure in this dictionary.
    '''
    
    patient_features = None
    hf = None

    patient_features = {}
    for pid, group in aggregated_events.groupby('pid'):
        patient_features[pid] = group[['feature_id', 'feature_value']].to_records(index=False).tolist()

    hf_pids = hf_in['pid'].tolist() 
    hf = {pid: 1 for pid in hf_pids}
    return patient_features, hf
   
def save_svmlight(patient_features, hf, op_file):
    
    '''
    TODO: This function needs to be completed.

    Create op_file: - which saves the features in svmlight format. (See instructions in section 2.4 for detailed explanatiom)
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    To save the files, you could write:
        deliverable.write(bytes(f"{label} {feature_value}\n", 'utf-8'))
    '''
    

    #print(patient_features)
    #patient_features = sorted(patient_features, key=lambda x: (x['pid'], x['feature_id']))
    deliverable = open(op_file, 'wb')
    with open(op_file, 'wb') as deliverable:
        for pid in sorted(patient_features.keys()):  # Sort patients in ascending order
            label = hf.get(pid, 0)
            features = sorted(patient_features[pid], key=lambda x: x[0])  # Sort features by feature_id
            feature_str = ''  
            feature_str = utils.bag_to_svmlight(features)
            # Convert features into SVMLight format
            line = f"{label} {feature_str}\n"
            deliverable.write(bytes(line, 'utf-8'))


# In[15]:


'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''

events_in, hf_in, feature_map_in = read_csv(TRAIN_DATA_PATH)
events_in = events_in.loc[:1000]
hf_in = hf_in.loc[:100]
patient_features, hf = create_features(events_in, hf_in, feature_map_in)
assert 78 in patient_features, "create_features is missing patients"
assert len(patient_features[78]) == 127, "create_features is wrong"
assert patient_features[78][:5] == [(20, 1.0), (164, 1.0), (175, 1.0), (182, 1.0), (190, 1.0)], "create_features is wrong"
assert len(hf) == 101, "create_features is wrong"



# The whole pipeline:

# In[16]:


def main():
    events_in, hf_in, feature_map_in = read_csv(TRAIN_DATA_PATH)
    patient_features, hf = create_features(events_in, hf_in, feature_map_in)
    save_svmlight(patient_features, hf, 'features_svmlight.train')
    
    events_in, hf_in, feature_map_in = read_csv(VAL_DATA_PATH)
    patient_features, hf = create_features(events_in, hf_in, feature_map_in)
    save_svmlight(patient_features, hf, 'features_svmlight.val')
    
main()


# ## 3 Predictive Modeling [40 points]
# 
# Make sure you have finished section 2 before you start to work on this question because some of the files generated in section 2 (*features_svmlight.train*) will be used in this question.

# ### 3.1 Model Creation [20 points]
# 
# In the previous question, you constructed feature vectors for patients to be used as training data in various predictive models (classifiers). Now you will use this training data (*features_svmlight.train*) in 3 predictive models. 

# **Step - a. Implement Logistic Regression, SVM and Decision Tree. Skeleton code is provided in the following code cell.**

# In[17]:


import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

import utils


# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT.
# USE THIS RANDOM STATE FOR ALL OF THE PREDICTIVE MODELS.
# OR THE TESTS WILL NEVER PASS.
RANDOM_STATE = 545510477


#input: X_train, Y_train
#output: Y_pred
def logistic_regression_pred(X_train, Y_train):
    
    """
    TODO: Train a logistic regression classifier using X_train and Y_train.
    Use this to predict labels of X_train. Use default params for the classifier.
    """
    
    # your code here
    model = LogisticRegression(random_state=RANDOM_STATE) 
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_train)
    return Y_pred

    
#input: X_train, Y_train
#output: Y_pred
def svm_pred(X_train, Y_train):
    
    """
    TODO: Train a SVM classifier using X_train and Y_train.
    Use this to predict labels of X_train. Use default params for the classifier
    """
    
    # your code here
    svm = LinearSVC(random_state=RANDOM_STATE)
    svm.fit(X_train, Y_train)
    Y_pred = svm.predict(X_train)
    
    return Y_pred

    
#input: X_train, Y_train
#output: Y_pred
def decisionTree_pred(X_train, Y_train):

    """
    TODO: Train a logistic regression classifier using X_train and Y_train.
    Use this to predict labels of X_train. Use max_depth as 5.
    """
    
    # your code here
    dt_classifier = DecisionTreeClassifier(max_depth=5,random_state=RANDOM_STATE)

    # 2. Train the classifier
    dt_classifier.fit(X_train, Y_train)

    # 3. Predict labels for the test set
    Y_pred = dt_classifier.predict(X_train)

    return Y_pred

    
#input: Y_pred,Y_true
#output: accuracy, precision, recall, f1-score
def classification_metrics(Y_pred, Y_true):
    
    """
    TODO: Calculate the above mentioned metrics.
    NOTE: It is important to provide the output in the same order.
    """
    
    # your code here
    acc, precision, recall, f1score = accuracy_score(Y_true,Y_pred),                                    precision_score(Y_true, Y_pred),                                    recall_score(Y_true, Y_pred),                                    f1_score(Y_true, Y_pred)
    return acc, precision, recall, f1score


    
#input: Name of classifier, predicted labels, actual labels
def display_metrics(classifierName, Y_pred, Y_true):
    print("______________________________________________")
    print(("Classifier: "+classifierName))
    acc, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
    print(("Accuracy: "+str(acc)))
    print(("Precision: "+str(precision)))
    print(("Recall: "+str(recall)))
    print(("F1-score: "+str(f1score)))
    print("______________________________________________")
    print("")

    
def main():
    X_train, Y_train = utils.get_data_from_svmlight("features_svmlight.train")
    
    display_metrics("Logistic Regression", logistic_regression_pred(X_train, Y_train), Y_train)
    display_metrics("SVM",svm_pred(X_train, Y_train),Y_train)
    display_metrics("Decision Tree", decisionTree_pred(X_train, Y_train), Y_train)

    
main()



# In[18]:


'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''

from utils import get_data_from_svmlight
from numpy.testing import assert_almost_equal

### 3.1a Training Accuracy [3 points]
X_train, Y_train = get_data_from_svmlight("features_svmlight.train")

# test_accuracy_lr
expected = 0.856338028169014
Y_pred = logistic_regression_pred(X_train, Y_train)
actual = classification_metrics(Y_pred, Y_train)[0]
assert_almost_equal(actual, expected, decimal=2, verbose=False, err_msg="test_accuracy_lr failed!")


# **Step - b. Evaluate your predictive models on a separate test dataset in *features_svmlight.val* (binary labels are provided in that svmlight file as the first field). Skeleton code is provided in the following code cell.**

# In[19]:


import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

import utils


# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT.
# USE THIS RANDOM STATE FOR ALL OF THE PREDICTIVE MODELS.
# OR THE TESTS WILL NEVER PASS.
RANDOM_STATE = 545510477


#input: X_train, Y_train and X_test
#output: Y_pred
def logistic_regression_pred(X_train, Y_train, X_test):

    """
    TODO: train a logistic regression classifier using X_train and Y_train. 
    Use this to predict labels of X_test use default params for the classifier.
    """
    
    # your code here
    model = LogisticRegression(random_state=RANDOM_STATE) 
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    return Y_pred
    

#input: X_train, Y_train and X_test
#output: Y_pred
def svm_pred(X_train, Y_train, X_test):
    
    """
    TODO: Train a SVM classifier using X_train and Y_train.
    Use this to predict labels of X_test use default params for the classifier.
    """
    
    # your code here
    svm = LinearSVC(random_state=RANDOM_STATE)
    svm.fit(X_train, Y_train)
    Y_pred = svm.predict(X_test)
    
    return Y_pred
    
#input: X_train, Y_train and X_test
#output: Y_pred
def decisionTree_pred(X_train, Y_train, X_test):
    
    """
    TODO: Train a logistic regression classifier using X_train and Y_train.
    Use this to predict labels of X_test.
    IMPORTANT: use max_depth as 5. Else your test cases might fail.
    """
    
    # your code here
    dt_classifier = DecisionTreeClassifier(max_depth=5,random_state=RANDOM_STATE)

    # 2. Train the classifier
    dt_classifier.fit(X_train, Y_train)

    # 3. Predict labels for the test set
    Y_pred = dt_classifier.predict(X_test)

    return Y_pred


#input: Y_pred,Y_true
#output: accuracy, precision, recall, f1-score
def classification_metrics(Y_pred, Y_true):
    
    """
    TODO: Calculate the above mentioned metrics.
    NOTE: It is important to provide the output in the same order.
    """
    
    # your code here
    accuracy = accuracy_score(Y_true, Y_pred)
    precision = precision_score(Y_true, Y_pred)
    recall = recall_score(Y_true, Y_pred)
    f1 = f1_score(Y_true, Y_pred)

    return accuracy, precision, recall, f1
    
#input: Name of classifier, predicted labels, actual labels
def display_metrics(classifierName, Y_pred, Y_true):
    print("______________________________________________")
    print(("Classifier: "+classifierName))
    acc, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
    print(("Accuracy: "+str(acc)))
    print(("Precision: "+str(precision)))
    print(("Recall: "+str(recall)))
    print(("F1-score: "+str(f1score)))
    print("______________________________________________")
    print("")

    
def main():
    X_train, Y_train = utils.get_data_from_svmlight("features_svmlight.train")
    X_test, Y_test = utils.get_data_from_svmlight(os.path.join("features_svmlight.val"))

    display_metrics("Logistic Regression", logistic_regression_pred(X_train, Y_train, X_test), Y_test)
    display_metrics("SVM", svm_pred(X_train, Y_train, X_test), Y_test)
    display_metrics("Decision Tree", decisionTree_pred(X_train, Y_train, X_test), Y_test)


    
from utils import get_data_from_svmlight
from numpy.testing import assert_almost_equal

### 3.1b Prediction Accuracy [3 points]
X_train, Y_train = get_data_from_svmlight("features_svmlight.train")
X_test, Y_test = get_data_from_svmlight("features_svmlight.val")

# test_accuracy_lr
expected = 0.6937086092715232
Y_pred = logistic_regression_pred(X_train, Y_train, X_test)
actual = classification_metrics(Y_pred, Y_test)[0]
print(actual, expected)

main()


# In[20]:


'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''

from utils import get_data_from_svmlight
from numpy.testing import assert_almost_equal

### 3.1b Prediction Accuracy [3 points]
X_train, Y_train = get_data_from_svmlight("features_svmlight.train")
X_test, Y_test = get_data_from_svmlight("features_svmlight.val")

# test_accuracy_lr
expected = 0.6937086092715232
Y_pred = logistic_regression_pred(X_train, Y_train, X_test)
actual = classification_metrics(Y_pred, Y_test)[0]
assert_almost_equal(actual, expected, decimal=2, verbose=False, err_msg="test_accuracy_lr failed!")


# ### 3.2 Model Validation [20 points]
# 
# In order to fully utilize the available data and obtain more reliable results, machine learning practitioners use cross-validation to evaluate and improve their predictive models. You will demonstrate using two cross-validation strategies against SVM. 
# 
# - K-fold: Divide all the data into $k$ groups of samples. Each time $\frac{1}{k}$ samples will be used as test data and the remaining samples as training data.
# - Randomized K-fold: Iteratively random shuffle the whole dataset and use top specific percentage of data as training and the rest as test. 

# **Implement the two cross-validation strategies.**
# - **K-fold:** Use the number of iterations k=5; 
# - **Randomized K-fold**: Use a test data percentage of 20\% and k=5 for the number of iterations for Randomized

# In[21]:


from sklearn.model_selection import KFold, ShuffleSplit
from numpy import mean

import utils


# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT.
# USE THIS RANDOM STATE FOR ALL OF THE PREDICTIVE MODELS.
# OR THE TESTS WILL NEVER PASS.
RANDOM_STATE = 545510477


#input: training data and corresponding labels
#output: f1
def get_f1_kfold(X, Y, k=5):
    
    """
    TODO: First get the train indices and test indices for each iteration.
    Then train the classifier accordingly.
    Report the mean f1 score of all the folds.
    
    Note that you do not need to set random_state for KFold, as it has no effect since shuffle is False by default.
    """
    
    # your code here
    f1_scores = []
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_pred = svm_pred(X_train, Y_train, X_test)
        acc, precision, recall, f1score = classification_metrics(Y_pred, Y_test)
        f1_scores.append(f1score)
            
    return np.mean(f1_scores)
    

    
#input: training data and corresponding labels
#output: f1
def get_f1_randomisedCV(X, Y, iterNo=5, test_percent=0.20):

    """
    TODO: First get the train indices and test indices for each iteration.
    Then train the classifier accordingly.
    Report the mean f1 score of all the iterations.
    
    Note that you need to set random_state for ShuffleSplit
    """

    # your code here
    shufflesplit = ShuffleSplit(n_splits=iterNo, test_size=test_percent, random_state=RANDOM_STATE)
    f1_scores = []  
    for train_index, test_index in shufflesplit.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_pred = svm_pred(X_train, Y_train, X_test)
        acc, precision, recall, f1score = classification_metrics(Y_pred, Y_test)
        f1_scores.append(f1score)


    return np.mean(f1_scores)


    
def main():
    X,Y = utils.get_data_from_svmlight("features_svmlight.train")
    print("Classifier: SVM")
    f1_k = get_f1_kfold(X,Y)
    print(("Average F1 Score in KFold CV: "+str(f1_k)))
    f1_r = get_f1_randomisedCV(X,Y)
    print(("Average F1 Score in Randomised CV: "+str(f1_r)))


main()


# In[22]:


'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''

from numpy.testing import assert_almost_equal

### 3.2 Cross Validation F1 [10 points]
# test_f1_cv_kfold
expected = 0.7258461959533061
X, Y = get_data_from_svmlight("features_svmlight.train")
actual = get_f1_kfold(X, Y)
assert_almost_equal(actual, expected, decimal=2, verbose=False, err_msg="test_f1_cv_kfold failed!")


# In[ ]:





# In[ ]:




