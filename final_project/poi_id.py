#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np
import matplotlib.pyplot as plt

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_financial = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
features_email = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi','shared_receipt_with_poi',
'fraction_from_poi', 'fraction_from_poi','email_address']
# features_list = ['poi','salary'] # You will need to use more features
features_list = ['poi'] + features_financial + features_email[:-1]
features_list = ['poi', 'bonus', 'total_stock_value', 'exercised_stock_options']
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
for person in data_dict.keys():
    data_dict[person]['fraction_from_poi'] = 0. if data_dict[person]['to_messages'] == 'NaN' else \
    float(data_dict[person]['from_poi_to_this_person'])/float(data_dict[person]['to_messages'])
    data_dict[person]['fraction_to_poi'] = 0. if data_dict[person]['from_messages'] == 'NaN' else \
    float(data_dict[person]['from_this_person_to_poi'])/float(data_dict[person]['from_messages'])

### Task 2: Remove outliers
del data_dict['TOTAL']

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
# print my_dataset['CAUSEY RICHARD A']
# Draw(my_dataset, features_list, 1, 1)
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=3)
##Finding the best features giving highest f1 score
# for ii in range(1,len(features_list)):
#     selection = SelectKBest(k=ii)
#     features_new = selection.fit_transform(features, labels)
#     features_list_new = ['poi'] + [features_list[1:][i] for i in np.where(selection.get_support()==True)[0]]
#     print features_list_new
#     test_classifier(clf, my_dataset, features_list_new)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# for ii in range(2,8):
#     print 'min_samples_split: ', ii
#     clf = tree.DecisionTreeClassifier(min_samples_split=ii)
#     test_classifier(clf, my_dataset, features_list)

test_classifier(clf, my_dataset, features_list)
### Dump your classifier, dataset, and features_list so
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)


