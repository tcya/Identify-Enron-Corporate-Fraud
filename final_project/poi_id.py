#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np
import matplotlib.pyplot as plt

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

def Draw(data, features, feature_i, feature_j, remove_TOTAL = True, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than 4 clusters
    colors = ["b", "r", "k", "m", "g"]
    if remove_TOTAL:
        if data.has_key('TOTAL'):
            del data['TOTAL']
    f1_name = features[feature_i]
    f2_name = features[feature_j]
    poi = [data[i]['poi'] if (data[i]['poi'] != 'NaN' and data[i][f1_name] != 'NaN' and data[i][f2_name] != 'NaN') else 0 for i in data.keys()]
    feature_1 = [data[i][features[feature_i]] if (data[i]['poi'] != 'NaN' and data[i][f1_name] != 'NaN' and data[i][f2_name] != 'NaN') else 0 for i in data.keys()]
    feature_2 = [data[i][features[feature_j]] if (data[i]['poi'] != 'NaN' and data[i][f1_name] != 'NaN' and data[i][f2_name] != 'NaN') else 0 for i in data.keys()]

    plt.scatter(feature_1, feature_2, color = [colors[1] if i else colors[0] for i in poi])
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    # plt.savefig(name)
    plt.show()

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
for person in data_dict.keys():
    data_dict[person]['fraction_from_poi'] = float(data_dict[person]['from_poi_to_this_person'])/float(data_dict[person]['to_messages'])
    data_dict[person]['fraction_to_poi'] = float(data_dict[person]['from_this_person_to_poi'])/float(data_dict[person]['from_messages'])
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

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)


