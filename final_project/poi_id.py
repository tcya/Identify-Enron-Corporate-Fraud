#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
sys.path.append("../k_means/")
import numpy as np
import matplotlib.pyplot as plt

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn import preprocessing

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than 4 clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_financial = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
features_email = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi','shared_receipt_with_poi',
'fraction_from_poi', 'fraction_to_poi','email_address']
# features_list = ['poi','salary'] # You will need to use more features
features_list = ['poi'] + features_financial + features_email[:-1]
features_list = ['poi', 'salary', 'bonus', 'deferred_income']
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
for key in my_dataset.keys():
    for feat in my_dataset[key].keys():
        if my_dataset[key][feat] == 'NaN':
            my_dataset[key][feat] = 0
# print my_dataset['CAUSEY RICHARD A']
# Draw(my_dataset, features_list, 1, 1)
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features = preprocessing.scale(features)
keys = sorted(my_dataset.keys())
idx = 0
for key in keys:
    if [my_dataset[key][i] for i in features_list[1:]] != [0]*(len(features_list)-1):
        for ii, feat in enumerate(features_list[1:]):
            my_dataset[key][feat] = features[idx][ii]
        idx += 1

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.cluster import KMeans
clf = KMeans(n_clusters=5)
pred = clf.fit_predict(features)
Draw(pred, features, labels, mark_poi=True, name="clusters_after_scaling.pdf", f1_name='salary', f2_name='bonus')
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

##Finding the best features giving highest f1 score
# for ii in range(1,len(features_list)):
#     selection = SelectKBest(k=ii)
#     features_new = selection.fit_transform(features, labels)
#     features_list_new = ['poi'] + [features_list[1:][i] for i in np.where(selection.get_support()==True)[0]]
#     print features_list_new

#     test_classifier(clf, my_dataset, features_list_new)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print 'accuracy: ', accuracy_score(labels, pred)
print 'precision: ', precision_score(labels, pred, average='micro')
print 'recall: ', recall_score(labels, pred, average='micro')
print 'f1_score: ', f1_score(labels, pred, average='micro')

test_classifier(clf, my_dataset, features_list)
### Dump your classifier, dataset, and features_list so
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)


