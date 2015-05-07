#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print len(enron_data)
# print enron_data.keys()
print len(enron_data['METTS MARK'])
print [enron_data[i]['poi'] for i in enron_data].count(True)
print enron_data['PRENTICE JAMES']
print enron_data['COLWELL WESLEY']
# print sorted(enron_data.keys())
print enron_data['SKILLING JEFFREY K']
print [enron_data[i]['total_payments'] for i in ['SKILLING JEFFREY K', 'LAY KENNETH L', 'FASTOW ANDREW S']]
print len(enron_data) - [enron_data[i]['salary'] for i in enron_data].count('NaN')
print len(enron_data) - [enron_data[i]['email_address'] for i in enron_data].count('NaN')
print [enron_data[i]['total_payments'] for i in enron_data].count('NaN')
print [enron_data[i]['total_payments'] for i in enron_data if enron_data[i]['poi'] == True].count('NaN')
