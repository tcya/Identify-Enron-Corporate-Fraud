#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = []

    ### your code goes here
    errors = map(lambda x, y:x-y, net_worths, predictions)
    data = zip(ages, net_worths, errors)
    data = sorted(data, key=lambda x:abs(x[2]), reverse=True)
    cleaned_data = data[int(len(errors)*0.1):]
    return cleaned_data