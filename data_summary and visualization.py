#!/usr/bin/python

""" 
    code for printing summary and visualising data with scatterplots
    
"""

import pickle
import pprint
import math
import csv
import numpy as np
from sklearn.preprocessing import Imputer, scale
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import sys
from time import time
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.decomposition import RandomizedPCA
import os


def get_summary(input_dict,key_list, feature_list):
    """ The function calculates basic summary from the dict
        and shows it in the form of dict
    """
    
    output_dict={}
    for feature in feature_list:
        cnt_NA=0
        cnt=0
        sm=0.0
        mn=None
        mx=None
        for key in key_list:
            value=input_dict[key][feature]
            #print(value)
            if value=='NaN':
                cnt_NA+=1
            else:
                cnt+=1
                if (feature!='email_address'):
                    sm+=value
                    if mn==None:
                        mn=value
                    elif (mn>value):
                        mn=value
                    if mx==None:
                        mx=value
                    elif (mx<value):
                        mx=value

        if cnt>0:
             mean=sm/cnt
        else:
            mean=0
           
        dct={'name':feature,
             'cnt_NA':cnt_NA,
             'cnt':cnt,
             'cnt_all':cnt+cnt_NA,
             'sum':sm,
             'min':mn,
             'max':mx,
             'mean':mean}
        

        output_dict[feature]=dct

    return output_dict

def fix_emails(enron_data):
    # add email addresses for
    #GATHMANN WILLIAM D
    #LOCKHART EUGENE E
    #NOLES JAMES L
    ######## not used in the submission
    enron_data['GATHMANN WILLIAM D']['email_address']='bill.gathmann@enron.com'
    enron_data['LOCKHART EUGENE E']['email_address']='gene.lockhart@enron.com'
    enron_data['NOLES JAMES L']['email_address']='james.noles@enron.com'

    return enron_data
    

def dict_to_table(input_dict,fieldnames):
    ## converts dict to a fancy table suitable for printing
    output_table = PrettyTable(fieldnames)

    for row in sorted(input_dict.keys()):
        aa=[]
        for i in range(0,len(fieldnames)):
            if(fieldnames[i]=='mean'):
                if(row=='poi'):
                    ndigits=3
                else:
                    ndigits=1
                aa.append(round(input_dict[row][fieldnames[i]],ndigits))
            elif(fieldnames[i] in ['min','max','mean','sum'] and
                 row in ['from_poi_to_this_person_ratio',
                           'from_this_person_to_poi_ratio', 'shared_receipt_with_poi_ratio']):
                ndigits=3
                aa.append(round(input_dict[row][fieldnames[i]],ndigits))
            else:
                aa.append(input_dict[row][fieldnames[i]])
        output_table.add_row(aa)
    return output_table

def dict_to_nparray(input_dict,fieldnames):
    ## converts dict to a numpy array to be used for classification
    ## and other tasks
    output_array = []

    for row in input_dict.keys():
        aa=[]
        for i in range(0,len(fieldnames)):
            if fieldnames[i]=='poi':
                aa.append(int(input_dict[row][fieldnames[i]]))
            elif fieldnames[i]=='email_address':
                if input_dict[row][fieldnames[i]]=='NaN':
                    aa.append(0)
                else:
                    aa.append(1)
            else:
                aa.append(input_dict[row][fieldnames[i]])
        output_array.append(aa)
    return np.array(output_array)




######################################################
## function for scatterplot with 2 variables from dict

def plot_figure(data,feature1,feature2,show_NaN=True):
    plt.figure()
    cnt=0
    cnt0=0
    points=[]
    for entry in data:
        
        x=data[entry][feature1]
        y=data[entry][feature2]
        z='normal'

    
        if (x=="NaN" and show_NaN==True):
            x=0.
            z='NaN'
        if (y=="NaN" and show_NaN==True):
            y=0.
            z='NaN'

        if data[entry]['poi']==True:
            z='poi'

            
        if x!="NaN" and y!="NaN":
            points.append([x,y,z])

    
    colors=['r','b','y']
    
    xx=[]
    yy=[]
    for point in points:
        if point[2]=='poi':
            xx.append(point[0])
            yy.append(point[1])
    p1= plt.scatter(xx,yy, marker='x', color=colors[0])

    
    xx=[]
    yy=[]
    for point in points:
        if point[2]=='normal':
            xx.append(point[0])
            yy.append(point[1])
    p0= plt.scatter(xx,yy, marker='o', color=colors[1])
    
    
    xx=[]
    yy=[]
    for point in points:
        if point[2]=='NaN':
            xx.append(point[0])
            yy.append(point[1])
    pn= plt.scatter(xx,yy, marker='o', color=colors[2])
    

    plt.xlabel(feature1)
    plt.ylabel(feature2)
    
    plt.legend((p1, p0, pn),
           ('POI', 'Not POI', 'Not POI (imputed values)'),
           scatterpoints=1,
           loc='upper left',
           ncol=1,
           fontsize=8)

    
    plt.show()
    ############################################


def main():
    enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


# uncomment to save data as csv
 #   with open('data.csv', 'w') as csvfile:
 #       fieldnames = ['name']+enron_data[enron_data.keys()[0]].keys()
 #       writer = csv.DictWriter(csvfile, fieldnames=fieldnames,delimiter=',', lineterminator='\n')
#
 #       writer.writeheader()
 #       for item in enron_data:
 #           entry=enron_data[item]
 #           entry['name']=item
 #           writer.writerow(entry)
            
        
       
    ## delete outlier
    del enron_data['TOTAL']

    ## uncomment to add 3 email addresses to the dict
    #enron_data=fix_emails(enron_data)

    ## transform 'from_this_person_to_poi' ,'from_poi_to_this_person', and
    ## 'shared_receipt_with_poi' to ratios (i.e. values from 0 to 1)
    for person in enron_data.keys():
        all_to=enron_data[person]['to_messages']
        all_from=enron_data[person]['from_messages']
        to_poi=enron_data[person]['from_this_person_to_poi']
        from_poi=enron_data[person]['from_poi_to_this_person']
        sr_with_poi=enron_data[person]['shared_receipt_with_poi']
        if all_to=='NaN' or from_poi=='NaN':
            enron_data[person]['from_poi_to_this_person_ratio']='NaN'
        else:
            enron_data[person]['from_poi_to_this_person_ratio']=1.0*from_poi/all_to
        if all_from=='NaN' or to_poi=='NaN':
            enron_data[person]['from_this_person_to_poi_ratio']='NaN'
        else:
            enron_data[person]['from_this_person_to_poi_ratio']=1.0*to_poi/all_from
        if all_to=='NaN' or sr_with_poi=='NaN':
            enron_data[person]['shared_receipt_with_poi_ratio']='NaN'
        else:
            enron_data[person]['shared_receipt_with_poi_ratio']=1.0*sr_with_poi/all_to
            
    for person in enron_data:
        x=enron_data[person]['from_poi_to_this_person_ratio']
        if x>1 and x!='NaN':
            print enron_data[person]

    person_list=enron_data.keys()
    feature_list=enron_data[enron_data.keys()[1]].keys()


    plot_figure(enron_data,'from_poi_to_this_person_ratio','shared_receipt_with_poi_ratio',show_NaN=True)

    # uncomment to get summary on part of the data
    """
    person_list1=[]
    for person in person_list:
        if enron_data[person]['poi']!=True:
            person_list1.append(person)
    person_list=person_list1
    """
    
    ## create summary and print it as PrettyTable
    summary=get_summary(enron_data,person_list, feature_list)
    tbl=dict_to_table(summary,['name', 'cnt_all', 'cnt', 'cnt_NA',
                               'min','max','mean','sum'])
    print tbl

         



if __name__ == '__main__':
    main()
