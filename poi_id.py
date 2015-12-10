#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.cross_validation import KFold, StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.preprocessing import Imputer, scale, StandardScaler

from sklearn.feature_selection import SelectKBest, f_classif
from prettytable import PrettyTable

import numpy as np
from time import time
import matplotlib.pyplot as plt

from tester import test_classifier
from sklearn import grid_search
from sklearn.pipeline import Pipeline

from sklearn.metrics import make_scorer,fbeta_score

## manually implemented function for scoring
## penalize_below_03_coef is used to penalize recall and precision below 0.3
def my_score_func(y_true, y_pred):
    rec=recall_score(y_true, y_pred)
    pre=precision_score(y_true, y_pred)
    #if rec<0.35 and pre<0.35:
    #    score=-0.1-(rec-0.35)*(pre-0.35)
    #else:
    #    score=(rec-0.35)*(pre-0.35)
    score=min(rec,0.5)*min(pre,0.5) + 0.5*rec
    
    return score


def dict_to_nparray(input_dict,fieldnames):
    ## converts dict to a numpy array to be used for classification
    ## and other tasks
    output_array = []
    names_arr = []
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
        names_arr.append(row)
    return np.array(output_array),np.array(names_arr)

def add_to_ptable(ptable,np_array,method):
    if(method=='mean'):
        ptable.add_row([method,
                        np.mean(np_array[:,1]),
                        np.mean(np_array[:,2]),
                        np.mean(np_array[:,3])])
    elif (method=='median'):
        ptable.add_row([method,
                        np.median(np_array[:,1]),
                        np.median(np_array[:,2]),
                        np.median(np_array[:,3])])
    elif (method=='min'):
        ptable.add_row([method,
                        np.min(np_array[:,1]),
                        np.min(np_array[:,2]),
                        np.min(np_array[:,3])])
    elif (method=='max'):
        ptable.add_row([method,
                        np.max(np_array[:,1]),
                        np.max(np_array[:,2]),
                        np.max(np_array[:,3])])
    return ptable

## This function uses StratifiedShuffleSplit to make N_folds partitions
## and returns the algorithm perfomance on these partitions
def  KFold_summary(features, labels, clf, N_folds,test_size,short):
    results_ptable = PrettyTable(["iteration", "accuracy",
                                  "recall", "precision"])
    results_arr=[]
    cnt=0
    kf= StratifiedShuffleSplit(labels,n_iter=N_folds,test_size=test_size,random_state=42)
    for train_indices, test_indices in kf:
        cnt+=1
        features_train =[features[ii] for ii in train_indices]
        features_test =[features[ii] for ii in test_indices]
        labels_train =[labels[ii] for ii in train_indices]
        labels_test =[labels[ii] for ii in test_indices]

        clf.fit(features_train,labels_train)
        acc=accuracy_score(labels_test, clf.predict(features_test))
        rec=recall_score(labels_test, clf.predict(features_test))
        pre=precision_score(labels_test, clf.predict(features_test))
    
        results_arr.append([cnt,round(acc,3),round(rec,3),round(pre,3)])

    if short==False:
        for item in np.array(results_arr):
            results_ptable.add_row(item)
    add_to_ptable(results_ptable,np.array(results_arr),'mean')
    add_to_ptable(results_ptable,np.array(results_arr),'median')
    add_to_ptable(results_ptable,np.array(results_arr),'min')

    return results_arr, results_ptable


## This function is similar to KFold_summary, but return only feature 
## importances averaged over all partitions
def  KFold_feature_importances(features, labels, clf, N_folds,test_size):
    
    results_arr=[]
    cnt=0
    kf= StratifiedShuffleSplit(labels,n_iter=N_folds,test_size=test_size,random_state=42)
    for train_indices, test_indices in kf:
        cnt+=1
        features_train =[features[ii] for ii in train_indices]
        features_test =[features[ii] for ii in test_indices]
        labels_train =[labels[ii] for ii in train_indices]
        labels_test =[labels[ii] for ii in test_indices]

        clf.fit(features_train,labels_train)
        acc=accuracy_score(labels_test, clf.predict(features_test))
        rec=recall_score(labels_test, clf.predict(features_test))
        pre=precision_score(labels_test, clf.predict(features_test))

        
        results_arr.append(clf.feature_importances_)
    output_arr=[]
    for i in range(0,len(results_arr[0])):
        output_arr.append(np.mean(np.array(results_arr)[:,i]))

    return output_arr

#returns accuracy, recall and precision for selectKBest given n_select features
def  KFold_Kbest_summary(features, labels, clf, N_folds,test_size,n_select):
    results_ptable = PrettyTable(["iteration", "accuracy",
                                  "recall", "precision"])
    results_arr=[]
    cnt=0

    skb=SelectKBest(score_func=f_classif, k=n_select)
    features=skb.fit_transform(features,labels)
    
    kf= StratifiedShuffleSplit(labels,n_iter=N_folds,test_size=test_size,random_state=42)
    for train_indices, test_indices in kf:
        cnt+=1
        features_train =[features[ii] for ii in train_indices]
        features_test =[features[ii] for ii in test_indices]
        labels_train =[labels[ii] for ii in train_indices]
        labels_test =[labels[ii] for ii in test_indices]


        #skb=SelectKBest(score_func=f_classif, k=n_select)
        #features_train=skb.fit_transform(features_train,labels_train)
        #features_test=skb.transform(features_test)
        
        clf.fit(features_train,labels_train)
        acc=accuracy_score(labels_test, clf.predict(features_test))
        rec=recall_score(labels_test, clf.predict(features_test))
        pre=precision_score(labels_test, clf.predict(features_test))
    
        results_arr.append([cnt,acc,rec,pre])

    return np.mean(np.array(results_arr)[:,1]), np.mean(np.array(results_arr)[:,2]), np.mean(np.array(results_arr)[:,3])



## runs KFold_Kbest_summary over all possible number of features
## (e.g. from 1 to len(all_features_list))
def performance_KBest(features,labels,all_features_list,clf,N_folds=200,test_size=0.3):
    perf_list=[]
    for k in range(1,len(all_features_list)):
        acc,rec,pre = KFold_Kbest_summary(features, labels, clf=clf, N_folds=N_folds,test_size=test_size,n_select=k)
        perf_list.append([k,acc,rec,pre])

    return np.array(perf_list)



####################################################################
### Task 1: [Also please see Task 3.7] Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# all financial features
all_financial_features=['deferral_payments', 'expenses', 'deferred_income', 
                    'long_term_incentive', 'restricted_stock_deferred', 'loan_advances',
                    'other', 'director_fees', 'bonus', 'total_stock_value',
                    'restricted_stock', 'salary', 'total_payments', 'exercised_stock_options']

# financial features used in the first submission
financial_features=['expenses', 'other', 'bonus', 'total_stock_value',
                    'restricted_stock', 'salary', 'total_payments', 'exercised_stock_options'] 

#email_features to be created
email_features=['from_this_person_to_poi_ratio','from_poi_to_this_person_ratio',
                'shared_receipt_with_poi_ratio']
#text features to be created
text_features = ['ena', 'ect', 'guy'] # these features were used in the first submission


features_list = financial_features + email_features + text_features

## Later I will also create text_features 
## For the classifier I will use a combination
## of financial, email and text_features

### Load the dictionary containing the dataset
enron_data = pickle.load(open("final_project_dataset.pkl", "r") )


#####################################################################
### Task 2: Remove outliers

del enron_data['TOTAL']

#####################################################################
### Task 3: Create new feature(s)

## 3.1 New features from the available dataset
## I transform 'from_this_person_to_poi' ,'from_poi_to_this_person', and
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
            
## 3.2 New text features from email corpus
## Features are the best words to predict whether an email was sent by POI.
## In the model I will use their frequencies - how often a particular word (feature)
## appears in emails from particular person

load_dict={}
import csv
with open('text_features.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        key=row['email_address']
        del row['email_address']
        load_dict[key]=row
    

all_text_features=load_dict[load_dict.keys()[0]].keys()
all_text_features.remove('cnt_emails')


## loaded is email count with particular feature
## now it's time to transform the data info fractions between 0 and 1
for entry in load_dict:
    if load_dict[entry]['cnt_emails']!="0":
        for feature in all_text_features:
            load_dict[entry][feature]=float(load_dict[entry][feature])/float(load_dict[entry]['cnt_emails'])

## merge enron_data with text features on email_address
for entry in enron_data:
    email_address=enron_data[entry]['email_address']
    if (email_address!="NaN"):
        for feature in all_text_features:
            if int(load_dict[email_address]['cnt_emails'])>=10:
                enron_data[entry][feature]=load_dict[email_address][feature]
            else:
                enron_data[entry][feature]="NaN"
    else:
        for feature in all_text_features:
            enron_data[entry][feature]="NaN"        

       

### Store to my_dataset for easy export below.
my_dataset = enron_data

#print len(my_dataset.keys())



########################################################################
#### Task 3.3 - Prepare dataset to be used by classification functions
########################################################################
all_features_list = all_financial_features + email_features + all_text_features
print 'Importing text features:', all_text_features

# imput features to the feature selection process
labels, names1 = dict_to_nparray(my_dataset, ['poi'])
labels=labels.reshape(1,len(labels))[0]
features, names2=dict_to_nparray(my_dataset, all_features_list)

if np.all(names1==names2):
    print "\nArrays are ordered properly"
else:
    print "\nError in data ordering"


## in financial feature replace 'NaN' with 0
for i in range(0,len(all_financial_features)):
        features[:,i][features[:,i]=='NaN']=0

# in other features replace 'NaN' with median
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(features)
features=imp.transform(features)

#scale financial features
features_scaled=scale(features)
for i in range (0,len(all_financial_features)):
    features[:,i]=features_scaled[:,i]


############## save features and labels for future use
features_saved=np.asarray(features)
labels_saved=np.asarray(labels)
all_features_list_saved=list(all_features_list)


print "Number of all features: "+str(len(all_features_list_saved))
##############





#################################################################################
### Step 3.8 - convert data to be compatible with featureFormat function
###
dataset_to_export={}
for i in range(0,len(labels)):
    entry={}
    for k in range(0,len(all_features_list_saved)):
        entry[all_features_list_saved[k]]=features_saved[i,k]
    entry['poi']=labels_saved[i]
    dataset_to_export[names1[i]]=entry




### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


## keep only features from features_list 
i=len(all_features_list)-1
while i>=0:
    if (all_features_list[i] not in features_list):
        del all_features_list[i]
        features=np.delete(features,i,1)
    i=i-1

if all_features_list==features_list:
    print "Only features from features_list are present"
else:
    print "Oops... Too many features."

all_features_list=list(all_features_list_saved)

print "\n"
print "***** WITHOUT TRAIN/TEST SPLIT *****"
print '----- NB classifier -----'
t0 = time()

clf = GaussianNB()
clf.fit(features,labels)
pred=clf.predict(features)
print "   accuracy:", accuracy_score(labels, pred)
print "   recall:", recall_score(labels, pred)
print "   precision:", precision_score(labels, pred)
print 'Time:',round(time()-t0,3) ,'s\n'
t0 = time()


print '----- SVM classifier -----'
t0 = time()
clf=SVC(kernel="rbf", C=10000.0)
clf.fit(features,labels)
pred=clf.predict(features)
print "   accuracy:", accuracy_score(labels, pred)
print "   recall:", recall_score(labels, pred)
print "   precision:", precision_score(labels, pred)
print 'Time:',round(time()-t0,3) ,'s\n'
t0 = time()



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print "***** WITH TRAIN/TEST SPLIT *****"

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

print '----- NB classifier -----'
t0 = time()

clf = GaussianNB()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print "   accuracy:", accuracy_score(labels_test, pred)
print "   recall:", recall_score(labels_test, pred)
print "   precision:", precision_score(labels_test, pred)
print 'Time:',round(time()-t0,3) ,'s\n'
t0 = time()


print '----- SVM classifier -----'
t0 = time()
clf=SVC(kernel="rbf", C=10000.0,random_state=42)
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print "   accuracy:", accuracy_score(labels_test, pred)
print "   recall:", recall_score(labels_test, pred)
print "   precision:", precision_score(labels_test, pred)
print 'Time:',round(time()-t0,3) ,'s\n'
t0 = time()

print '----- Decision Tree classifier -----'
t0 = time()
clf = tree.DecisionTreeClassifier(min_samples_leaf=1,random_state=42)
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print "   accuracy:", accuracy_score(labels_test, pred)
print "   recall:", recall_score(labels_test, pred)
print "   precision:", precision_score(labels_test, pred)
print 'Time:',round(time()-t0,3) ,'s\n'
t0 = time()





#####################
## K-Fold validation 

clf=SVC(kernel="rbf", C=10000.0,random_state=42)
_,ptable=KFold_summary(features, labels, clf, N_folds=10,test_size=0.3,short=False)
print "\n-- Summary for SVM 10-fold validation --"
print ptable

clf = tree.DecisionTreeClassifier(min_samples_leaf=1,random_state=42)
_,ptable=KFold_summary(features, labels, clf, N_folds=10,test_size=0.3,short=False)
print "\n-- Summary for Decision Tree 10-fold validation --"
print ptable

#####################
### Try to implement Implement PCA to avoid overfitting
from sklearn.decomposition import RandomizedPCA
N_features=7 # number of features to retain after PCA
pca = RandomizedPCA(n_components=N_features)
pca.fit(features)
print "\nPCA explained variances:"
print(pca.explained_variance_ratio_) 

features_pca = pca.transform(features)

clf=SVC(kernel="rbf", C=10000.0,random_state=42)
_,ptable=KFold_summary(features_pca, labels, clf, N_folds=1000,test_size=0.3,short=True)
print "\n-- Summary for 'SVM after PCA (n_features="+str(N_features)+\
      ")' 1000-fold validation --"
print ptable

clf = tree.DecisionTreeClassifier(min_samples_leaf=1,random_state=42)
_,ptable=KFold_summary(features_pca, labels, clf, N_folds=1000,test_size=0.3,short=True)
print "\n-- Summary for Decision Tree after PCA (n_features="+str(N_features)+\
      ")' 1000-fold validation --"
print ptable

print "......The results are worse after PCA"
##########################


##########################
## Trying different parameter values:
print "\n----------- TUNING SVM MODEL -----------"
for kernel in ["linear"]: #["linear", "rbf"]:
    for C_value in [1.0,10.0, 100.0]:
        clf=SVC(kernel=kernel, C=C_value, random_state=42)
        _,ptable=KFold_summary(features, labels, clf, N_folds=1000,test_size=0.3,short=True)
        print "\nTune C -- kernel="+kernel+", C="+str(C_value)
        print ptable
print "   LINEAR KERNEL: high precision, but low recall."

for kernel in ["rbf"]: #["linear", "rbf"]:
    for C_value in [1.0,100.0, 1000., 10000.0]:
        clf=SVC(kernel=kernel, C=C_value, random_state=42)
        _,ptable=KFold_summary(features, labels, clf, N_folds=1000,test_size=0.3,short=True)
        print "\nTune C -- kernel="+kernel+", C="+str(C_value)
        print ptable

for kernel in ["rbf"]:
    for gamma_value in [0.0001,0.001, 0.003, 0.005, 0.007, 0.01,0.02]:
        clf=SVC(kernel=kernel, C=10000.0, gamma=gamma_value,random_state=42)
        _,ptable=KFold_summary(features, labels, clf, N_folds=1000,test_size=0.3,short=True)
        print "\nTune gamma -- kernel="+kernel+", gamma="+str(gamma_value)+", C="+str(10000.0)
        print ptable

print "   kernel='rbf': I select C=10000.0, gamma=0.007"



print "\n----------- TUNING DECISION TREE MODEL -----------"

for max_depth in [None, 5,3]:
    for min_samples_leaf in [1,2]:
        clf = tree.DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,max_depth=max_depth,random_state=42)
        _,ptable=KFold_summary(features, labels, clf, N_folds=1000,test_size=0.3,short=True)
        print "\nTune DT model -- max_depth="+str(max_depth)+", min_samples_leaf="+str(min_samples_leaf)
        print ptable
print "   HIGHEST RECALL: max_depth=None, HIGHEST PRECISION: max_depth=3"
print "   I select: min_samples_leaf=1, max_depth=None"


print "\n----------- FINAL MODELS: Tune number of features -----------"
nb_clf = GaussianNB()
dt_clf = tree.DecisionTreeClassifier(min_samples_leaf=1,max_depth=None,random_state=42)
svm_clf=SVC(kernel="rbf", C=10000.0, gamma=0.007,random_state=42)




_,ptable=KFold_summary(features, labels, svm_clf, N_folds=1000,test_size=0.3,short=True)
print "\n-- SVM: Financial + email + text features"
print ptable
_,ptable=KFold_summary(features[:,1:(len(financial_features)+len(email_features))], labels, svm_clf, N_folds=1000,test_size=0.3,short=True)
print "\n-- SVM: Financial + email features"
print ptable
_,ptable=KFold_summary(features[:,1:(len(financial_features))], labels, svm_clf, N_folds=1000,test_size=0.3,short=True)
print "\n-- SVM: Financial features only"
print ptable

"""
cols=[ii for ii in range(1,(len(financial_features)))]+\
      [ii for ii in range((len(financial_features)+len(email_features)),(len(features_list)))]
_,ptable=KFold_summary(features[:,cols], labels, svm_clf, N_folds=10,test_size=0.3,short=True)
print "\n-- SVM: Financial features only"
print ptable
    """

_,ptable=KFold_summary(features, labels, dt_clf, N_folds=1000,test_size=0.3,short=True)
print "\n-- Decision tree: Financial + email + text features"
print ptable
_,ptable=KFold_summary(features[:,1:(len(financial_features)+len(email_features))], labels, dt_clf, N_folds=1000,test_size=0.3,short=True)
print "\n-- Decision tree: Financial + email features"
print ptable
_,ptable=KFold_summary(features[:,1:(len(financial_features))], labels, dt_clf, N_folds=1000,test_size=0.3,short=True)
print "\n-- Decision tree: Financial features only"
print ptable



_,ptable=KFold_summary(features, labels, nb_clf, N_folds=1000,test_size=0.3,short=True)
print "\n-- Naive Bayes: Financial + email + text features"
print ptable
_,ptable=KFold_summary(features[:,1:(len(financial_features)+len(email_features))], labels, nb_clf, N_folds=1000,test_size=0.3,short=True)
print "\n-- Naive Bayes: Financial + email features"
print ptable
_,ptable=KFold_summary(features[:,1:(len(financial_features))], labels, nb_clf, N_folds=1000,test_size=0.3,short=True)
print "\n-- Naive Bayes: Financial features only"
print ptable



print "\nAccording to the results above, SVM model has better results."

print "\nOutput of tester.py."
test_classifier(svm_clf, dataset_to_export, ['poi']+features_list)

##################################################################################
##### Stet 5.5 Implement automatic feature selection process
##################################################################################
#nb_clf = GaussianNB()
#dt_clf = tree.DecisionTreeClassifier(min_samples_leaf=1,max_depth=None,random_state=42)
#svm_clf=SVC(kernel="rbf", C=10000.0, gamma=0.007,random_state=42)



#recover all features and feature names
features=np.asarray(features_saved)
labels=np.asarray(labels_saved)
all_features_list=list(all_features_list_saved)


importances=KFold_feature_importances(features, labels, dt_clf, N_folds=1000,test_size=0.3)
print "\n-- Arrays of features and their importances"
print all_features_list
print importances


NN=len(importances)
KFolds=75 #how many calculations to perform for each combination of model:number_of_selected_features

t0 = time()
svm_perf1=performance_KBest(features,labels,all_features_list,svm_clf,N_folds=KFolds,test_size=0.3)
print 'Time:',round(time()-t0,3) ,'s\n'
t0 = time()
nb_perf1=performance_KBest(features,labels,all_features_list,nb_clf,N_folds=KFolds,test_size=0.3)

print 'Time:',round(time()-t0,3) ,'s\n'
t0 = time()
dt_perf1=performance_KBest(features,labels,all_features_list,dt_clf,N_folds=KFolds,test_size=0.3)
print 'Time:',round(time()-t0,3) ,'s\n'
t0 = time()

"""
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(svm_perf1[:,0], svm_perf1[:,2], 'g^')
ax1.plot(svm_perf1[:,0], svm_perf1[:,3], 'bo')
ax1.axhline(y=0.3)
ax1.set_title('SVM')
ax2.plot(dt_perf1[:,0], dt_perf1[:,2], 'g^')
ax2.plot(dt_perf1[:,0], dt_perf1[:,3], 'bo')
ax2.axhline(y=0.3)
ax2.set_title('DT')
ax3.plot(nb_perf1[:,0], nb_perf1[:,2], 'g^')
ax3.plot(nb_perf1[:,0], nb_perf1[:,3], 'bo')
ax3.axhline(y=0.3)
ax3.set_title('NB')

f.subplots_adjust(hspace=0)
plt.show()
"""

#delete some features of low importance
i=len(importances)-1
while i>=0:
    if (importances[i]<0.33*1/len(importances)):
        del all_features_list[i]
        features=np.delete(features,i,1)
    i=i-1

#fit the models again
importances=KFold_feature_importances(features, labels, dt_clf, N_folds=1000,test_size=0.3)
print "\n-- Arrays of features and their importances"
print all_features_list
print importances

t0 = time()
svm_perf2=performance_KBest(features,labels,all_features_list,svm_clf,N_folds=KFolds,test_size=0.3)
print 'Time:',round(time()-t0,3) ,'s\n'
t0 = time()
nb_perf2=performance_KBest(features,labels,all_features_list,nb_clf,N_folds=KFolds,test_size=0.3)

print 'Time:',round(time()-t0,3) ,'s\n'
t0 = time()
dt_perf2=performance_KBest(features,labels,all_features_list,dt_clf,N_folds=KFolds,test_size=0.3)
print 'Time:',round(time()-t0,3) ,'s\n'
t0 = time()

"""
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(svm_perf2[:,0], svm_perf2[:,2], 'g^')
ax1.plot(svm_perf2[:,0], svm_perf2[:,3], 'bo')
ax1.axhline(y=0.3)
ax1.set_title('SVM')
ax2.plot(dt_perf2[:,0], dt_perf2[:,2], 'g^')
ax2.plot(dt_perf2[:,0], dt_perf2[:,3], 'bo')
ax2.axhline(y=0.3)
#ax2.xlim([0, NN])
ax2.set_title('DT')
ax3.plot(nb_perf2[:,0], nb_perf2[:,2], 'g^')
ax3.plot(nb_perf2[:,0], nb_perf2[:,3], 'bo')
ax3.axhline(y=0.3)
ax3.set_title('NB')

f.subplots_adjust(hspace=0)
x1,x2,y1,y2=plt.axis()
plt.axis((0,NN,y1,y2))

plt.show()
"""


#delete some features of low importance
i=len(importances)-1
while i>=0:
    if (importances[i]<0.33*1/len(importances)):
        del all_features_list[i]
        features=np.delete(features,i,1)
    i=i-1

#fit the models again
importances=KFold_feature_importances(features, labels, dt_clf, N_folds=1000,test_size=0.3)
print "\n-- Arrays of features and their importances"
print all_features_list
print importances

t0 = time()
svm_perf3=performance_KBest(features,labels,all_features_list,svm_clf,N_folds=KFolds,test_size=0.3)
print 'Time:',round(time()-t0,3) ,'s\n'
t0 = time()
nb_perf3=performance_KBest(features,labels,all_features_list,nb_clf,N_folds=KFolds,test_size=0.3)

print 'Time:',round(time()-t0,3) ,'s\n'
t0 = time()
dt_perf3=performance_KBest(features,labels,all_features_list,dt_clf,N_folds=KFolds,test_size=0.3)
print 'Time:',round(time()-t0,3) ,'s\n'
t0 = time()

"""
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(svm_perf3[:,0], svm_perf3[:,2], 'g^')
ax1.plot(svm_perf3[:,0], svm_perf3[:,3], 'bo')
ax1.axhline(y=0.3)
ax1.set_title('SVM')
ax2.plot(dt_perf3[:,0], dt_perf3[:,2], 'g^')
ax2.plot(dt_perf3[:,0], dt_perf3[:,3], 'bo')
ax2.axhline(y=0.3)
ax2.set_title('DT')
ax3.plot(nb_perf3[:,0], nb_perf3[:,2], 'g^')
ax3.plot(nb_perf3[:,0], nb_perf3[:,3], 'bo')
ax3.axhline(y=0.3)
ax3.set_title('NB')

f.subplots_adjust(hspace=0)

x1,x2,y1,y2=plt.axis()
plt.axis((0,NN,y1,y2))
plt.show()
"""

#### Plots showing performance vs number of features

plt.plot(svm_perf1[:,0], svm_perf1[:,2], 'g:')
plt.plot(svm_perf1[:,0], svm_perf1[:,3], 'b:')
plt.axhline(y=0.3)
plt.plot(svm_perf2[:,0], svm_perf2[:,2], 'g--')
plt.plot(svm_perf2[:,0], svm_perf2[:,3], 'b--')
plt.axhline(y=0.3)
rec,=plt.plot(svm_perf3[:,0], svm_perf3[:,2], 'g-',label='Recall')
pre,=plt.plot(svm_perf3[:,0], svm_perf3[:,3], 'b-',label='Precision')
plt.legend(handles=[rec, pre])
plt.title('SVM')
plt.xlabel('Features included')
print "\nPlease check plot window"
plt.show()


plt.plot(dt_perf1[:,0], dt_perf1[:,2], 'g:')
plt.plot(dt_perf1[:,0], dt_perf1[:,3], 'b:')
plt.axhline(y=0.3)
plt.plot(dt_perf2[:,0], dt_perf2[:,2], 'g--')
plt.plot(dt_perf2[:,0], dt_perf2[:,3], 'b--')
plt.axhline(y=0.3)
rec,=plt.plot(dt_perf3[:,0], dt_perf3[:,2], 'g-',label='Recall')
pre,=plt.plot(dt_perf3[:,0], dt_perf3[:,3], 'b-',label='Precision')
plt.legend(handles=[rec, pre])
plt.title('Decision tree')
plt.xlabel('Features included')
print "\nPlease check plot window"
plt.show()


plt.plot(nb_perf1[:,0], nb_perf1[:,2], 'g:')
plt.plot(nb_perf1[:,0], nb_perf1[:,3], 'b:')
plt.axhline(y=0.3)
plt.plot(nb_perf2[:,0], nb_perf2[:,2], 'g--')
plt.plot(nb_perf2[:,0], nb_perf2[:,3], 'b--')
plt.axhline(y=0.3)
rec,=plt.plot(nb_perf3[:,0], nb_perf3[:,2], 'g-',label='Recall')
pre,=plt.plot(nb_perf3[:,0], nb_perf3[:,3], 'b-',label='Precision')
plt.legend(handles=[rec, pre])
plt.title('Naive Bayes')
plt.xlabel('Features included')
print "\nPlease check plot window"
plt.show()

##################################################################################
##### Implement GridSearchCV for SVM
print "\nImplementing GridSearchCV for SVM. Please wait a few minutes..."
t0 = time()
clf_params= {
                       'skbest__k': [8,10,12,14,16,18,20,22,24,26,28,30,32],
                       'clf__C': [1e-2,1e-1, 1, 10, 1e2, 1e4],
                       'clf__gamma': [0.001, 0.005,  0.01, 0.05],
                       'clf__kernel': ['rbf'],
                       #'clf__tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],  
                       'clf__class_weight': [{True: 8, False: 1},
                                               {True: 4, False: 1},
                                               'auto', None]
                      }
# my scoring function which penalizes small precision and (especially) recall values
#my_scoring = make_scorer(my_score_func, greater_is_better=True)
my_scorer = make_scorer(fbeta_score, beta=0.5)


pipe = Pipeline(steps=[('skbest', SelectKBest(score_func=f_classif)), ('clf', SVC())])
cv = StratifiedShuffleSplit(labels,n_iter = 60,random_state = 42)
a_grid_search = grid_search.GridSearchCV(pipe, param_grid = clf_params,cv = cv,scoring = my_scorer)
a_grid_search.fit(features_saved,labels_saved)

print 'Time:',round(time()-t0,3) ,'s\n'
t0 = time()

# pick a winner
best_clf = a_grid_search.best_estimator_
print best_clf

found_skb=best_clf.steps[0][1]
found_clf=best_clf.steps[1][1]

features=found_skb.fit_transform(features_saved,labels_saved)
features_list_to_use=np.asarray(all_features_list_saved)[found_skb.get_support()].tolist()
print "\nFeatures used:"
print features_list_to_use

test_classifier(found_clf, dataset_to_export, ['poi']+features_list_to_use)

##################################################################################
##### Implement GridSearchCV for Naive Bayes
print "\nImplementing GridSearchCV for Naive Bayes"
t0 = time()
clf_params= {'skbest__k': range(1,len(features_saved[0]))}

pipe = Pipeline(steps=[('skbest', SelectKBest(score_func=f_classif)), ('clf', GaussianNB())])
cv = StratifiedShuffleSplit(labels,n_iter = 60,random_state = 42)
b_grid_search = grid_search.GridSearchCV(pipe, param_grid = clf_params,cv = cv,scoring = 'precision')
b_grid_search.fit(features_saved,labels_saved)

print 'Time:',round(time()-t0,3) ,'s\n'
t0 = time()

# pick a winner
best_clf_nb = b_grid_search.best_estimator_
print best_clf_nb

found_skb_nb=best_clf_nb.steps[0][1]
found_clf_nb=best_clf_nb.steps[1][1]

features=found_skb_nb.fit_transform(features_saved,labels_saved)
features_list_to_use_nb=np.asarray(all_features_list_saved)[found_skb_nb.get_support()].tolist()
print "\nFeatures used:"
print features_list_to_use_nb

test_classifier(found_clf_nb, dataset_to_export, ['poi']+features_list_to_use_nb)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(svm_clf, dataset_to_export, ['poi']+features_list)
