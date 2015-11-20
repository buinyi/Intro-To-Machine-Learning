#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string
import os
import pprint
import csv
import pickle
from time import time
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy


def parseText(f, email_list, poi_indicator_list):
    """
        Return the stemmed version of email document, sender's email address
        and indication whether the sender is POI.
        
        open email file f
        split it for metadata and email text
        find the sender's email address (from_field)
        if from_field in our email database, then proceed.
            Otherwise skip the email and return empty string.
            .. if poi's email found, the function will return poi_flag=1,
               i.e. the email is somehow associated with poi
            .. otherwise return poi_flag=0
        proceed with original email text only
        (i.e. ignore text after words 'Original Message').
        then stem the text of the email using Snawballstemmer
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")

    ## get the sender's email address to from_field variable
    from_content = content[0].split("From: ")
    from_field = ""
    if len(from_content) > 1:
         from_field=from_content[1].split("\n")[0]


    ## check whether email address is from our database
    i=0
    person_from_our_db=0
    poi_flag=0
    email_addresses=[]
    while i<len(email_list) and person_from_our_db==0: 
        #print email_list[i], poi_indicator_list[i]
        if(from_field==email_list[i]):
            person_from_our_db=1
            email_addresses.append(email_list[i])
            if(poi_indicator_list[i]==1):
                poi_flag=1
        i+=1
            

    ## if from our database, then proceed with the email text   
    words = ""
    if len(content) > 1 & person_from_our_db==1:
        # drop everything after the words 'Original Message'
        email=content[1].split("Original Message")[0]
        
        ### remove punctuation
        text_string = email.translate(string.maketrans("", ""), string.punctuation)


        from nltk.stem import *
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer("english")

        words=""
        for word in text_string.split():
            words=words+stemmer.stem(word) + " "

    return words, poi_flag, email_addresses



def getFrom(f):
    """ open email file f
        get the 'from' email address
        return it
        """
    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()
    ### split off metadata
    content = all_text.split("From: ")
    from_field = ""
    if len(content) > 1:
         from_field=content[1].split("\n")[0]
    return from_field


def getAllFromAddresses(mailir_path,dir_list):
    #return a dict of all email addresses that appeared in "From:" at least once
    
    emails_dict={}
    cnt_all=0
    for directory0 in dir_list:
        cnt_dir=0
        
        for directory1 in os.listdir(os.path.join(mailir_path,directory0)):
            #print os.path.join(mailir_path,directory0,directory1)
            for root, dirs, files in os.walk(
                os.path.join(mailir_path,directory0,directory1), topdown=False):
                for name in files:
                    email = open(os.path.join(root, name), "r")
                    from_str= getFrom(email)
                    if from_str in emails_dict.keys():
                        emails_dict[from_str]+=1
                    else:
                        emails_dict[from_str]=1
                    cnt_dir+=1
                    cnt_all+=1
                    email.close()

                    
        print os.path.join(mailir_path,directory0), cnt_dir, 'emails'
    print 'total', cnt_all, "emails processed"
    return emails_dict


def process_emails2(mailir_path,dir_list,email_list,poi_indicator_list):
    #for directory0 in os.listdir("..\maildir"):
    #poi_data = []
    text_dict = {}
    #addr_data = []
    cnt_all=0
    cnt1_all=0
    cnt_poi_emails1_all=0

    
    """
    sw=["david","dave", "skill", "jeff", "mark","regard","steven","date","mark",
        "steven","2000","ken", "lay", "mr", "phillip", "louis", "2001", "mike",
        "jame","nonprivilegedpst", "vkaminsnsf", "skeannsf","friday","ddelainnsf",
        "tim","salli","wilson","kathi","sbecknsf","steve","bob","john","greg","paula",
        "jdasovicnsf", "delaineyhouect", "delainey","jshankmnsf","mmcconnnsf",
        "vinc","jlavoransf","beckhouectect"]
        """
    for directory0 in dir_list:
        cnt_dir=0
        cnt1_dir=0
        cnt_poi_emails1_dir=0
        
        for directory1 in os.listdir(os.path.join(mailir_path,directory0)):
            #print os.path.join(mailir_path,directory0,directory1)
            for root, dirs, files in os.walk(
                os.path.join(mailir_path,directory0,directory1), topdown=False):
                for name in files:
                    email = open(os.path.join(root, name), "r")
                    ##print os.path.join(root, name)
                    # text - email text
                    # poi_indicator = 1 if poi in 'from', 'cc' or 'bcc' 
                    text, poi_indicator, addresses = parseText(email, email_list, poi_indicator_list)
                                           
                        # for item in sw:
                        #     text=text.replace(item,"")

                    for address in addresses:
                        if address in text_dict.keys():
                            text_dict[address]+= " " + text
                        else:
                            text_dict[address]=text

                    if len(addresses)>0:
                        #addr_data.append(addresses)
                        #poi_data.append(poi_indicator)
                        cnt1_dir+=1
                        cnt1_all+=1
                        cnt_poi_emails1_dir+=poi_indicator
                        cnt_poi_emails1_all+=poi_indicator
                        
                    
                    cnt_dir+=1
                    cnt_all+=1
                    email.close()

                    
                    #print(os.path.join(root, name))
                    #text, poi_flag = parseText(ff,email_list, poi_indicator_list)
        print os.path.join(mailir_path,directory0), cnt_dir, 'total emails,',  \
              cnt1_dir, 'with addresses from our database, ', \
              "including", cnt_poi_emails1_dir, "emails indicated as poi"
    print 'total', cnt_all, "emails found,", cnt1_all, " with addresses from our database, ", \
          "including", cnt_poi_emails1_all, "emails indicated as poi"

    text_list=[]
    for i in range(0,len(email_list)):
        if email_list[i] in text_dict.keys():
            text_list.append(text_dict[email_list[i]])
        else:
            text_list.append("")

    pickle.dump( text_list, open("enron_text_list.pkl", "w") )
    pickle.dump( poi_indicator_list, open("enron_poi_list.pkl", "w") )



def process_emails(mailir_path,dir_list,email_list,poi_indicator_list):
    """
        Run parseText on all email files
        Then dump the results to pickle files
        (list of processed email texts and list of indicators whether email is from POI)
        """
    poi_data = []
    text_data = []
    addr_data = []
    # counters to count how many emails are processed
    cnt_all=0 #all emails
    cnt1_all=0 # all emails with senders from our database
    cnt_poi_emails1_all=0 # including those from POIs

    #list of stopwords: mostly names and transformed email addresses
    sw=["david","dave", "skill", "jeff", "mark","regard","steven","date","mark",
        "steven","2000","ken", "lay", "mr", "phillip", "louis", "2001", "mike",
        "jame","nonprivilegedpst", "vkaminsnsf", "skeannsf","friday","ddelainnsf",
        "tim","salli","wilson","kathi","sbecknsf","steve","bob","john","greg","paula",
        "jdasovicnsf", "delaineyhouect", "delainey","jshankmnsf","mmcconnnsf",
        "vinc","jlavoransf","beckhouectect",
        "mcconnellhouectect","keannaenronenron","kaminskihouect",
        "lavoratocorpenronenron","beckhouect",
        "kay","janet","frevertnaenronenron",
        "chris","christoph","edward","dan","shankmanhouectect","mhaedicnsf"]
    
    for directory0 in dir_list:
        # similar counter for every dir
        cnt_dir=0 
        cnt1_dir=0
        cnt_poi_emails1_dir=0
        
        for directory1 in os.listdir(os.path.join(mailir_path,directory0)):
            #print os.path.join(mailir_path,directory0,directory1)
            for root, dirs, files in os.walk(
                os.path.join(mailir_path,directory0,directory1), topdown=False):
                for name in files:
                    email = open(os.path.join(root, name), "r")
                    # text - email text
                    # poi_indicator = 1 if the sender is POI
                    text, poi_indicator, addresses = parseText(email, email_list, poi_indicator_list)

                   
                    if len(text)>1:
                        for item in sw:
                            text=text.replace(item,"")
                        text_data.append(text)
                        addr_data.append(addresses)
                        poi_data.append(poi_indicator)
                        cnt1_dir+=1
                        cnt1_all+=1
                        cnt_poi_emails1_dir+=poi_indicator
                        cnt_poi_emails1_all+=poi_indicator

                    cnt_dir+=1
                    cnt_all+=1
                    email.close()

        print os.path.join(mailir_path,directory0), cnt_dir, 'total emails,',  \
              cnt1_dir, 'with addresses from our database, ', \
              "including", cnt_poi_emails1_dir, "emails indicated as poi"
        
    print 'total', cnt_all, "emails found,", cnt1_all, " with addresses from our database, ", \
          "including", cnt_poi_emails1_all, "emails indicated as poi"


    pickle.dump( text_data, open("enron_word_data.pkl", "w") )
    pickle.dump( poi_data, open("enron_poi_indicator_data.pkl", "w") )
    pickle.dump( addr_data, open("enron_addresses_data.pkl", "w") )



def main():
    #################################
    #### test parseText function
    #################################
    ff = open("../maildir/sturm-f/calendar/3", "r")
    email_list=['dale.furrow@enron.com','1bill.abler@enron.com']
    poi_indicator_list=[0,1]
    text, poi_flag, address = parseText(ff,email_list, poi_indicator_list)
    print text, poi_flag, address
    ff.close()

    #############################################################################
    #####uncomment to get the list of all email addresses from the Enron database
    #####that initiated an email (e.g. all email addresses in 'from' field)
    """
    emails_addr_count=getAllFromAddresses("..\maildir",os.listdir("..\maildir"))
    
    #Save email addreses
      
    with open('emails.csv', 'w') as csvfile:
        fieldnames = ['email','cnt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,delimiter=',', lineterminator='\n')

        writer.writeheader()
        for item in emails_addr_count:
            entry={'email':item,
                   'cnt':emails_addr_count[item]}
            writer.writerow(entry)
        """
        

    ############################################################################
    ##### extract information from email documents
    ##### function process_emails stores the results into a pickle file
    # comment if the data is already extracted
    ############################################################################
    
    #"""
    ## load enron_data
    enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
    
    del enron_data['TOTAL'] ## remove outlier
    ##### uncomment to add 3 email addresses to our database
    ##enron_data=fix_emails(enron_data)

    ## get the list of email addresses
    email_list=[]
    poi_indicator_list=[]
    for person in enron_data:
        if enron_data[person]['email_address']!='NaN':
            email_list.append(enron_data[person]['email_address'])
            poi_indicator_list.append(int(enron_data[person]['poi']))
    print "Email addresses:", len(poi_indicator_list)
    print "    including those of POIs:", sum(poi_indicator_list)


    ## process all email documents
    t0 = time()
    process_emails("..\maildir",os.listdir("..\maildir"),email_list,poi_indicator_list)
    print 'Processing time:',round(time()-t0,1) ,'s'
    #    """
    #############################################################################



    #############################################################################
    #### TFIDF vectorizer
    #############################################################################
    #"""
    # load the output from the previous step:    
    t0 = time()
    text_data = pickle.load(open("../final_project/enron_word_data.pkl", "r"))
    poi_indicator_data = pickle.load(open("../final_project/enron_poi_indicator_data.pkl", "r"))
    print 'Loading time:',round(time()-t0,1) ,'s'
    t0 = time()
  
    features_train, features_test, labels_train, labels_test = \
                    cross_validation.train_test_split(\
                        text_data, poi_indicator_data, \
                        test_size=0.3, random_state=42)

    vectorizer =  TfidfVectorizer(stop_words='english',max_df=0.5,
                                  max_features=1500)
    features_train = vectorizer.fit_transform(features_train).toarray()
    print 'Vectorizer fit time:',round(time()-t0,1) ,'s'
    t0 = time()
    features_test  = vectorizer.transform(features_test).toarray()
    print 'Transform time:',round(time()-t0,1) ,'s'
    t0 = time()

   
    print len(vectorizer.get_feature_names()), 'total features extracted'

    #### find the best features for prediction
    from sklearn import tree
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    clf = tree.DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    print 'CLF fit time:',round(time()-t0,1) ,'s'
    t0 = time()
    pred=clf.predict(features_test)
    print 'Predict time:',round(time()-t0,1) ,'s'
    t0 = time()

    print "Accuracy:", accuracy_score(labels_test,pred)
    print "Recall:", recall_score(labels_test,pred)
    print "Precision:", precision_score(labels_test,pred)
    i=0
    threshold=numpy.percentile(clf.feature_importances_,85)
    sorted_list=[]
    for entry in clf.feature_importances_:
        sorted_list.append([i,vectorizer.get_feature_names()[i],entry])
        i+=1

    sorted_list=sorted(sorted_list, key=lambda tup: tup[2],reverse=True)

    
    for i in range(0,int(0.10*len(clf.feature_importances_))):
        print sorted_list[i]
    
        #"""
    ###########################################################################


    ###########################################################################
    ##### create a dataset with frequencies of particular words in emails
    ##### the output is dict of dicts, 1-levels keys are email addresses,
    ##### 2-level keys are the top predicting words (features)
    ##### and 'cnt_emails'=the number of emails sent
    ###########################################################################

    t0 = time()
    enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

    text_data = pickle.load(open("../final_project/enron_word_data.pkl", "r"))
    poi_indicator_data = pickle.load(open("../final_project/enron_poi_indicator_data.pkl", "r"))
    enron_addresses_data = pickle.load(open("../final_project/enron_addresses_data.pkl", "r"))

    print 'Loading time:',round(time()-t0,1) ,'s'
    t0 = time()

    text_features=['cnt_emails',
               'ena', 'ect', 'guy', 'rob', 'review', 'turbin', 'forward',
               'deal', 'thank', 'plan', 'toph', 'goal', 'val', 'asset',
               'pleas', 'want', 'asap', 'target', 'manag',
               'inform', 'ray', 'control', 'ee', 'look', 'america', 'hard',
               'given', 'view', 'ben', 'memo', 'onli', 'just', 'know',
               'howev', 'ani', 'east', 'pm', 'max', 'profit', 'meet',
               'need', 'yes', 'term', 'alloc', 'liz', 'note', 'use',
               'paul', 'fact', 'corp']


    #create dict with all 1-level and 2-level keys 
    stats_dict={}
    for person in enron_data:
        if enron_data[person]['email_address']!='NaN':
            stats_dict[enron_data[person]['email_address']]={}
            for k in range(0,len(text_features)):
                stats_dict[enron_data[person]['email_address']][text_features[k]]=0

        
    print "Emails:", len(stats_dict.keys())
    print 'Creating dict time:',round(time()-t0,1) ,'s'
    t0 = time()

    ## for particular email address
    ## count the number of sent emails
    ## and the number of emails with a particular word (i.e. feature)
    email_addresses=stats_dict.keys()
    for i in range(0,len(text_data)):
        #print enron_addresses_data[i]
        if (i+1) % 5000 ==0:
            print i+1
        for k in range(0,len(email_addresses)):
            #print  stats_dict[email_addresses[k]]['cnt_emails']
            if(email_addresses[k] in enron_addresses_data[i]):
               stats_dict[email_addresses[k]]['cnt_emails']+=1
               for feature in text_features:
                   if(text_data[i].find(feature)>0):
                       stats_dict[email_addresses[k]][feature]+=1
                    
    print 'Processing time:',round(time()-t0,1) ,'s'
    t0 = time()


    #write the output to csv file
    with open('text_features.csv', 'w') as csvfile:
        fieldnames = ['email_address']+stats_dict[stats_dict.keys()[0]].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,delimiter=',', lineterminator='\n')
        writer.writeheader()
        for item in stats_dict:
            entry=stats_dict[item]
            entry['email_address']=item
            writer.writerow(entry)



    
    

if __name__ == '__main__':
    main()

