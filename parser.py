from graphlib import TopologicalSorter
from random import Random
from tempfile import tempdir
from typing import Dict, OrderedDict
from bs4 import BeautifulSoup
from collections import Counter
import os
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as metrics

stemmer = WordNetLemmatizer()
places_vectorizer = CountVectorizer(min_df=0.01, max_df=0.1, stop_words=stopwords.words('english'))
topics_vectorizer = CountVectorizer(min_df=0.01, max_df=0.45, stop_words=stopwords.words('english'))



placesDictionary = {}
topicsDictionary = {}
allReuters = []
for file in os.listdir("./data"):
    # For each .sgm file in the /data directory, read the contents
    currentFile = os.path.join("./data/", file)
    input = open(currentFile, 'r')
    dataRead = input.read()

    # Setup the soup for the current file
    soup = BeautifulSoup(dataRead, 'html.parser')
    places = soup.findAll('places')
    topics = soup.findAll('topics')
    reuters = soup.findAll('reuters')

    # initialize lists, not necessary but good practice
    placesList = []
    topicsList = []
    reutersList = []

    for reuters in soup.find_all('reuters'):
        tempPlaceList = []
        tempTopicsList = []
        for places in reuters.find_all('places'):
            for place in places.find_all('d'):
                tempPlaceList.append(place.text)
        for topics in reuters.find_all('topics'):
            for topic in topics.find_all('d'):
                tempTopicsList.append(topic.text)
        for reuter in reuters.find_all('body'):
            body = reuter.text
        reutersList = [tempPlaceList, tempTopicsList, body]
        allReuters.append(reutersList)
        # [[topics], [places], body]



    for places in soup.find_all('places'): # find all the <places> tags
        if len(places.text) == 0:
            # Handle the empty places tags
            placesList.append('{}')
        for place in places.find_all('d'): # find all the <d> tags within <places>
            placesList.append(place.text) # get the text and append it to our list

    for topics in soup.find_all('topics'): # find all <topics> tags
        if len(topics.text) == 0:
            # Handle the empty topics tags
            topicsList.append('{}')
        for topic in topics.find_all('d'): # find all the <d> tags within <topics>
            topicsList.append(topic.text) # get the text and append it to our list

    # List comprehension to convert lists into dictionaries with counts
    currPlacesDictionary = {place:placesList.count(place) for place in placesList}
    currTopicsDictionary = {topic:topicsList.count(topic) for topic in topicsList}

    # combine temporary dictionary (currPlacesDictionary or currTopicsDictionary)
    # with the placesDictionary or topicsDictionary that will be saved after going to the next file
    placesDictionary = Counter(placesDictionary) + Counter(currPlacesDictionary)
    topicsDictionary = Counter(topicsDictionary) + Counter(currTopicsDictionary)

# For sorting the dictionaries and retaining some form of order since python dicts are unordered
orderedPlaces = OrderedDict(sorted(placesDictionary.items()))
orderedTopics = OrderedDict(sorted(topicsDictionary.items()))

# initialize final dictionaries that will store as a <sequence_number, [place/topic, frequency]> dictionary
placesDictFinal = OrderedDict()
topicsDictFinal = OrderedDict()

# set some sequenceNumber var to use as the key in the final dictionary
sequenceNumber = 1
for k,v in orderedPlaces.items():
    placesDictFinal[sequenceNumber] = [k,v] # the value of the dictionary will be the [k,v] pair from the original dictionary
    sequenceNumber = sequenceNumber + 1

# set some sequenceNumber var to use as the key in the final dictionary
sequenceNumber = 1
for k,v in orderedTopics.items():
    topicsDictFinal[sequenceNumber] = [k,v] # the value of the dictionary will be the [k,v] pair from the original dictionary
    sequenceNumber = sequenceNumber + 1

# print things out. We can then pipe this into a txt file for easy printing and submission of part 1
# print("Places Dictionary:")
# for k,v in placesDictFinal.items():
#     print("Sequence Number:", k, "| Place:", v[0], "| Frequency:", v[1])

# print("\n\nTopics Dictionary: ")
# for k,v in topicsDictFinal.items():
#     print("Sequence Number:", k, "| Topic:", v[0], "| Frequency:", v[1])

# [[places], [topics], [body]]
# places = ['usa', 'uganda', 'poland'] --> [1,2,3] 
# places = []
# places = ['usa']



placesTruthTemp = []
topicsTruthTemp = []
bodyDataForPlaces = []
bodyDataForTopics = []
for reuter in allReuters:
    if len(reuter[0]) == 1: # only grab if there is 1 truth value, if [] then ignore
        placesTruthTemp.append(reuter[0])
        bodyDataForPlaces.append(reuter[2])
    if len(reuter[1]) == 1: # only grab if there is 1 truth value, if [] then ignore
        topicsTruthTemp.append(reuter[1])
        bodyDataForTopics.append(reuter[2])

words = []
# places feature vector pre-processing
for sen in range(0, len(bodyDataForPlaces)):
    # Remove all the special characters
    word = re.sub(r'\W', ' ', str(bodyDataForPlaces[sen]))

    # Remove numbers and digits
    word = re.sub('W*dw*',' ',word)

    # Substituting multiple spaces with single space
    word = re.sub(r'\s+', ' ', word, flags=re.I)

    # Converting to Lowercase
    word = word.lower()

    # Lemmatization
    word = word.split()

    word = [stemmer.lemmatize(x) for x in word]
    word = ' '.join(word)

    words.append(word)

# topics feature vector pre-processing
wordsForTopics = []
for sen in range(0, len(bodyDataForTopics)):
    
    # Remove all the special characters
    # word = re.sub(r'\W', ' ', str(bodyDataForTopics[sen]))

    # Substituting multiple spaces with single space
    word = re.sub(r'\s+', ' ', str(bodyDataForTopics[sen]), flags=re.I)

    # Lemmatization
    word = word.split()

    word = [stemmer.lemmatize(x) for x in word]
    word = ' '.join(word)

    wordsForTopics.append(word)


i = 1
placesTruthDict = dict()
# creation of dictionary to assign numbers to places
for place in placesDictionary.keys():
    if place == "{}":
        placesTruthDict["{}"] = 0
    else:
        placesTruthDict[place] = i
        i += 1

i = 1
topicsTruthDict = dict()
# creation of dictionary to assign numbers to topics
for topic in topicsDictionary.keys():
    if topic == "{}":
        topicsTruthDict["{}"] = 0
    else:
        topicsTruthDict[topic] = i
        i += 1
#print(topicsTruthDict)

placesTruth = []
topicsTruth = []

# create a truth vector for places based on numbers from dictionaries created above
for entry in placesTruthTemp:
    # ['usa', 'uganda', 'poland']
    temp = 0
    for place in entry:
        if temp == 0: # this will only trigger before the first entry
            temp = placesTruthDict.get(place)
    placesTruth.append(temp)

#create a truth vector for topics based on numbers from dictionaries created above
for entry in topicsTruthTemp:
    # ['usa', 'uganda', 'poland']
    temp = 0
    for topic in entry:
        if temp == 0: # only triggers before first entry
            temp = topicsTruthDict.get(topic)
    topicsTruth.append(temp)
#print(topicsTruth)

print("TRUTH DICTIONARIES")
for k,v in placesTruthDict.items():
    print("Place: " + k + " Number: " + str(v))
for k,v in topicsTruthDict.items():
    print("Topic: " + k + " Number: " + str(v))
# classification time
X = places_vectorizer.fit_transform(words).toarray()
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()
print(len(X[0]))
X_train, X_test, y_train, y_test = train_test_split(X, placesTruth, test_size=0.2, random_state=0)


X_topics = topics_vectorizer.fit_transform(wordsForTopics).toarray()
tfidfconverter_topics = TfidfTransformer()
X_topics = tfidfconverter_topics.fit_transform(X_topics).toarray()
print(len(X_topics[0]))
X_train_topics, X_test_topics, y_train_topics, y_test_topics = train_test_split(X_topics, topicsTruth, test_size=0.2, random_state=0)

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

topics_classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
topics_classifier.fit(X_train_topics, y_train_topics)

y_pred = classifier.predict(X_test)
y_pred_topics = topics_classifier.predict(X_test_topics)

print("PLACES CLASSIFICATION:")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

print("TOPICS CLASSIFICATION:")
print(confusion_matrix(y_test_topics,y_pred_topics))
print(classification_report(y_test_topics,y_pred_topics))
print(accuracy_score(y_test_topics, y_pred_topics))



