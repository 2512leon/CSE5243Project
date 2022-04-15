from graphlib import TopologicalSorter
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

stemmer = WordNetLemmatizer()
vectorizer = CountVectorizer(max_features=500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))


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


placesTruthTemp = []
topicsTruthTemp = []
bodyData = []
for reuter in allReuters:
    placesTruthTemp.append(reuter[0])
    topicsTruthTemp.append(reuter[1])
    bodyData.append(reuter[2])

words = []
for sen in range(0, len(bodyData)):
    # Remove all the special characters
    word = re.sub(r'\W', ' ', str(bodyData[sen]))
    # remove all single characters
    word = re.sub(r'\s+[a-zA-Z]\s+', ' ', word)

    # Remove single characters from the start
    word = re.sub(r'\^[a-zA-Z]\s+', ' ', word)

    # Substituting multiple spaces with single space
    word = re.sub(r'\s+', ' ', word, flags=re.I)

    # Removing prefixed 'b'
    word = re.sub(r'^b\s+', '', word)

    # Converting to Lowercase
    word = word.lower()

    # Lemmatization
    word = word.split()

    word = [stemmer.lemmatize(x) for x in word]
    word = ' '.join(word)

    words.append(word)

i = 1
placesTruthDict = dict()
for place in placesDictionary.keys():
    if place == "{}":
        placesTruthDict["{}"] = 0
    else:
        placesTruthDict[place] = i
        i += 1

i = 1
topicsTruthDict = dict()
for topic in topicsDictionary.keys():
    if topic == "{}":
        topicsTruthDict["{}"] = 0
    else:
        topicsTruthDict[topic] = i
        i += 1
#print(topicsTruthDict)

placesTruth = []
topicsTruth = []

for entry in placesTruthTemp:
    # ['usa', 'uganda', 'poland']
    temp = 0
    for place in entry:
        temp = placesTruthDict.get(place)
    placesTruth.append(temp)

for entry in topicsTruthTemp:
    # ['usa', 'uganda', 'poland']
    temp = []
    for topic in entry:
        temp.append(topicsTruthDict.get(topic))
    if len(temp) == 0:
        temp.append(0)
    topicsTruth.append(temp)
#print(topicsTruth)






X = vectorizer.fit_transform(words).toarray()
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()
#print(len(X))
#print(len(placesTruth))
X_train, X_test, y_train, y_test = train_test_split(X, placesTruth, test_size=0.2, random_state=0)

# print(X_train[1])
# print(y_train[1])

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))




