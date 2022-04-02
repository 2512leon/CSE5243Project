from typing import OrderedDict
from bs4 import BeautifulSoup
import collections
from collections import Counter
import os

#function to merge dictionaries, add the counts for dictionaries that contain identical keys
def mergeDictionaries(dict1, dict2):
    mergedDict = Counter(dict1) + Counter(dict2)

placesDictionary = {}
topicsDictionary = {}

placesList = []
topicsList = []

for file in os.listdir("/data"):
    # For each .sgm file in the /data directory, read the contents
    if file.endswith(".sgm"):
        currentFile = os.path.join("data", file)
        input = open("currentFile", 'r')
        dataRead = input.read()

    # Setup the soup for the current file
    soup = BeautifulSoup(dataRead, 'html.parser')
    places = soup.findAll('places')
    topics = soup.findAll('topics')

    for places in soup.find_all('places'):
        if len(places.text) == 0:
            # Handle the empty places tags
            placesList.append('{}')
        for place in places.find_all('d'):
            placesList.append(place.text)

    for topics in soup.find_all('topics'):
        if len(topics.text) == 0:
            # Handle the empty topics tags
            topicsList.append('{}')
        for topic in topics.find_all('d'):
            topicsList.append(topic.text)

    # list comprehension to convert lists into dictionaries with counts
    currPlacesDictionary = {place:placesList.count(place) for place in placesList} # yay list comprehension
    currTopicsDictionary = {topic:topicsList.count(topic) for topic in topicsList}

    placesDictionary = mergeDictionaries(placesDictionary, currPlacesDictionary)
    topicsDictionary = mergeDictionaries(placesDictionary, currTopicsDictionary)

# for sorting the dictionaries and retaining some form of order since python dicts are unordered
orderedPlaces = OrderedDict(sorted(placesDictionary.items()))
orderedTopics = OrderedDict(sorted(topicsDictionary.items()))

placesDictFinal = OrderedDict()
topicsDictFinal = OrderedDict() 

sequenceNumber = 1
for k,v in orderedPlaces.items():
    placesDictFinal[sequenceNumber] = [k,v]
    sequenceNumber = sequenceNumber + 1

sequenceNumber = 1
for k,v in orderedTopics.items():
    topicsDictFinal[sequenceNumber] = [k,v]
    sequenceNumber = sequenceNumber + 1

print("Places Dictionary:")
for k,v in placesDictFinal.items():
    print("Sequence Number:", k)
    print("Place:",v[0])
    print("Frequency:",v[1])

print("\n\nTopics Dictionary: ")
for k,v in topicsDictFinal.items():
    print("Sequence Number:", k)
    print("Topic:",v[0])
    print("Frequency:",v[1])
