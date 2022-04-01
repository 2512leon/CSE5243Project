from typing import OrderedDict
from bs4 import BeautifulSoup
import collections

fileNames = []
topics = []
places = []

placesList = []
topicsList = []
# for file in os.listdir("/data"):
# if file.endswith(".sgm"):
# filename = os.path.join("data", file)
f = open("reut2-000.sgm", 'r')
dataRead = f.read()

soup = BeautifulSoup(dataRead, 'html.parser')
places = soup.findAll('places')
topics = soup.findAll('topics')

for places in soup.find_all('places'):
    if len(places.text) == 0:
        placesList.append('{}')
    for place in places.find_all('d'):
        placesList.append(place.text)

for topics in soup.find_all('topics'):
    if len(topics.text) == 0:
        topicsList.append('{}')
    for topic in topics.find_all('d'):
        topicsList.append(topic.text)

# list comprehension to convert lists into dictionaries with counts
placesDictionary = {place:placesList.count(place) for place in placesList} # yay list comprehension
topicsDictionary = {topic:topicsList.count(topic) for topic in topicsList}

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
