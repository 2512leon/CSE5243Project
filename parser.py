from tempfile import tempdir
from typing import OrderedDict
from bs4 import BeautifulSoup
from collections import Counter
import os

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
print("Places Dictionary:")
for k,v in placesDictFinal.items():
    print("Sequence Number:", k, "| Place:", v[0], "| Frequency:", v[1])

print("\n\nTopics Dictionary: ")
for k,v in topicsDictFinal.items():
    print("Sequence Number:", k, "| Topic:", v[0], "| Frequency:", v[1])

# print("Reuters Count: ", len(allReuters))
# firstReuter = allReuters[140]

# firstPlaces = firstReuter[0]
# firstTopics = firstReuter[1]
# firstBody = firstReuter[2]
# print("Topics: ", firstTopics)
# print("Places: ", firstPlaces)
# print("Body: ", firstBody)
# featuresVec = []

# for i in range(len(allReuters)):
#     reuter = allReuters[i]
#     places = reuter[0]
#     topics = reuter[1]
#     body = reuter[2]
#     # TODO add body to features