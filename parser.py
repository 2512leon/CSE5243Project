from bs4 import BeautifulSoup

fileNames = []
topics = []
places = []

placesList=[]
topicsList=[]

# for file in os.listdir("/data"):
# if file.endswith(".sgm"):
# filename = os.path.join("data", file)
f = open("reut2-000.sgm", 'r')
dataRead = f.read()

soup = BeautifulSoup(dataRead, 'html.parser')
places = soup.findAll('places')
topics = soup.findAll('topics')

for place in places:
    placesList = soup.findAll('d')

for topic in topics:
    topicsList = soup.findAll('d')


print(placesList)
print("Topics:")
print(topicsList)



