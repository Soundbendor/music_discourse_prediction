from pymongo import MongoClient

client = MongoClient()

db = client['test']
songs = db["songs"]

#  songs.insert_one({
    #  "artist_name": "Madeon",
    #  "track_title": "Pop Culture"
    #  })
#  print(db)

item_details = songs.find()
for item in item_details:
    print(item)


