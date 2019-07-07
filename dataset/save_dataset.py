import csv
from database_connection import collection
from news_model import NewsObject

MAX_LINE_COUNT = 10000
data_tags = {
    'fake': 0, 
    'satire': 0, 
    'bias': 0, 
    'conspiracy': 0, 
    'junksci': 0, 
    'hate': 0, 
    'clickbait': 0, 
    'unreliable': 0, 
    'political': 0, 
    'reliable': 0
}
dataset_file = './dataset/news_cleaned.csv'

with open(dataset_file, mode='r', encoding='UTF-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count = line_count + 1
            continue
        else:
            isFull = True
            for key, value in data_tags.items():
                if value < MAX_LINE_COUNT:
                    isFull = False
            if isFull:
                break

            tag = row['type']
            try:
                mock = data_tags[tag]
            except:
                continue

            if data_tags[tag] >= MAX_LINE_COUNT:
                continue
            else:
                print("insert: " + tag)
                data_tags[tag] = data_tags[tag] + 1
                document = NewsObject(row['content'], tag)
                document_id = collection.insert_one(document.__dict__)
            
    print(f'Processed {line_count} lines.')