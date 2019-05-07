import csv
from database_connection import collection
from news_model import NewsObject

MAX_LINE_COUNT = 20000
dataset_file = 'news_cleaned.csv'

with open(dataset_file, mode='r', encoding='UTF-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            if line_count > MAX_LINE_COUNT:
                break
            
            document = NewsObject(row['content'], row['type'])
            document_id = collection.insert_one(document.__dict__)
            line_count += 1
    print(f'Processed {line_count} lines.')