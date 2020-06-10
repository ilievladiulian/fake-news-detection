from database_connection import collection
import csv
import random

LABEL = 'label'
CONTENT = 'content'

csv_columns = [LABEL, CONTENT]

train_file_name = 'train.csv'
test_file_name = 'test.csv'
labels_file_name = 'labels.txt'

labels = collection.distinct('label')
labels_map = {}
no_examples = {}

for i in range(len(labels)):
    labels_map[labels[i]] = str(i+1)
    no_examples[labels[i]] = 0

with open(labels_file_name, 'w') as labels_file:
    for label in labels:
        labels_file.write('%s\n' % label)

train_examples = []
test_examples = []

for document in collection.find():
    current_label = document[LABEL]
    current_doc = { LABEL: labels_map[current_label], CONTENT: document[CONTENT].replace('\n', '').replace('\t', '').replace('\'', '').replace('\"', '') }
    no_examples[current_label] += 1
    if no_examples[current_label] <= 7000:
        train_examples.append(current_doc)
    else:
        test_examples.append(current_doc)

random.shuffle(train_examples)
random.shuffle(test_examples)

with open(train_file_name, 'w') as train_file:
    for example in train_examples:
        train_file.write('\"%s\",\"%s\"\n' % (example[LABEL], example[CONTENT]))

with open(test_file_name, 'w') as test_file:
    for example in test_examples:
        test_file.write('\"%s\",\"%s\"\n' % (example[LABEL], example[CONTENT]))
