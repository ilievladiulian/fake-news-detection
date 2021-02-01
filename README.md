# FakeNewsClassifier

### Setting up

Install MongoDB Community Server v4.4.3 and start it using the default host and port.

This project uses the following Python v3.7+ dependencies:    

- pytorch 0.4.1   
- torchtext 0.3.1     
- numpy 1.16.3   
- pymongo 3.7.2    


### Dataset

There is a tool in /dataset/save_dataset.py used to extract examples from /dataset/news_cleaned.csv and save them to MongoDB. To check this feature, you need to manually add the file news_cleaned.csv in the /dataset directory.

To save the dataset to mongo run:

```
python ./dataset/save_dataset.py
```

For testing the software, we also provide our full dataset in /data folder as an arhive. Please use monogimport to import it o MongoDB.

```
mongoimport --db=fake_news --collection=fake_news_corpus --type=json --file=dataset_clean.json
```

### Usage

To see the help for this program run

```
python main.py -h
```

To run the training on the logistic regression model:

    python main.py -m logreg

To run the training on the rcnn model:

    python main.py -m rcnn

To run the training on the rnn model:

    python main.py -m rnn

The program also accepts long-options: --help, --model=  
    
