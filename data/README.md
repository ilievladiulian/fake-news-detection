### Archive of used dataset

Merge zip parts to singe archive:

`$ zip -s 0 dataset_clean.json.zip --out dataset_clean.zip`

Unzip resulting archive:

`$ unzip dataset_clean.zip`

Import the resulting json file using mongoimport:

`$ mongoimport --db fake_news --collection fake_news_corpus --file fake_news_dataset_clean.json`
