import dataset.load_dataset_generic as generic_dataset
import dataset.load_dataset_linear as linear_dataset

def load(datasetType='generic'):
    if datasetType == 'linear':
        return linear_dataset.load()
    else:
        return generic_dataset.load()