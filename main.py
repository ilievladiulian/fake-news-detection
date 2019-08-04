import sys, getopt
from logistic_regression import LogisticRegression
from recurrent_cnn import RecurrentConvolutionalNN
from rnn import RecurrentNN
from cnn import ConvolutionalNN
from lstm import LongShortTermMemory
from output_handler import OutputHandler
import torch

outputFileHandler = None

def main(argv):
    modelName = ''
    modelPossibilities = {
        'logreg': LogisticRegression,
        'rcnn': RecurrentConvolutionalNN,
        'rnn': RecurrentNN,
        'cnn': ConvolutionalNN,
        'lstm': LongShortTermMemory
    }
    outputFile = None
    classifierType = None
    classifierTypePossibilities = {
        'longer': 'longer',
        'repeater': 'repeater'
    }

    try:
        opts, args = getopt.getopt(argv, 'hmot:', ['help', 'model=', 'output=', 'type='])
    except getopt.GetoptError:
        print('usage: main.py -m <modelname> or main.py --model=<modelname>, where <modelname>: rnn, lstm, cnn, rcnn or logreg')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: main.py -m <modelname> or main.py --model=<modelname>, where <modelname>: rnn, lstm, cnn, rcnn or logreg')
            sys.exit()
        elif opt in ('-m', '--model'):
            modelName = arg
        elif opt in ('-o', '--output'):
            outputFile = arg
        elif opt in ('-t', '--type'):
            classifierType = arg
    
    modelHandlerName = modelPossibilities.get(modelName, 'Invalid model')
    if modelHandlerName == 'Invalid model':
        print('Invalid model name. Type python main.py -h for help')
        sys.exit()
    
    outputFileHandler = OutputHandler(outputFile)
    outputFileHandler.write("Start log \n")

    numberOfEpochs = 10

    if classifierType == classifierTypePossibilities['longer']:
        numberOfEpochs = 20
        modelHandler = modelHandlerName()
        modelHandler.train(numberOfEpochs)
        test_loss, test_acc = modelHandler.test()
        print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
        outputFileHandler.write(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%\n')
    else:
        results = []
        modelHandler = None
        torch.cuda.empty_cache()
        for i in range(numberOfEpochs):
            modelHandler = modelHandlerName()
            modelHandler.train(numberOfEpochs)
            test_loss, test_acc = modelHandler.test()
            print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
            outputFileHandler.write(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%\n')

    outputFileHandler.close()

if __name__ == '__main__':
    main(sys.argv[1:])