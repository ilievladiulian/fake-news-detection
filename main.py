import sys, getopt
from logistic_regression import LogisticRegression
from recurrent_cnn import RecurrentConvolutionalNN
from rnn import RecurrentNN


def main(argv):
    modelName = ''
    modelPossibilities = {
        'logreg': LogisticRegression,
        'rcnn': RecurrentConvolutionalNN,
        'rnn': RecurrentNN
    }

    try:
        opts, args = getopt.getopt(argv, 'hm:', ['help' ,'model='])
    except getopt.GetoptError:
        print('usage: main.py -m <modelname> or main.py --model=<modelname>, where <modelname>: rnn, rcnn or logreg')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: main.py -m <modelname> or main.py --model=<modelname>, where <modelname>: rnn, rcnn or logreg')
            sys.exit()
        elif opt in ('-m', '--model'):
            modelName = arg
    
    modelHandlerName = modelPossibilities.get(modelName, 'Invalid model')
    if modelHandlerName == 'Invalid model':
        print('Invalid model name. Type python main.py -h for help')
        sys.exit()

    modelHandler = modelHandlerName()
    modelHandler.train()

    test_loss, test_acc = modelHandler.test()
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

    

if __name__ == '__main__':
    main(sys.argv[1:])