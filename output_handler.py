class OutputHandler():
    def __init__(self, filename):
        self.fileDescriptor = open(outputFile, "a+")

    def write(self, message):
        self.fileDescriptor.write(message)

    def close(self):
        self.fileDescriptor.close()
