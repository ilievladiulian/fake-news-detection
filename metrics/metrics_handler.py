metricsHandler = None

class MetricsHandler():
    def __init__(self):
        self.labels = {}
        self.trueLabels = {}

    def update(self, predictedLabel, trueLabel):
        if not predictedLabel in self.labels:
            self.labels[predictedLabel] = {}
            self.labels[predictedLabel]["positive"] = 0
            self.labels[predictedLabel]["negative"] = 0

        if not trueLabel in self.trueLabels:
            self.trueLabels[trueLabel] = {}
            self.trueLabels[trueLabel]["positive"] = 0
            self.trueLabels[trueLabel]["negative"] = 0
            
        if predictedLabel == trueLabel:
            self.labels[predictedLabel]["positive"] += 1
            self.trueLabels[trueLabel]["positive"] += 1
        else:
            self.labels[predictedLabel]["negative"] += 1
            self.trueLabels[trueLabel]["negative"] += 1

    def reset(self):
        for k, v in self.labels.items():
            self.labels[k]["positive"] = 0
            self.labels[k]["negative"] = 0

        for k, v in self.trueLabels.items():
            self.trueLabels[k]["positive"] = 0
            self.trueLabels[k]["negative"] = 0

    def getMetrics(self):
        return self.labels

    def getRecall(self):
        recalls = {}
        for k, v in self.trueLabels.items():
            if k in self.labels:
                truePositives = self.labels[k]["positive"]
                falsePositives = self.labels[k]["negative"]
                falseNegatives = self.trueLabels[k]["negative"]

                recalls[k] = float(float(truePositives) / (float(truePositives) + float(falseNegatives)))
            else:
                recalls[k] = 0

        avgRecall = 0
        noItems = 0
        for k, v in recalls.items():
            avgRecall += v
            noItems += 1

        return float(float(avgRecall) * 100 / float(noItems))

    def getPrecision(self):
        precisions = {}
        for k, v in self.trueLabels.items():
            if k in self.labels:
                truePositives = self.labels[k]["positive"]
                falsePositives = self.labels[k]["negative"]
                falseNegatives = self.trueLabels[k]["negative"]

                precisions[k] = float(float(truePositives) / (float(truePositives) + float(falsePositives)))
            else:
                precisions[k] = 0

        avgPrecision = 0
        noItems = 0
        for k, v in precisions.items():
            avgPrecision += v
            noItems += 1

        return float(float(avgPrecision) * 100 / float(noItems))

