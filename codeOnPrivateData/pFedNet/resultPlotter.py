import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#mpl.rcParams['text.usetex'] = True

class FileChecker(object):
    def __init__(self, givenFilePath):
        self.__filePath = givenFilePath
        self.__init()
        pass

    def __init(self):
        if os.path.exists(self.__filePath) == False:
            dir = os.path.dirname(self.__filePath)
            if os.path.exists(dir) == False:
                os.makedirs(dir)
            with open(self.__filePath, mode='a', encoding='utf-8') as f:
                print('Successfully creat {%s}' % self.__filePath)
        pass 

    def get(self):
        return self.__filePath   
    pass

class AccWrtLambdaPlotterForLuna(object):
    def __init__(self, givenAux):
        self.__aux = givenAux
        self.__lambdaValList = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
        self.__accMeanList = [
            [77.50, 77.33, 77.42, 77.92, 77.17, 78.33, 77.50, 78.17], # delta = 1
            [79.68, 79.51, 79.93, 80.95, 79.59, 79.59, 78.40, 78.91], # delta = 2
            [79.97, 80.05, 80.56, 83.16, 70.79, 69.78, 69.61, 69.36], # delta = 4
            [83.75, 85.08, 83.33, 84.17, 60.42, 65.92, 64.17, 65.42] # delta = 7
        ]
        self.__accStdList = [
            [0.49, 0.38, 0.52, 1.13, 0.63, 0.14, 0.75, 0.52],
            [0.39, 0.53, 0.39, 0.53, 0.44, 0.00, 0.39, 0.15],
            [0.89, 1.57, 0.44, 0.77, 0.96, 0.64, 0.53, 0.15],
            [0.00, 0.14, 0.29, 0.38, 1.66, 0.52, 0.72, 0.14]
        ]
        self.__savedDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', 'accWrtLambda')
        pass

    def execute(self):
        xPos, xShift = list(range(len(self.__lambdaValList))), [-0.3, -0.1, 0.1, 0.3]
        for i in range(len(self.__accMeanList)):
            self.__aux.bar([xPos[j]+xShift[i] for j in range(len(xPos))], self.__accMeanList[i], width = 0.2, yerr = self.__accStdList[i], alpha = 0.5)
            yMax = max(zip(self.__accMeanList[i], self.__accStdList[i]))
            yMin = min(zip(self.__accMeanList[i], self.__accStdList[i]))
            self.__aux.set_ylim([yMin[0]*0.95 - yMin[1], yMax[0]*1.01+yMax[1]])
            self.__aux.set_ylabel('Accuracy')
            self.__aux.set_xlabel('$\lambda$')
            self.__aux.set_xticks(xPos, [str(lambdav) for lambdav in self.__lambdaValList])
        self.__aux.legend(['$\delta=1$', '$\delta=2$', '$\delta=4$', '$\delta=7$'])
        targetFilePath = FileChecker(givenFilePath=os.path.join(self.__savedDir, 'luna.pdf')).get()
        plt.savefig(targetFilePath) 
        plt.show()
        return self.__aux
    pass

class CommunicationEfficiencyWrtGammaPlotterForLuna(object):
    def __init__(self, givenAux):
        self.__aux = givenAux
        self.__gammaValList = [0, 0.1, 1, 10]
        self.__ceMeanList = [
            [34.91, 63.69, 74.91, 82.24], # at server
            [147.84, 169.91, 191.54, 224.96]
        ]
        self.__ceStdList = [
            [0.44, 2.07, 2.66, 2.34], # at server
            [1.59, 5.05, 7.48, 5.36]
        ]
        self.__savedDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', 'ceWrtGamma')
        pass

    def execute(self):
        xPos, xShift = list(range(len(self.__gammaValList))), [-0.1, 0.1]
        for i in range(len(self.__ceMeanList)):
            self.__aux.bar([xPos[j]+xShift[i] for j in range(len(xPos))], self.__ceMeanList[i], width = 0.2, yerr = self.__ceStdList[i], alpha = 0.5)
            yMax = max(zip(self.__ceMeanList[i], self.__ceStdList[i]))
            yMin = min(zip(self.__ceMeanList[i], self.__ceStdList[i]))
            #self.__aux.set_ylim([yMin[0]*0.95 - yMin[1], yMax[0]*1.01+yMax[1]])
            self.__aux.set_ylabel('Compression ratio')
            self.__aux.set_xlabel('$\gamma$')
            self.__aux.set_xticks(xPos, [str(gamma) for gamma in self.__gammaValList])
        self.__aux.legend(['at server (downstream)', 'at client (upstream)'])
        targetFilePath = FileChecker(givenFilePath=os.path.join(self.__savedDir, 'luna.pdf')).get()
        plt.savefig(targetFilePath) 
        plt.show()
        return self.__aux
    pass

class AccWrtLambdaPlotterForBraTS2017(object):
    def __init__(self, givenAux):
        self.__aux = givenAux
        self.__lambdaValList = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
        self.__accMeanList = [
            [70.39, 69.92, 70.32, 71.21, 73.47, 74.13, 73.61, 73.34], # delta = 1
            [66.80, 67.53, 67.44, 68.11, 70.80, 70.61, 71.04, 71.67], # delta = 2
            [70.42, 70.17, 70.21, 70.34, 72.18, 72.44, 71.84, 71.84], # delta = 4
            [69.62, 69.91, 69.98, 70.00, 70.58, 70.26, 70.78, 70.29] # delta = 7
        ]
        self.__accStdList = [
            [0.08, 0.17, 0.09, 0.30, 0.19, 0.21, 0.13, 0.66],
            [0.43, 0.48, 0.32, 1.12, 0.10, 0.48, 0.43, 0.09],
            [0.25, 0.06, 0.66, 0.56, 0.40, 0.02, 0.43, 0.24],
            [0.75, 0.37, 0.32, 0.20, 0.08, 0.50, 0.65, 0.60]
        ]
        self.__savedDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', 'accWrtLambda')
        pass

    def execute(self):
        xPos, xShift = list(range(len(self.__lambdaValList))), [-0.3, -0.1, 0.1, 0.3]
        for i in range(len(self.__accMeanList)):
            self.__aux.bar([xPos[j]+xShift[i] for j in range(len(xPos))], self.__accMeanList[i], width = 0.2, yerr = self.__accStdList[i], alpha = 0.5)
            yMax = max(zip(self.__accMeanList[i], self.__accStdList[i]))
            yMin = min(zip(self.__accMeanList[i], self.__accStdList[i]))
            self.__aux.set_ylim([yMin[0]*0.95 - yMin[1], yMax[0]*1.05+yMax[1]])
            self.__aux.set_ylabel('IoU')
            self.__aux.set_xlabel('$\lambda$')
            self.__aux.set_xticks(xPos, [str(lambdav) for lambdav in self.__lambdaValList])
        self.__aux.legend(['lack #0', 'lack #1', 'lack #2', 'lack #3'])
        targetFilePath = FileChecker(givenFilePath=os.path.join(self.__savedDir, 'braTS2017.pdf')).get()
        plt.savefig(targetFilePath) 
        plt.show()
        return self.__aux
    pass

class CommunicationEfficiencyWrtGammaPlotterForBraTS2017(object):
    def __init__(self, givenAux):
        self.__aux = givenAux
        self.__gammaValList = [50, 100, 500, 1000]
        self.__ceMeanList = [
            [7.53, 9.84, 11.45, 17.26], # at server
            [28.13, 52.50, 63.14, 92.40]
        ]
        self.__ceStdList = [
            [2.2, 2.75, 3.16, 4.35], # at server
            [10.76, 13.11, 14.57, 17.33]
        ]
        self.__savedDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', 'ceWrtGamma')
        pass

    def execute(self):
        xPos, xShift = list(range(len(self.__gammaValList))), [-0.1, 0.1]
        for i in range(len(self.__ceMeanList)):
            self.__aux.bar([xPos[j]+xShift[i] for j in range(len(xPos))], self.__ceMeanList[i], width = 0.2, yerr = self.__ceStdList[i], alpha = 0.5)
            yMax = max(zip(self.__ceMeanList[i], self.__ceStdList[i]))
            yMin = min(zip(self.__ceMeanList[i], self.__ceStdList[i]))
            self.__aux.set_ylabel('Compression ratio')
            self.__aux.set_xlabel('$\gamma$')
            self.__aux.set_xticks(xPos, [str(gamma) for gamma in self.__gammaValList])
        self.__aux.legend(['at server (downstream)', 'at client (upstream)'], loc=2)
        targetFilePath = FileChecker(givenFilePath=os.path.join(self.__savedDir, 'brats2017.pdf')).get()
        plt.savefig(targetFilePath) 
        plt.show()
        return self.__aux
    pass

class AccWrtLambdaPlotterForLngxbhbcdexzlhzcxfxyc(object):
    def __init__(self, givenAux):
        self.__aux = givenAux
        self.__lambdaValList = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
        self.__accMeanList = [
            [79.52, 79.52, 81.67, 76.43, 92.86, 92.86, 92.86, 92.86], # delta = 1
            [69.29, 67.62, 69.05, 69.76, 69.52, 69.76, 69.76, 69.76], # delta = 2
            [69.95, 70.42, 71.36, 69.72, 69.72, 69.25, 69.95, 69.25], # delta = 3
            [68.79, 67.38, 68.79, 69.50, 69.26, 69.74, 70.45, 68.56]  # delta = 4
        ]
        self.__accStdList = [
            [11.57, 11.57, 17.56, 28.46, 0.00, 0.00, 0.00, 0.00],
            [2.86, 3.52, 1.80, 0.41, 0.83, 0.41, 0.41, 1.09],
            [1.77, 1.22, 2.85, 0.00, 0.00, 0.81, 0.40, 0.81],
            [1.23, 0.72, 1.23, 0.00, 0.41, 0.41, 1.64, 1.63]
        ]
        self.__savedDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', 'accWrtLambda')
        pass

    def execute(self):
        xPos, xShift = list(range(len(self.__lambdaValList))), [-0.3, -0.1, 0.1, 0.3]
        for i in range(len(self.__accMeanList)):
            self.__aux.bar([xPos[j]+xShift[i] for j in range(len(xPos))], self.__accMeanList[i], width = 0.2, yerr = self.__accStdList[i], alpha = 0.5)
            yMax = max(zip(self.__accMeanList[i], self.__accStdList[i]))
            yMin = min(zip(self.__accMeanList[i], self.__accStdList[i]))
            self.__aux.set_ylim([yMin[0]*0.1 - yMin[1], yMax[0]*1.5+yMax[1]])
            self.__aux.set_ylabel('Accuracy')
            self.__aux.set_xlabel('$\lambda$')
            self.__aux.set_xticks(xPos, [str(lambdav) for lambdav in self.__lambdaValList])
        self.__aux.legend(['$\delta=1$', '$\delta=2$', '$\delta=3$', '$\delta=4$'])
        targetFilePath = FileChecker(givenFilePath=os.path.join(self.__savedDir, 'Lngxbhbcdexzlhzcxfxyc.pdf')).get()
        plt.savefig(targetFilePath) 
        plt.show()
        return self.__aux
    pass

class AccWrtLambdaPlotterForIITnbswmbbfxyc(object):
    def __init__(self, givenAux):
        self.__aux = givenAux
        self.__lambdaValList = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
        self.__accMeanList = [
            [91.26, 92.50, 93.45, 93.63, 93.63, 93.63, 93.63, 93.63], # delta = 1
            [67.36, 67.96, 67.60, 67.71, 67.75, 67.67, 67.55, 67.32], # delta = 2
            [66.44, 66.70, 67.72, 68.29, 67.74, 67.62, 67.20, 67.63], # delta = 3
            [66.30, 66.70, 67.69, 68.59, 68.72, 68.67, 66.26, 68.66]  # delta = 4
        ]
        self.__accStdList = [
            [3.23, 1.89, 0.19, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.68, 0.41, 0.19, 0.10, 0.00, 0.13, 0.35, 0.76],
            [1.22, 0.95, 0.18, 0.68, 0.00, 0.18, 0.86, 0.20],
            [2.46, 1.86, 1.37, 0.16, 0.00, 0.09, 4.19, 0.05]
        ]
        self.__savedDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', 'accWrtLambda')
        pass

    def execute(self):
        xPos, xShift = list(range(len(self.__lambdaValList))), [-0.3, -0.1, 0.1, 0.3]
        for i in range(len(self.__accMeanList)):
            self.__aux.bar([xPos[j]+xShift[i] for j in range(len(xPos))], self.__accMeanList[i], width = 0.2, yerr = self.__accStdList[i], alpha = 0.5)
            yMax = max(zip(self.__accMeanList[i], self.__accStdList[i]))
            yMin = min(zip(self.__accMeanList[i], self.__accStdList[i]))
            self.__aux.set_ylim([yMin[0]*0.1 - yMin[1], yMax[0]*1.5+yMax[1]])
            self.__aux.set_ylabel('Accuracy')
            self.__aux.set_xlabel('$\lambda$')
            self.__aux.set_xticks(xPos, [str(lambdav) for lambdav in self.__lambdaValList])
        self.__aux.legend(['$\delta=1$', '$\delta=2$', '$\delta=3$', '$\delta=4$'])
        targetFilePath = FileChecker(givenFilePath=os.path.join(self.__savedDir, 'IITnbswmbbfxyc.pdf')).get()
        plt.savefig(targetFilePath) 
        plt.show()
        return self.__aux
    pass

class AccWrtLambdaPlotterForCovid19EventPrediction(object):
    def __init__(self, givenAux):
        self.__aux = givenAux
        self.__lambdaValList = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
        self.__accMeanList = [
            [93.20, 93.40, 94.52, 93.20, 93.20, 95.56, 92.43, 88.68], # delta = 1
            [78.26, 78.32, 78.27, 78.96, 79.17, 79.24, 78.61, 78.75], # delta = 2
            [77.78, 78.89, 78.47, 77.78, 79.31, 79.24, 79.24, 78.90], # delta = 3
            [77.64, 78.26, 77.85, 78.27, 79.17, 78.61, 78.39, 79.03]  # delta = 4
        ]
        self.__accStdList = [
            [3.32, 2.96, 1.93, 4.39, 4.39, 0.24, 5.72, 12.21],
            [1.82, 1.69, 0.85, 0.21, 0.00, 0.12, 0.96, 0.42],
            [2.13, 0.64, 0.84, 2.41, 0.24, 0.12, 0.12, 0.30],
            [1.36, 0.73, 2.10, 1.56, 0.00, 0.96, 1.34, 0.12]
        ]
        self.__savedDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', 'accWrtLambda')
        pass

    def execute(self):
        xPos, xShift = list(range(len(self.__lambdaValList))), [-0.3, -0.1, 0.1, 0.3]
        for i in range(len(self.__accMeanList)):
            self.__aux.bar([xPos[j]+xShift[i] for j in range(len(xPos))], self.__accMeanList[i], width = 0.2, yerr = self.__accStdList[i], alpha = 0.5)
            yMax = max(zip(self.__accMeanList[i], self.__accStdList[i]))
            yMin = min(zip(self.__accMeanList[i], self.__accStdList[i]))
            self.__aux.set_ylim([yMin[0]*0.1 - yMin[1], yMax[0]*1.5+yMax[1]])
            self.__aux.set_ylabel('Accuracy')
            self.__aux.set_xlabel('$\lambda$')
            self.__aux.set_xticks(xPos, [str(lambdav) for lambdav in self.__lambdaValList])
        self.__aux.legend(['$\delta=1$', '$\delta=2$', '$\delta=3$', '$\delta=4$'])
        targetFilePath = FileChecker(givenFilePath=os.path.join(self.__savedDir, 'Covid19EventPrediction.pdf')).get()
        plt.savefig(targetFilePath) 
        plt.show()
        return self.__aux
    pass


class Test(object):
    def __init__(self):
        fig = plt.figure(figsize=(15,5)) 
        self.__aux = fig.add_subplot(1, 1, 1)
        pass

    def testAccWrtLambdaPlotterForLuna(self):
        AccWrtLambdaPlotterForLuna(givenAux=self.__aux).execute()
        pass
    
    def testCommunicationEfficiencyWrtGammaPlotterForLuna(self):
        CommunicationEfficiencyWrtGammaPlotterForLuna(givenAux=self.__aux).execute()
        pass
    
    def testAccWrtLambdaPlotterForBraTS2017(self):
        AccWrtLambdaPlotterForBraTS2017(givenAux=self.__aux).execute()
        pass

    def testAccWrtLambdaPlotterForLngxbhbcdexzlhzcxfxyc(self):
        AccWrtLambdaPlotterForLngxbhbcdexzlhzcxfxyc(givenAux=self.__aux).execute()
        pass

    def testAccWrtLambdaPlotterForIITnbswmbbfxyc(self):
        AccWrtLambdaPlotterForIITnbswmbbfxyc(givenAux=self.__aux).execute()
        pass

    def testAccWrtLambdaPlotterForCovid19EventPrediction(self):
        AccWrtLambdaPlotterForCovid19EventPrediction(givenAux=self.__aux).execute()
        pass
    
    def testCommunicationEfficiencyWrtGammaPlotterForBraTS2017(self):
        CommunicationEfficiencyWrtGammaPlotterForBraTS2017(givenAux=self.__aux).execute()
    pass

Test().testAccWrtLambdaPlotterForBraTS2017()


