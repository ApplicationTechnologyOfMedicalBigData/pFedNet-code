import numpy as np
from sklearn import preprocessing

class ToyDataOfConvexCluster(object):
    def __init__(self, givenNumberOfSamples = 100, givenNumberOfFeatures = 6):
        self.__name = 'toyData'
        self.__dataMat = np.random.randn(givenNumberOfSamples, givenNumberOfFeatures) # numberOfSamples by numberOfFeatures
        self.__numOfSamples, self.__numOfFeatures = self.__dataMat.shape
        
    def getDataMat(self):
        return self.__dataMat
    
    def querySampleList(self, givenSampleIds):
        sampleList = [self.__dataMat[id] for id in givenSampleIds] 
        return np.array(sampleList)
    
    def querySample(self, givenSampleId):
        return self.__dataMat[givenSampleId]
    
    def getNumberOfSamples(self):
        return self.__numOfSamples
    
    def getNumberOfFeatures(self):
        return self.__numOfFeatures
    
    def getName(self):
        return self.__name
    pass

class HalfMoonsDataset(object):
    """
    The halfMoon dataset from the website: https://cs.joensuu.fi/sipu/datasets/jain.txt

    * Features: 2
    * Samples: 373
    * Original label: 1, 2
    * Task: clustering (2 clusters)
    """
    def __init__(self, givenFn):
        self.__name = 'HalfMoons'
        self.__numberOfSamples = 373
        self.__numberOfFeatures = 2
        self.__fn = givenFn
        self.__sampleMat = np.zeros((self.__numberOfSamples, self.__numberOfFeatures)) # numberOfSamples by numberOfFeatures
        self.__labelVec = np.zeros(self.__numberOfSamples)
        self.__init()
        pass

    def __init(self):
        self.__initSample()
        self.__initLabel()

    def __loadData(self):
        with open(file = self.__fn, encoding='utf-8') as file:
            content = file.read()
        for lineId, line in enumerate(content.split('\n')):
            if line.strip() == "": 
                continue
            pairList = line.strip().split(',')
            for pairId, pairItem in enumerate(pairList):
                if pairId == 0:
                    label = pairItem.strip()
                    self.__labelVec[lineId] = float(label)
                if pairId > 0:
                    val = pairItem.strip()
                    self.__sampleMat[lineId, int(pairId)-1] = float(val)
        pass

    def __initSample(self):
        self.__loadData()
        scaler = preprocessing.MaxAbsScaler() # all values locate in [-1, 1].
        self.__sampleMat = scaler.fit_transform(self.__sampleMat)
        pass

    def __initLabel(self):
        pass

    def querySample(self, givenSampleId):
        return self.__sampleMat[givenSampleId]
    
    def queryLabel(self, givenSampleId):
        return self.__labelVec[givenSampleId]
    
    def querySampleList(self, givenSampleIds):
        sampleList = [self.__sampleMat[id] for id in givenSampleIds] 
        return np.array(sampleList)
    
    def getSampleMat(self):
        return self.__sampleMat

    def getLabelVec(self):
        return self.__labelVec
    
    def getNumberOfSamples(self):
        return self.__numberOfSamples
    
    def getNumberOfFeatures(self):
        return self.__numberOfFeatures
    
    def getName(self):
        return self.__name
    pass

class LibSvmDataset(object):
    """
    Dataset is represented like the LIBSVM dataset. 
    Every sample and its label looks like: :math:`[\mathbf{y}, \mathbf{x_1}, \mathbf{x_2}, \mathbf{x_3}]`. There is **NO** header, and the first column, i.e. :math:`\mathbf{y}`, represents label.
    """
    def __init__(self, givenFnPath, givenNumberOfSamples, givenNumberOfFeatures):
        self.__path = givenFnPath
        self.__sampleMat = np.zeros((givenNumberOfSamples, givenNumberOfFeatures))
        self.__labelVec = np.zeros(givenNumberOfSamples)
        self.__numberOfSamples, self.__numOfFeatures = givenNumberOfSamples, givenNumberOfFeatures
        self.__typeOfTask = 'binary classification'
        self.__initData()
        pass

    def __initData(self):
        with open(file = self.__path, encoding='utf-8') as file:
            content = file.read()
        for lineId, line in enumerate(content.split('\n')):
            if line.strip() == "": 
                continue
            pairList = line.strip().split(' ')
            for pairId, pairItem in enumerate(pairList):
                if pairId == 0:
                    label = pairItem.strip()
                    self.__labelVec[lineId] = float(label)
                if pairId > 0:
                    columnId, val = pairItem.split(':') 
                    self.__sampleMat[lineId, int(columnId)-1] = float(val)
        pass

    def getNumberOfSamples(self):
        return self.__numberOfSamples

    def querySample(self, givenId):
        sample = self.__sampleMat[givenId]
        return sample

    def queryLabel(self, givenId):
        label = self.__labelVec[givenId]
        return label
    
    def querySampleList(self, givenIdList):
        sampleList= []
        for id in givenIdList:
            sample = self.querySample(id)
            sampleList.append(sample)
        return sampleList

    def queryLabelList(self, givenIdList):
        labelList= []
        for id in givenIdList:
            label = self.queryLabel(id)
            labelList.append(label)
        return labelList
    
    def getDataMat(self):
        """
        all samples. numberOfSamples by numberOfFeatures
        """
        return self.__sampleMat

    def getLabels(self):
        return self.__labelVec
    
    def getTypeOfTask(self):
        return self.__typeOfTask
    pass

class IrisOfLibsvm(object):
    """
    The iris dataset from the Libsvm website: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#iris
    
    * Features: 4
    * Samples: 150
    * Original label: 1, 2, 3
    * Task: classification (3 class)
    """
    def __init__(self, givenFn):
        self.__name = 'Iris'
        self.__fn = givenFn
        self.__numberOfSamples = 150
        self.__numberOfFeatures = 4
        datasetObj = LibSvmDataset(givenFnPath=self.__fn, givenNumberOfSamples=self.__numberOfSamples, givenNumberOfFeatures=self.__numberOfFeatures)
        self.__sampleMat = datasetObj.getDataMat() # numberOfSamples by numberOfFeatures
        self.__labelVec = datasetObj.getLabels()
        self.__init()
        pass
    
    def __init(self):
        self.__initSample()
        self.__initLabel()

    def __initSample(self):
        pass

    def __initLabel(self):
        pass

    def querySample(self, givenSampleId):
        return self.__sampleMat[givenSampleId]
    
    def queryLabel(self, givenSampleId):
        return self.__labelVec[givenSampleId]
    
    def querySampleList(self, givenSampleIds):
        sampleList = [self.__sampleMat[id] for id in givenSampleIds] 
        return np.array(sampleList)
    
    def getSampleMat(self):
        return self.__sampleMat

    def getLabelVec(self):
        return self.__labelVec
    
    def getNumberOfSamples(self):
        return self.__numberOfSamples
    
    def getNumberOfFeatures(self):
        return self.__numberOfFeatures
    
    def getName(self):
        return self.__name
    pass

class BreastCancerOfLibsvm(object):
    """
    The breast cancer dataset from the Libsvm website: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#breast-cancer
    
    * Features: 10
    * Samples: 683
    * Original label: 2, 4
    * Task: classification (2 class)
    """
    def __init__(self, givenFn):
        self.__name = 'BreastCancer'
        self.__fn = givenFn
        self.__numberOfSamples = 683
        self.__numberOfFeatures = 10
        datasetObj = LibSvmDataset(givenFnPath=self.__fn, givenNumberOfSamples=self.__numberOfSamples, givenNumberOfFeatures=self.__numberOfFeatures)
        self.__sampleMat = datasetObj.getDataMat() # numberOfSamples by numberOfFeatures
        self.__labelVec = datasetObj.getLabels()
        self.__init()
        pass
    
    def __init(self):
        self.__initSample()
        self.__initLabel()
        self.__shuffle()
        pass
    
    def __shuffle(self):
        self.__shuffleSamples()
        self.__shuffleLabels()
        pass
    
    def __shuffleSamples(self, givenRandomSeed = 0):
        """
        re-organize the samples randomly
        """
        np.random.seed(givenRandomSeed)
        self.__sampleMat = np.random.permutation(self.__sampleMat)
        pass
    
    def __shuffleLabels(self, givenRandomSeed = 0):
        """
        re-organize the samples randomly
        """
        np.random.seed(givenRandomSeed)
        self.__labelVec = np.random.permutation(self.__labelVec)
        pass

    def __initSample(self):
        # add the constant feature
        self.__numberOfFeatures += 1
        rawSampleMat = np.column_stack((self.__sampleMat, np.ones(self.__numberOfSamples))) 
        # re-organize the data randomly
        self.__sampleMat = np.random.permutation(rawSampleMat)
        pass

    def __initLabel(self):
        # set labels by '1' and '-1'
        self.__labelVec[self.__labelVec == 2] = -1
        self.__labelVec[self.__labelVec == 4] = 1
        pass

    def querySample(self, givenSampleId):
        return self.__sampleMat[givenSampleId]
    
    def queryLabel(self, givenSampleId):
        return self.__labelVec[givenSampleId]
    
    def querySampleList(self, givenSampleIds):
        sampleList = [self.__sampleMat[id] for id in givenSampleIds] 
        return np.array(sampleList)
    
    def getSampleMat(self):
        return self.__sampleMat

    def getLabelVec(self):
        return self.__labelVec
    
    def getNumberOfSamples(self):
        return self.__numberOfSamples
    
    def getNumberOfFeatures(self):
        return self.__numberOfFeatures

    def getName(self):
        return self.__name
    pass

class CovtypeOfLibsvm(object):
    """
    The covtype dataset from the Libsvm website: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#covtype.binary
    
    * Features: 54
    * Samples: 581,012
    * Original label: 1, 2
    * Task: classification (2 class)
    """
    def __init__(self, givenFn):
        self.__name = 'covtype'
        self.__fn = givenFn
        self.__numberOfSamples = 581012
        self.__numberOfFeatures = 54
        datasetObj = LibSvmDataset(givenFnPath=self.__fn, givenNumberOfSamples=self.__numberOfSamples, givenNumberOfFeatures=self.__numberOfFeatures)
        self.__sampleMat = datasetObj.getDataMat() # numberOfSamples by numberOfFeatures
        self.__labelVec = datasetObj.getLabels()
        self.__init()
        pass
    
    def __init(self):
        self.__initSample()
        self.__initLabel()

    def __initSample(self):
        pass

    def __initLabel(self):
        self.__labelVec[self.__labelVec==2] = -1

    def querySample(self, givenSampleId):
        return self.__sampleMat[givenSampleId]
    
    def queryLabel(self, givenSampleId):
        return self.__labelVec[givenSampleId]
    
    def querySampleList(self, givenSampleIds):
        sampleList = [self.__sampleMat[id] for id in givenSampleIds] 
        return np.array(sampleList)
    
    def getSampleMat(self):
        return self.__sampleMat

    def getLabelVec(self):
        return self.__labelVec
    
    def getNumberOfSamples(self):
        return self.__numberOfSamples
    
    def getNumberOfFeatures(self):
        return self.__numberOfFeatures
    
    def getName(self):
        return self.__name
    pass

class Ijcnn1OfLibsvm(object):
    """
    The ijcnn dataset from the Libsvm website: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#ijcnn1
    
    * Features: 22
    * Samples: 49,990 + 91701
    * Original label: 1, -1
    * Task: classification (2 class)
    """
    def __init__(self, givenFn):
        self.__name = 'ijcnn1'
        self.__fn = givenFn
        self.__numberOfSamples = 49990
        self.__numberOfFeatures = 22
        datasetObj = LibSvmDataset(givenFnPath=self.__fn, givenNumberOfSamples=self.__numberOfSamples, givenNumberOfFeatures=self.__numberOfFeatures)
        self.__sampleMat = datasetObj.getDataMat() # numberOfSamples by numberOfFeatures
        self.__labelVec = datasetObj.getLabels()
        self.__init()
        pass
    
    def __init(self):
        self.__initSample()
        self.__initLabel()
        self.__shuffle()
        pass
    
    def __shuffle(self):
        self.__shuffleSamples()
        self.__shuffleLabels()
        pass
    
    def __shuffleSamples(self, givenRandomSeed = 0):
        """
        re-organize the samples randomly
        """
        np.random.seed(givenRandomSeed)
        self.__sampleMat = np.random.permutation(self.__sampleMat)
        pass
    
    def __shuffleLabels(self, givenRandomSeed = 0):
        """
        re-organize the samples randomly
        """
        np.random.seed(givenRandomSeed)
        self.__labelVec = np.random.permutation(self.__labelVec)
        pass

    def __initSample(self):
        # add the constant feature
        self.__numberOfFeatures += 1
        self.__sampleMat = np.column_stack((self.__sampleMat, np.ones(self.__numberOfSamples))) 
        pass

    def __initLabel(self):
        pass

    def querySample(self, givenSampleId):
        return self.__sampleMat[givenSampleId]
    
    def queryLabel(self, givenSampleId):
        return self.__labelVec[givenSampleId]
    
    def querySampleList(self, givenSampleIds):
        sampleList = [self.__sampleMat[id] for id in givenSampleIds] 
        return np.array(sampleList)

    def getSampleMat(self):
        return self.__sampleMat

    def getLabelVec(self):
        return self.__labelVec
    
    def getNumberOfSamples(self):
        return self.__numberOfSamples
    
    def getNumberOfFeatures(self):
        return self.__numberOfFeatures
    
    def getName(self):
        return self.__name
    pass

class SusyOfLibsvm(object):
    """
    The Susy dataset from the Libsvm website: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#SUSY
    
    * Features: 18
    * Samples: 5,000,000
    * Original label: ?????????
    * Task: classification (2 class)
    """
    def __init__(self, givenFn):
        self.__name = 'Susy'
        self.__fn = givenFn
        self.__numberOfSamples = 5000000
        self.__numberOfFeatures = 18
        datasetObj = LibSvmDataset(givenFnPath=self.__fn, givenNumberOfSamples=self.__numberOfSamples, givenNumberOfFeatures=self.__numberOfFeatures)
        self.__sampleMat = datasetObj.getDataMat() # numberOfSamples by numberOfFeatures
        self.__labelVec = datasetObj.getLabels()
        self.__init()
        pass
    
    def __init(self):
        self.__initSample()
        self.__initLabel()

    def __initSample(self):
        pass

    def __initLabel(self):
        pass

    def querySample(self, givenSampleId):
        return self.__sampleMat[givenSampleId]
    
    def queryLabel(self, givenSampleId):
        return self.__labelVec[givenSampleId]
    
    def querySampleList(self, givenSampleIds):
        sampleList = [self.__sampleMat[id] for id in givenSampleIds] 
        return np.array(sampleList)

    def getSampleMat(self):
        return self.__sampleMat

    def getLabelVec(self):
        return self.__labelVec
    
    def getNumberOfSamples(self):
        return self.__numberOfSamples
    
    def getNumberOfFeatures(self):
        return self.__numberOfFeatures

    def getName(self):
        return self.__name
    pass

class HiggsOfLibsvm(object):
    """
    The ijcnn dataset from the Libsvm website: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#SUSY
    
    * Features: 28
    * Samples: 11,000,000
    * Original label: ?????????
    * Task: classification (2 class)
    """
    def __init__(self, givenFn):
        self.__name = 'Higgs'
        self.__fn = givenFn
        self.__numberOfSamples = 11000000
        self.__numberOfFeatures = 28
        datasetObj = LibSvmDataset(givenFnPath=self.__fn, givenNumberOfSamples=self.__numberOfSamples, givenNumberOfFeatures=self.__numberOfFeatures)
        self.__sampleMat = datasetObj.getDataMat() # numberOfSamples by numberOfFeatures
        self.__labelVec = datasetObj.getLabels()
        self.__init()
        pass
    
    def __init(self):
        self.__initSample()
        self.__initLabel()

    def __initSample(self):
        pass

    def __initLabel(self):
        pass

    def querySample(self, givenSampleId):
        return self.__sampleMat[givenSampleId]
    
    def queryLabel(self, givenSampleId):
        return self.__labelVec[givenSampleId]
    
    def querySampleList(self, givenSampleIds):
        sampleList = [self.__sampleMat[id] for id in givenSampleIds] 
        return np.array(sampleList)

    def getSampleMat(self):
        return self.__sampleMat

    def getLabelVec(self):
        return self.__labelVec
    
    def getNumberOfSamples(self):
        return self.__numberOfSamples
    
    def getNumberOfFeatures(self):
        return self.__numberOfFeatures
    
    def getName(self):
        return self.__name
    pass

class StructureMedicalDatasets(object):
    """
    Every sample and its label looks like: :math:`[\mathbf{y}, \mathbf{x1}, \mathbf{x2}, \mathbf{x3}]`. There **HAS** header, and the first column, i.e. :math:`\mathbf{y}`, represents label. 
    When conducting binary classification, labels include '-1' and '1'.
    """
    def __init__(self, givenFnPath, givenNumberOfSamples, givenNumberOfFeatures):
        self.__path = givenFnPath
        self.__sampleMat = np.zeros((givenNumberOfSamples, givenNumberOfFeatures))
        self.__labelVec = np.zeros(givenNumberOfSamples)
        self.__headerList = []
        self.__numberOfSamples, self.__numOfFeatures = givenNumberOfSamples, givenNumberOfFeatures
        self.__typeOfTask = 'binary classification'
        self.__initData()
        pass

    def __initData(self):
        with open(file = self.__path, encoding='utf-8') as file:
            content = file.read()
        rowId = 0
        for lineId, line in enumerate(content.split('\n')):
            if line.strip() == "":
                continue
            if lineId == 0: # the first line is the header. 
                self.__headerList = line.strip().split(',')
                continue 
            pairList = line.strip().split(',')
            for pairId, pairItem in enumerate(pairList):
                if pairId == 0:
                    label = pairItem.strip()
                    self.__labelVec[rowId] = float(label)
                if pairId > 0:
                    if pairItem == '' or pairItem == 'NA': # missing value
                        self.__sampleMat[rowId, int(pairId)-1] = 0
                        continue
                    self.__sampleMat[rowId, int(pairId)-1] = float(pairItem)
            rowId += 1
        pass

    def getNumberOfSamples(self):
        return self.__numberOfSamples

    def querySample(self, givenId):
        sample = self.__sampleMat[givenId]
        return sample

    def queryLabel(self, givenId):
        label = self.__labelVec[givenId]
        return label
    
    def querySampleList(self, givenIdList):
        sampleList= []
        for id in givenIdList:
            sample = self.querySample(id)
            sampleList.append(sample)
        return sampleList

    def queryLabelList(self, givenIdList):
        labelList= []
        for id in givenIdList:
            label = self.queryLabel(id)
            labelList.append(label)
        return labelList
    
    def getDataMat(self):
        """
        all samples. numberOfSamples by numberOfFeatures
        """
        return self.__sampleMat

    def getLabels(self):
        return self.__labelVec
    
    def getHeaders(self):
        return self.__headerList
    
    def getTypeOfTask(self):
        return self.__typeOfTask
    pass

class LngxbhbcdexzlhzcxfxycDataset(object):
    """
    老年冠心病合并肠道恶性肿瘤患者出血风险预测数据集
    
    * Features: 58
    * Samples: 716
    * Original label: 0, 1
    * Task: classification (2 class)
    """
    def __init__(self, givenFn):
        self.__name = 'LngxbhbcdexzlhzcxfxycDataset'
        self.__fn = givenFn
        self.__numberOfSamples = 716
        self.__numberOfFeatures = 58
        datasetObj = StructureMedicalDatasets(givenFnPath=self.__fn, givenNumberOfSamples=self.__numberOfSamples, givenNumberOfFeatures=self.__numberOfFeatures)
        self.__sampleMat = datasetObj.getDataMat() # numberOfSamples by numberOfFeatures
        self.__labelVec = datasetObj.getLabels()
        self.__init()
        pass
    
    def __init(self):
        self.__initSample()
        self.__initLabel()
        self.__shuffle()
        pass
    
    def __shuffle(self):
        self.__shuffleSamples()
        self.__shuffleLabels()
        pass
    
    def __shuffleSamples(self, givenRandomSeed = 0):
        """
        re-organize the samples randomly
        """
        np.random.seed(givenRandomSeed)
        self.__sampleMat = np.random.permutation(self.__sampleMat)
        pass
    
    def __shuffleLabels(self, givenRandomSeed = 0):
        """
        re-organize the samples randomly
        """
        np.random.seed(givenRandomSeed)
        self.__labelVec = np.random.permutation(self.__labelVec)
        pass

    def __initSample(self):
        scaler = preprocessing.MaxAbsScaler() # all values locate in [-1, 1].
        self.__sampleMat = scaler.fit_transform(self.__sampleMat)
        self.__numberOfFeatures += 1 # add the constant feature
        self.__sampleMat = np.column_stack((self.__sampleMat, np.ones(self.__numberOfSamples))) 
        pass

    def __initLabel(self):
        self.__labelVec[self.__labelVec==0] = -1
        pass

    def querySample(self, givenSampleId):
        return self.__sampleMat[givenSampleId]
    
    def queryLabel(self, givenSampleId):
        return self.__labelVec[givenSampleId]
    
    def querySampleList(self, givenSampleIds):
        sampleList = [self.__sampleMat[id] for id in givenSampleIds] 
        return np.array(sampleList)
    
    def getSampleMat(self):
        return self.__sampleMat

    def getLabelVec(self):
        return self.__labelVec

    def getNumberOfSamples(self):
        return self.__numberOfSamples
    
    def getNumberOfFeatures(self):
        return self.__numberOfFeatures
    
    def getName(self):
        return self.__name
    pass

class LngxbhbwexzlhzcxfxycDataset(object):
    """
    老年冠心病合并胃恶性肿瘤患者出血风险预测数据集
    
    * Features: 61
    * Samples: 594
    * Original label: 0, 1
    * Task: classification (2 class)
    """
    def __init__(self, givenFn):
        self.__name = 'LngxbhbwexzlhzcxfxycDataset'
        self.__fn = givenFn
        self.__numberOfSamples = 594
        self.__numberOfFeatures = 61
        datasetObj = StructureMedicalDatasets(givenFnPath=self.__fn, givenNumberOfSamples=self.__numberOfSamples, givenNumberOfFeatures=self.__numberOfFeatures)
        self.__sampleMat = datasetObj.getDataMat() # numberOfSamples by numberOfFeatures
        self.__labelVec = datasetObj.getLabels()
        self.__init()
        pass
    
    def __init(self):
        self.__initSample()
        self.__initLabel()
        self.__shuffle()
        pass
    
    def __shuffle(self):
        self.__shuffleSamples()
        self.__shuffleLabels()
        pass
    
    def __shuffleSamples(self, givenRandomSeed = 0):
        """
        re-organize the samples randomly
        """
        np.random.seed(givenRandomSeed)
        self.__sampleMat = np.random.permutation(self.__sampleMat)
        pass
    
    def __shuffleLabels(self, givenRandomSeed = 0):
        """
        re-organize the samples randomly
        """
        np.random.seed(givenRandomSeed)
        self.__labelVec = np.random.permutation(self.__labelVec)
        pass
    
    def __initSample(self):
        scaler = preprocessing.MaxAbsScaler() # all values locate in [-1, 1].
        self.__sampleMat = scaler.fit_transform(self.__sampleMat)
        self.__numberOfFeatures += 1 # add the constant feature
        self.__sampleMat = np.column_stack((self.__sampleMat, np.ones(self.__numberOfSamples))) 
        pass

    def __initLabel(self):
        self.__labelVec[self.__labelVec==0] = -1
        pass

    def querySample(self, givenSampleId):
        return self.__sampleMat[givenSampleId]
    
    def queryLabel(self, givenSampleId):
        return self.__labelVec[givenSampleId]
    
    def querySampleList(self, givenSampleIds):
        sampleList = [self.__sampleMat[id] for id in givenSampleIds] 
        return np.array(sampleList)
    
    def getSampleMat(self):
        return self.__sampleMat

    def getLabelVec(self):
        return self.__labelVec

    def getNumberOfSamples(self):
        return self.__numberOfSamples
    
    def getNumberOfFeatures(self):
        return self.__numberOfFeatures
    
    def getName(self):
        return self.__name
    pass

class IITnbswmbbfxycDataset(object):
    """
    2型糖尿病视网膜病变风险预测数据集
    
    * Features: 63
    * Samples: 31476
    * Original label: 0, 1
    * Task: classification (2 class)
    """
    def __init__(self, givenFn):
        self.__name = 'IITnbswmbbfxyc'
        self.__fn = givenFn
        self.__numberOfSamples = 31476
        self.__numberOfFeatures = 63
        datasetObj = StructureMedicalDatasets(givenFnPath=self.__fn, givenNumberOfSamples=self.__numberOfSamples, givenNumberOfFeatures=self.__numberOfFeatures)
        self.__sampleMat = datasetObj.getDataMat() # numberOfSamples by numberOfFeatures
        self.__labelVec = datasetObj.getLabels()
        self.__init()
        pass
    
    def __init(self):
        self.__initSample()
        self.__initLabel()
        self.__shuffle()
        pass
    
    def __shuffle(self):
        self.__shuffleSamples()
        self.__shuffleLabels()
        pass
    
    def __shuffleSamples(self, givenRandomSeed = 0):
        """
        re-organize the samples randomly
        """
        np.random.seed(givenRandomSeed)
        self.__sampleMat = np.random.permutation(self.__sampleMat)
        pass
    
    def __shuffleLabels(self, givenRandomSeed = 0):
        """
        re-organize the samples randomly
        """
        np.random.seed(givenRandomSeed)
        self.__labelVec = np.random.permutation(self.__labelVec)
        pass

    def __initSample(self):
        scaler = preprocessing.MaxAbsScaler() # all values locate in [-1, 1].
        self.__sampleMat = scaler.fit_transform(self.__sampleMat)
        self.__numberOfFeatures += 1 # add the constant feature
        self.__sampleMat = np.column_stack((self.__sampleMat, np.ones(self.__numberOfSamples))) 
        pass

    def __initLabel(self):
        self.__labelVec[self.__labelVec==0] = -1
        pass

    def querySample(self, givenSampleId):
        return self.__sampleMat[givenSampleId]
    
    def queryLabel(self, givenSampleId):
        return self.__labelVec[givenSampleId]
    
    def querySampleList(self, givenSampleIds):
        sampleList = [self.__sampleMat[id] for id in givenSampleIds] 
        return np.array(sampleList)
    
    def getSampleMat(self):
        return self.__sampleMat

    def getLabelVec(self):
        return self.__labelVec

    def getNumberOfSamples(self):
        return self.__numberOfSamples
    
    def getNumberOfFeatures(self):
        return self.__numberOfFeatures
    
    def getName(self):
        return self.__name
    pass

class Covid19EventPredictionDataset(object):
    """
    covid-19 event prediction
    
    * Features: 77
    * Samples: 2402
    * Original label: 0, 1
    * Task: classification
    """
    def __init__(self, givenFn):
        self.__name = 'covid-19 event prediction'
        self.__fn = givenFn
        self.__numberOfSamples = 2402
        self.__numberOfFeatures = 77
        datasetObj = StructureMedicalDatasets(givenFnPath=self.__fn, givenNumberOfSamples=self.__numberOfSamples, givenNumberOfFeatures=self.__numberOfFeatures)
        self.__sampleMat = datasetObj.getDataMat() # numberOfSamples by numberOfFeatures
        self.__labelVec = datasetObj.getLabels()
        self.__headers = datasetObj.getHeaders()
        self.__init()
        pass
    
    def __init(self):
        self.__initSample()
        self.__initLabel()
        self.__shuffle()
        pass
    
    def __shuffle(self):
        self.__shuffleSamples()
        self.__shuffleLabels()
        pass
    
    def __shuffleSamples(self, givenRandomSeed = 0):
        """
        re-organize the samples randomly
        """
        np.random.seed(givenRandomSeed)
        self.__sampleMat = np.random.permutation(self.__sampleMat)
        pass
    
    def __shuffleLabels(self, givenRandomSeed = 0):
        """
        re-organize the samples randomly
        """
        np.random.seed(givenRandomSeed)
        self.__labelVec = np.random.permutation(self.__labelVec)
        pass

    def __initSample(self):
        scaler = preprocessing.MaxAbsScaler() # all values locate in [-1, 1].
        self.__sampleMat = scaler.fit_transform(self.__sampleMat)
        self.__numberOfFeatures += 1 # add the constant feature
        self.__sampleMat = np.column_stack((self.__sampleMat, np.ones(self.__numberOfSamples))) 
        pass

    def __initLabel(self):
        self.__labelVec[self.__labelVec==0] = -1
        pass

    def querySample(self, givenSampleId):
        return self.__sampleMat[givenSampleId]
    
    def queryLabel(self, givenSampleId):
        return self.__labelVec[givenSampleId]
    
    def querySampleList(self, givenSampleIds):
        sampleList = [self.__sampleMat[id] for id in givenSampleIds] 
        return np.array(sampleList)
    
    def getSampleMat(self):
        return self.__sampleMat

    def getLabelVec(self):
        return self.__labelVec

    def getNumberOfSamples(self):
        return self.__numberOfSamples
    
    def getNumberOfFeatures(self):
        return self.__numberOfFeatures
    
    def getName(self):
        return self.__name
    
    def getHeaders(self):
        return self.__headers
    pass

class AkiPredictionDataset(object):
    """
    AKI prediction
    
    * Features: 79
    * Samples: 18350
    * Original label: -1, 1
    * Task: classification
    """
    def __init__(self, givenFn):
        self.__name = 'Aki detection'
        self.__fn = givenFn
        self.__numberOfSamples = 18350
        self.__numberOfFeatures = 79
        datasetObj = StructureMedicalDatasets(givenFnPath=self.__fn, givenNumberOfSamples=self.__numberOfSamples, givenNumberOfFeatures=self.__numberOfFeatures)
        self.__sampleMat = datasetObj.getDataMat() # numberOfSamples by numberOfFeatures
        self.__labelVec = datasetObj.getLabels()
        self.__headers = datasetObj.getHeaders()
        self.__init()
        pass
    
    def __init(self):
        self.__initSample()
        self.__initLabel()
        self.__shuffle()
        pass
    
    def __shuffle(self):
        self.__shuffleSamples()
        self.__shuffleLabels()
        pass
    
    def __shuffleSamples(self, givenRandomSeed = 0):
        """
        re-organize the samples randomly
        """
        np.random.seed(givenRandomSeed)
        self.__sampleMat = np.random.permutation(self.__sampleMat)
        pass
    
    def __shuffleLabels(self, givenRandomSeed = 0):
        """
        re-organize the samples randomly
        """
        np.random.seed(givenRandomSeed)
        self.__labelVec = np.random.permutation(self.__labelVec)
        pass

    def __initSample(self):
        scaler = preprocessing.MaxAbsScaler() # all values locate in [-1, 1].
        self.__sampleMat = scaler.fit_transform(self.__sampleMat)
        self.__numberOfFeatures += 1 # add the constant feature
        self.__sampleMat = np.column_stack((self.__sampleMat, np.ones(self.__numberOfSamples))) 
        pass

    def __initLabel(self):
        self.__labelVec[self.__labelVec==0] = -1
        pass

    def querySample(self, givenSampleId):
        return self.__sampleMat[givenSampleId]
    
    def queryLabel(self, givenSampleId):
        return self.__labelVec[givenSampleId]
    
    def querySampleList(self, givenSampleIds):
        sampleList = [self.__sampleMat[id] for id in givenSampleIds] 
        return np.array(sampleList)
    
    def getSampleMat(self):
        return self.__sampleMat

    def getLabelVec(self):
        return self.__labelVec

    def getNumberOfSamples(self):
        return self.__numberOfSamples
    
    def getNumberOfFeatures(self):
        return self.__numberOfFeatures
    
    def getName(self):
        return self.__name
    
    def getHeaders(self):
        return self.__headers
    pass







class Test(object):
    def testCovtype(self):
        covtypeDatasetObj = CovtypeOfLibsvm(givenFn='/home/yawei/data/libsvmDatasets/classification/covtype.txt')
        sampleMat = covtypeDatasetObj.getSampleMat()
        labelVec = covtypeDatasetObj.queryLabel()
        pass

    def testLngxbhbcdexzlhzcxfxycDataset(self):
        datasetObj = LngxbhbcdexzlhzcxfxycDataset(givenFn='/home/yawei/data/medicalDatasets/real/classification/老年冠心病合并肠道恶性肿瘤患者的出血风险预测.csv')
        sampleMat = datasetObj.getSampleMat()
        labelVec = datasetObj.getLabelVec()

    def testHalfMoonDatasets(self):
        moonObj = HalfMoonsDataset(givenFn='/home/yawei/data/others/clustering/halfMoons.csv')
        sampleMat = moonObj.getSampleMat()
        pass
    
    def testCovid19EventPredictionDataset(self):
        covid19 = Covid19EventPredictionDataset(givenFn='/home/yawei/data/medicalDatasets/real/classification/covid19EventPrediction.csv')
        sampleMat = covid19.getSampleMat()
        headers = covid19.getHeaders()
        
    def testAkiPredictionDataset(self):
        covid19 = AkiPredictionDataset(givenFn='/home/yawei/data/medicalDatasets/real/classification/AkiPrediction.csv')
        sampleMat = covid19.getSampleMat()
        headers = covid19.getHeaders()
        pass

if __name__ == '__main__':
    Test().testAkiPredictionDataset()


