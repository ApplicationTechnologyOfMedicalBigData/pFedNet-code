import numpy as np
import cvxpy as cp
from settingsOfpFedNet import QMatrix, QQMatrix, FormulationSettings, GraphGenerater
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from settings import DataFederation, FederatedLearningSettings

class ADMM(object):
    """
    Optimize Z by performing:
    
    .. math:: \min_{\\mathbf{Z}} \\frac{\\mathbf{1}_{d}^\\top ((\\mathbf{N}^\\top \\mathbf{\\nabla}) \odot \\mathbf{Z}) \\mathbf{1}_N}{N} + \lambda \left \| \\mathbf{Z}\\mathbf{Q} \\right\|_{1,p} + \\frac{1}{2 \eta_t} \\left\| \\mathbf{Z} - \\mathbf{Z}_t \\right\|_F^2
    """
    def __init__(self, givenQ, givenNabla, givenNMat, givenZ, givenEta, givenLambda):
        self.__Q = givenQ
        self.__N, self.__M = self.__Q.shape
        self.__Nabla = givenNabla # d by N
        self.__NMat = givenNMat # d by d2
        self.__Zt = givenZ # d2 by N
        self.__d, self.__d2 = self.__NMat.shape
        QQ = QQMatrix(givenQMatrix=self.__Q, givenTransposeOperatorLoc='right')
        self.__P = QQ.getEigenVec()
        self.__sigma = QQ.getEigenValue()
        self.__eta = givenEta
        self.__lambda = givenLambda
        self.__Z, self.__W, self.__Omega = np.zeros((self.__d2, self.__N)), np.zeros((self.__d2, self.__M)), np.zeros((self.__d2, self.__M))
        self.__K = 25
        self.execute()
        pass

    def updateZbyCVX(self):
        """
        Update :math:`\\mathbf{Z}` by performing:
        
        .. math:: \min_{\\mathbf{Z}} \\frac{\\mathbf{1}_{d}^\\top ((\\mathbf{N}^\\top \\mathbf{\\nabla}) \odot \\mathbf{Z}) \\mathbf{1}_N}{N} + \\frac{1}{2 \eta_t} || \\mathbf{Z} - \\mathbf{Z}_t ||_F^2 + \\mathbf{1}_{d_2}^\\top (\\mathbf{\Omega}_k \odot (\\mathbf{Z}\\mathbf{Q}))\\mathbf{1}_{M}  + \\frac{\\rho}{2}\\left\| \\mathbf{Z}\\mathbf{Q} - \\mathbf{\Omega}_k \\right\|_F^2
        """
        Z = cp.Variable(self.__Z.shape)
        hZ = cp.sum(cp.sum(cp.multiply((self.__NMat.T @ self.__Nabla), Z))) / self.__N + 1.0/(2*self.__eta) * (cp.norm(Z - self.__Zt, 'fro') ** 2)
        middle = cp.sum(cp.sum(cp.multiply(self.__Omega, (Z @ self.__Q))))
        right = 0.5*(cp.norm(self.__W - Z @ self.__Q, 'fro') ** 2)
        loss = cp.Minimize( hZ + middle + right)
        problem = cp.Problem(loss)
        problem.solve()
        self.__Z = Z.value
        return Z.value

    def __updateZ(self):
        """
        Update Z by performing:

        .. math:: 
        - min_Z 1 ((N Nabla) \cdot Z) 1/N + 1/(2 eta) || Z - Z_t ||_F^2
        """
        left = self.__eta * (self.__W @ self.__Q.T - self.__Omega @ self.__Q.T - (self.__NMat.T @ self.__Nabla)/self.__N) + self.__Zt
        right = self.__P @ np.diag(1.0 / (1+self.__eta*self.__sigma)) @ self.__P.T
        self.__Z = left @ right
        return self.__Z

    def __updateWbyCVX(self):
        W = cp.Variable(self.__W.shape)
        reg = 0
        for columni in range(self.__W.shape[1]):
            reg = reg + cp.norm2(W[:, columni])
        left = self.__lambda * reg 
        right = 0.5*(cp.norm(W - (self.__Z @ self.__Q + self.__Omega),'fro') ** 2)
        loss = cp.Minimize( left + right)
        problem = cp.Problem(loss)
        problem.solve()
        self.__W = W.value
        return W.value

    def __updateW(self):
        for m in range(self.__W.shape[1]):
            right = self.__Z @ self.__Q[:,m] + self.__Omega[:,m]
            left = np.maximum(1 - self.__lambda / np.linalg.norm(right, 2), 0)
            self.__W[:,m] = left * right
        return self.__W

    def __updateOmega(self):
        self.__Omega = self.__Omega + (self.__Z @ self.__Q - self.__W)
        pass

    def execute(self):
        for k in range(self.__K):
            # update Z
            self.__updateZ()
            # update W
            self.__updateW()
            # update Omega
            self.__updateOmega()
            pass
        pass

    def getZ(self):
        return self.__Z    
    pass

class AlternativeOptimizer(object):
    """
    Perform one step of alternative optimization.
    """
    def __init__(self, givenDataFederation, givenFormulationSettings, givenFederatedLearningSettings, givenSpecificModelObj):
        assert(isinstance(givenDataFederation, DataFederation))
        assert(isinstance(givenFormulationSettings, FormulationSettings))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettings))
        self.__dataFederation = givenDataFederation
        self.__formulationSettings = givenFormulationSettings
        self.__federatedLearningSettings = givenFederatedLearningSettings
        self.__specificModelObj = givenSpecificModelObj
        self.__MMat, self.__NMat = self.__formulationSettings.getMMat(), self.__formulationSettings.getNMat()
        graphObj = GraphGenerater(givenDataFederation=self.__dataFederation, givenGraphName='similarity graph').get()
        self.__QMat = QMatrix(givenGraph=graphObj).get()
        self.__d1, self.__d2, self.__N = self.__MMat.shape[1], self.__NMat.shape[1], self.__federatedLearningSettings.getNumOfClients()
        self.__x, self.__Z = np.zeros(self.__d1), np.zeros((self.__d2, self.__N))
        self.__personalizedModelVecs = np.einsum('i,j->ij', self.__MMat @ self.__x, np.ones(self.__N)) + self.__NMat @ self.__Z # d by N
        pass
    
    def computePersonalizedGradients(self):
        personalizedGrads = self.__specificModelObj.computePersonalizedGradients()
        return personalizedGrads
    
    def updatePersonalizedModel(self, givenPersonalizedModel):
        self.__measureCommunicationEfficiencyAtClient(givenUpdate=self.computePersonalizedGradients())
        self.__measureCommunicationEfficiencyAtServer(givenUpdate=givenPersonalizedModel - self.__personalizedModelVecs)
        self.__personalizedModelVecs = givenPersonalizedModel
        pass
    
    def getPersonalizedModel(self):
        self.__personalizedModelVecs = np.einsum('i,j->ij', self.__MMat @ self.__x, np.ones(self.__N)) + self.__NMat @ self.__Z # d by N
        return self.__personalizedModelVecs

    def updateX(self):
        update = self.__updateXbySGD()
        return update

    def updateZ(self):
        update = self.__updateZbySGD()
        return update
    
    def __updateXbySGD(self):
        """
        Update x by solving::

        min_{x} \frac{1}{N} \sum_{n=1}^N f_n(Mx + Nz^{(n)}; D_n)
        """
        xOld = self.__x
        for ix in range(1,100):
            self.__personalizedModelVecs = np.einsum('i,j->ij', self.__MMat @ self.__x, np.ones(self.__N)) + self.__NMat @ self.__Z # d by N
            nablaMat = self.computePersonalizedGradients()
            alpha = 1e-1 / np.sqrt(ix)
            self.__x = self.__x - alpha * (self.__MMat.T @ nablaMat) @ np.ones(self.__N) / self.__N 
        return self.__x - xOld

    def __updateZbyCVX(self, givenNablaMat, givenEta, givenLambda):
        """
        Update Z by solving:
        - min_{Z} 1/N 1 (N G) \cdot Z 1 + lambda || ZQ ||_{1,p} + 1/(2 eta) || Z - Z_t ||_F^2
        """
        Z = cp.Variable((self.__Z.shape))
        reg = 0
        for columni in range(self.__QMat.shape[1]):
            reg = reg + cp.norm2(Z @ self.__QMat[:,columni])
        loss = cp.Minimize(1.0 / self.__N * cp.sum(cp.sum(cp.multiply(self.__NMat.T @ givenNablaMat, Z))) + givenLambda * reg + 0.5/givenEta*(cp.norm(Z-self.__Z, 'fro') ** 2))
        problem = cp.Problem(loss)
        problem.solve()
        self.__Z = Z.value
        return Z.value

    def __updateZbySGD(self):
        """
        Update Z by solving:
        - min_{Z} 1/N 1 (N G) \cdot Z 1 + lambda || ZQ ||_{1,p} + 1/(2 eta) || Z - Z_t ||_F^2
        """
        zOld = self.__Z
        for iz in range(1,100):
            self.__personalizedModelVecs = np.einsum('i,j->ij', self.__MMat @ self.__x, np.ones(self.__N)) + self.__NMat @ self.__Z # d by N
            nablaMat = self.computePersonalizedGradients()
            alpha, lambdav = 1e-1/np.sqrt(iz), self.__formulationSettings.getLambda()
            self.__Z = ADMM(givenQ=self.__QMat, givenNabla=nablaMat, givenNMat=self.__NMat, givenZ = self.__Z, givenEta=alpha, givenLambda=lambdav).getZ() 
            pass
        return self.__Z - zOld
    pass

















