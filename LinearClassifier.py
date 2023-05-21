import numpy as np

class LinearClassifier:
    def __init__(self,lr=0.01,epoch=10000,verbose=False):
        self.threshold = 0.5
        self.lr=lr
        self.epoch=epoch
        self.verbose=verbose
    def add_b2w(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    def safe_sigmoid(self, inx):
        if inx >= 0 :
            return 1.0 / (1 + np.exp(-inx))
        else :
            return np.exp(inx) / (1 + np.exp(inx))
    def sigmoid(self, z):
        fv= np.vectorize(self.safe_sigmoid)
        return fv(z)
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    def loss(self, z, y):
        return (-y*z+np.log(1+np.exp(z))).mean()

    def fit(self, X_train, y_train):
        X_train=self.add_b2w(X_train)
        self.X = X_train
        self.y = y_train

        self.beta = np.zeros(X_train.shape[1])
        
        for e in range(self.epoch):
            z = np.dot(self.X, self.beta)
            h = self.sigmoid(z)
            gradient = np.dot(self.X.T, (h - self.y)) / self.y.size
            self.beta -= self.lr * gradient
            
            if(self.verbose == True and e % 10000 == 0):
                z = np.dot(self.X, self.beta)
                h = self.sigmoid(z)
                print(f'loss: {self.__loss(h, self.y)} \t')
        
    def predict(self, X_test):
        X_test=self.add_b2w(X_test)
        y_pred = self.sigmoid(np.dot(X_test, self.beta))
        for i in range(len(y_pred)):
            if y_pred[i] > self.threshold:
                y_pred[i]=1
            else:
                y_pred[i]=0
        return y_pred

