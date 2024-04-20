# params
# forward(x): y_
# loss(y, y_): mse
# backward(x, y): grads
# optimizer: sgd、adam for param update

# More General Flow
"""
1. class DataLoader(data_params): batch_generator
2. class Model(model_params): forward, loss
   class Optimizer(optimizer_params): params's update method
3. class Trainer(train_params, data_loader, model, optimizer)
"""

import math

def sigmoid(z):
    if z<-100:
        return 0.
    elif z>100:
        return 1.
    return 1 / (1+math.exp(-z))

class LogisticRegressionModel:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.weights = [0.]*hidden_size
        self.bias = 0.

    def forward(self, x):
        y_ = [sum(w*t for w,t in zip(self.weights, x_point)) + self.bias for x_point in x]
        y_ = [sigmoid(t) for t in y_]
        return y_

    def backward(self, x, y, y_):
        weights_bias_grads = [0.] * (self.hidden_size+1)
        for j in range(len(y)):
            for i in range(self.hidden_size):
                error = y_[j] - y[j]
                weights_bias_grads[i] += error*x[j][i]    # bce_grad
            weights_bias_grads[-1] += error
        n = len(y)
        weights_bias_grads = [g/n for g in weights_bias_grads]
        return weights_bias_grads

    def loss(self, y_, y, epsilon=1e-8):
        bce = sum(-t*math.log(t_+epsilon,math.e)-(1-t)*math.log(1-t_+epsilon, math.e) for t_,t in zip(y_,y))/len(y)
        return bce

    def sgd(self, weights_bias_grads, lr):
        for i in range(self.hidden_size):
            self.weights[i] -= lr*weights_bias_grads[i]
        self.bias -= lr*weights_bias_grads[-1]

    def train(self, x, y, epochs, batch_size, lr):
        total_batches = len(x) // batch_size
        for epoch in range(epochs):
            for batch_idx in range(total_batches):
                batch_x = x[batch_idx*batch_size:(batch_idx+1)*batch_size]
                batch_y = y[batch_idx*batch_size:(batch_idx+1)*batch_size]
                batch_y_ = self.forward(batch_x)
                bce = self.loss(batch_y, batch_y_)
                weights_bias_grads = self.backward(batch_x, batch_y, batch_y_)
                self.sgd(weights_bias_grads, lr)
                print(f"epoch: {epoch}/{epochs}, steps: {batch_idx+1}/{total_batches}, bce_loss: {bce:.4f}")

    def val(self, x, y):
        pass

    def infer(self, x):
        y_ = self.forward(x)
        return y_

if __name__ == "__main__":
    x = [[0.88,0.92], [3.4,3], [1.3,1.2], [3.5,2.8], [1.2,0.48], [2.7,3.2]]
    y = [0., 1., 0., 1., 0., 1.]

    LRModel = LogisticRegressionModel(2)
    LRModel.train(x, y, epochs=2000, batch_size=3, lr=1e-2)
    print(f"LRModel weights&bias: {LRModel.weights}, {LRModel.bias}")

    test_x = [[1.1,1.2], [3.23,2.99]]
    print("x: ",  test_x, "\ny_: ", LRModel.infer(test_x))
