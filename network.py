from typing import List
import numpy as np
from layer import Layer
from utils import *


class Network():
    def __init__(self, a_in: np.array, W1, b1, W2, b2, W3, b3, iterations, alpha):
        self.a_in = a_in
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.W3 = W3
        self.b3 = b3
        self.iterations = iterations
        self.alpha = alpha

    # def init_params(self):
    #     W1 = np.random.randn(3, 3)
    #     b1 = np.random.randn(3, 1) # zᴸ = Wᴸ * Aᴸ⁻¹ + bᴸ -> (3, 2)*(2, 1) = (3, 1)
    #     W2 = np.random.randn(3, 3) # batchsize=col=3 -> 3 features, 3 training samples
    #     b2 = np.random.randn(3, 1) # one bias term applied for every node (3 features 1 sample overall -> batch size = 1)
    #     W3 = np.random.randn(3, 1)
    #     b3 = np.random.rand(1, 1)

    #     return (W1, b1, W2, b2, W3, b3)
    
    def print_params(self):
        print("W1", self.W1)
        print("b1", self.b1)
        print("W2", self.W2)
        print("b2", self.b2)
        print("W3", self.W3)
        print("b3", self.b3)

    
    # FORWARD A0 --W1--> A1 --W2--> A2 --W3--> A3

    def forward_prop(self): # feed forward
        units = self.W1.shape[1] # batch size = m = num training samples
        a_out_1 = np.zeros(units)
        a_out_2 = np.zeros(units)
        a_out_3 = np.zeros(units)

        z_1 = np.dot(self.W1, self.a_in.T) + self.b1
        a_out_1 =  sigmoid(z_1)

        # w_2 = W2[:, j]
        z_2 = np.dot(self.W2, a_out_1) + self.b2
        a_out_2 = sigmoid(z_2)

        # w_3 = W3[:, j]
        z_3 = np.dot(self.W3.T, a_out_2) + self.b3
        a_out_3 = sigmoid(z_3)
        #a_out_3 = np.expand_dims(a_out_3, axis=0)

        cache = {
            "A0": np.array(self.a_in, dtype=np.float32),
            "A1": np.array(a_out_1, dtype=np.float32),
            "A2": np.array(a_out_2, dtype=np.float32),
            "A3": np.array(a_out_3, dtype=np.float32)
        }

        # print("Activated output in forward prop:", a_out_3.shape)

        return a_out_3, cache

    #  BACKPROP A0 (input layer - no backprop) <- dC/dA2 <- A1 <- dC/dA2 <- A2 <- dC/dw3 <- A3 (Cost)

    def backprop_l3(self, y_hat, Y, m, A2, W3):
        A3 = y_hat
        pos_weight = 10
        dC_dz3 = (1 / m) * (A3 - Y) # dC_dZ3 = dC_dAL * dAL_dZL
        dz3_dw3 = A2
        dC_dw3 = np.dot(dC_dz3, dz3_dw3.T)

        dz3_db3 = 1

        # bias in layer 3 
        dC_db3 = np.sum(dC_dz3) # dz3_db3 is 1 anyways because dC_db3 = dC_dz3 * dC_db3

        dz3_dA2 = W3
        dC_dA2 = np.dot(W3, dC_dz3)

        return dC_dw3, dC_db3, dC_dA2
    

    def backprop_l2(self, dC_dA2, A1, A2, W2):
        
        dA2_dz2 = A2 * (1 - A2) # coming from layer after (backprop) -> this is the derivative of the sigmoid acti.vation applied on layer 2
        dC_dz2 = dC_dA2 * dA2_dz2 # this chains the gradient from the output layer (dC_dA2) to the sigmoid activation derivative through this hidden layer
        
        dz2_dw2 = A1
        dC_dw2 = np.dot(dC_dz2, dz2_dw2.T)

        dz2_db2 = 1

        # bias in layer 3 
        dC_db2 = np.sum(dC_dz2) 

        dz2_dA1 = W2
        dC_dA1 = np.dot(W2.T, dC_dz2)

        return dC_dw2, dC_db2, dC_dA1
    
    def backprop_l1(self, dC_dA1, A0, A1, W1): # W1 is not needed as it isnt used to backprop to input layer A0 cuz backprop is not applied to input layer

        dA1_dz1 = A1 * (1 - A1)
        dC_dz1 = dC_dA1 * dA1_dz1

        dz1_dw1 = A0
        dC_dw1 = np.dot(dC_dz1, dz1_dw1)
        dz1_db1 = 1
        dC_db1 = np.sum(dC_dz1)

        # dz1_dA0 = W1
        # dC_dA0 = np.dot(W1.T, dC_dz1)  # not needed as A0 is just the input layer and backprop is not applied to the input layer (again mentioned)

        return dC_dw1, dC_db1 # , dC_dA0
    
    def gradient_descent(self, dC_dw3, dC_db3, dC_dw2, dC_db2, dC_dw1, dC_db1):

        # w = w - alpha * dC/dw
        # b = b - alpha * dC/db 
        # print("shape of weights before gradient descent")
        # print(self.W3.shape)
        # print(self.W2.shape)
        # print(self.W1.shape)

        self.W3 = self.W3 - (self.alpha * dC_dw3)
        self.b3 = self.b3 - (self.alpha * dC_db3)

        self.W2 = self.W2 - (self.alpha * dC_dw2)
        self.b2 = self.b2 - (self.alpha * dC_db2)
        
        self.W1 = self.W1 - (self.alpha * dC_dw1)
        self.b1 = self.b1 - (self.alpha * dC_db1)

        # print("shape of weights after gradient descent")
        # print(self.W3.shape)
        # print(self.W2.shape)
        # print(self.W1.shape)

        return self.W1, self.b1, self.W2, self.b2, self.W3, self.b3
    
    def calc_accuracy(self, y_hat, y_true, threshold=0.5):
        preds = (y_hat >= threshold).astype(int)
        
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1

    def train(self, Y):
        Y = Y.reshape(1, -1)
        min_error = math.inf

        # print("Y shape:", Y.shape)
        m = Y.shape[1]
        costs = []
        optimal_wb_dict = {"W1": None, "b1": None, "W2": None, "b2": None, "W3": None, "b3": None}
        final_wb_dict = {"W1": None, "b1": None, "W2": None, "b2": None, "W3": None, "b3": None}
        for epoch in range(self.iterations):
            y_hat, cache = self.forward_prop()
            # print("DEBUG y_hat shape before loss:", y_hat.shape)
            error = cost(y_hat, Y)
            costs.append(error)

            # backprop
            dC_dw3, dC_db3, dC_dA2 = self.backprop_l3(y_hat, Y, m, cache["A2"], self.W3)
            dC_dw2, dC_db2, dC_dA1 = self.backprop_l2(dC_dA2, cache["A1"], cache["A2"], self.W2)
            dC_dw1, dC_db1 = self.backprop_l1(dC_dA1, cache["A0"], cache["A1"], self.W1)

            gW1, gb1, gW2, gb2, gW3, gb3, = self.gradient_descent(dC_dw3, dC_db3, dC_dw2, dC_db2, dC_dw1, dC_db1)
            # print("W3 mean:", np.mean(self.W3))
            # print("W2 mean:", np.mean(self.W2))
            # print("W1 mean:", np.mean(self.W1))
            if error < min_error:
                min_error = error
                optimal_wb_dict["W1"] = gW1
                optimal_wb_dict["b1"] = gb1
                optimal_wb_dict["W2"] = gW2
                optimal_wb_dict["b2"] = gb2
                optimal_wb_dict["W3"] = gW3
                optimal_wb_dict["b3"] = gb3

            final_wb_dict["W1"] = gW1
            final_wb_dict["b1"] = gb1
            final_wb_dict["W2"] = gW2
            final_wb_dict["b2"] = gb2
            final_wb_dict["W3"] = gW3
            final_wb_dict["b3"] = gb3

            # print("yhat", y_hat.shape)
            # print("y", Y.shape)

            # print("y_hat shape:", y_hat.shape)
            # print("y_hat:", y_hat)
            
            if epoch == 0 or epoch % 10 == 0:
                print(f"epoch {epoch+1}: cost = {error:4f} | accuracy = {self.calc_accuracy(y_hat, Y)}")
                # pass
            
        print("min:", np.min(y_hat))
        print("max:", np.max(y_hat))
        print("mean:", np.mean(y_hat))
    
        return costs, min_error, optimal_wb_dict, final_wb_dict
        
    




if __name__ == "__main__":

    X, y = feature_conv()
    # print("Counts:", np.unique(y, return_counts=True))
    # print("feature and output shapes:")
    # print(X.shape) # (4094, 3) -> 4094 rows, 3 cols for the features
    # print(y.shape) # (4904, 1) -> 4094 rows, 1 col for the y_true

    W1 = np.random.randn(3, 3) * np.sqrt(1 / 3)
    b1 = np.random.randn(3, 1) # zᴸ = Wᴸ * Aᴸ⁻¹ + bᴸ -> (3, 2)*(2, 1) = (3, 1)
    W2 = np.random.randn(3, 3) * np.sqrt(1 / 3) # batchsize=col=3 -> 3 features, 3 training samples
    b2 = np.random.randn(3, 1) # one bias term applied for every node (3 features 1 sample overall -> batch size = 1)
    W3 = np.random.randn(3, 1) * np.sqrt(1 / 3) #  output node weight
    b3 = np.random.rand(1, 1) # output node bias

# input -> (4094, 3)
# W1 -> (3, 3), b1 -> (3, 1)
# z1 = (3, 3) * (4094, 3)^T + (3, 1) = (3, 4094) 


# W2 -> (3, 3), b2 -> (3, 1)
# z2 = (3, 3) * (3, 4094) + (3, 1) = (3, 4094)

# W3 -> (3, 1), b3 -> (1, 1)
# z3 = (3, 1)^T * (3, 4094) + (1, 1) = (1, 4094)^T = (4094, 1) -> y_pred (4094 rows, 1 col)

    iterations = 100000
    alpha = 0.01

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # Oversample positives to match negatives
    pos_oversampled = np.random.choice(pos_idx, size=len(neg_idx), replace=True)
    all_idx = np.concatenate([neg_idx, pos_oversampled])
    np.random.shuffle(all_idx)

    X_balanced = X[all_idx]
    y_balanced = y[all_idx]

    network = Network(np.array(X_balanced), W1, b1, W2, b2, W3, b3, iterations, alpha)
    # print(network.init_params())
    print(network.print_params())

    costs, min_error, optimal_wb_dict, final_wb_dict = network.train(y_balanced)
    # print("Costs:", costs, "\n")
    print(f"Min error found: {min_error} with optimal params {optimal_wb_dict}")
    for k, v in final_wb_dict.items():
        print(f"weight: {k} -> value: \n{v}\n")