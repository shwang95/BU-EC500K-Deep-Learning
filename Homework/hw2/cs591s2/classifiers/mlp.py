import numpy as np
import matplotlib.pyplot as plt

class TwoLayerMLP(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4, activation='relu'):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.activation = activation

    def loss(self, X, y=None, reg=0.0):
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        _, C = W2.shape
        N, D = X.shape

        # Compute the forward pass
        scores = None
        z1 = np.dot(X, W1) + b1  # 1st layer activation, N*H
        _,H = z1.shape

        def softplus_forward(z1):
            if np.amax(np.amax(z1))>300:
                return z1
            # using prevention on underflow made the CCR lower, so I comment it out
            # elif np.amax(np.amax(hidden))<-300:
            # 	return 0
            else:
                return np.log(1+np.exp(z1))

        def tanh_forward(z1):
            if np.amax(np.amax(z1))>300:
                return np.ones((N,H))
            else:
                return ((np.exp(z1)-np.exp(-z1))/(np.exp(z1)+np.exp(-z1)))
            
        act = self.activation
            
        if act == 'random': # for fun
            tmp = np.random.randint(6,size=1)
            if tmp == 0:
                act = 'relu'
            elif tmp == 1:
                act = 'softplus'
            elif tmp == 2:
                act = 'sigmoid'
            elif tmp == 3:
                act = 'tanh'
            elif tmp == 4:
                act = 'absrelu'
            else: act = 'leakyrelu'
                
#         print(act)

        # 1st layer nonlinearity, N*H
        if act ==  'relu':
            hidden = np.maximum(0, z1)   
        elif act ==  'softplus':
            hidden = softplus_forward(z1)
            # hidden = np.log(1+np.exp(z1))
        elif act ==  'sigmoid':
        	hidden = 1/(1+np.exp(-z1))
        elif act ==  'tanh':
            hidden = tanh_forward(z1)
#             hidden = ((np.exp(z1)-np.exp(-z1))/(np.exp(z1)+np.exp(-z1)))
        elif act ==  'absrelu':
            hidden = z1
            np.where(hidden<0, -hidden, hidden)
        elif act ==  'leakyrelu':
            hidden = z1
            np.where(hidden<0, -0.01*hidden, hidden)
        else:
            raise ValueError('Unknown activation type')

        scores = np.dot(hidden, W2) + b2  # 2nd layer activation, N*C

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        yy = np.zeros((N,C))
        for i in range(N):
            yy[i,y[i]] = 1
        y_hat = np.exp(scores)/np.exp(scores).sum(axis=1).reshape(-1,1)
        L = -np.sum(np.log(y_hat)*yy)/N
        w1regloss = np.sum(np.diag((W1.T @ W1)))
        w2regloss = np.sum(np.diag(((W2.T @ W2))))
        loss = L + (w1regloss + w2regloss)*reg/2

        pass

        # Backward pass: compute gradients
        grads = {}
        

        # output layer
        dscore = (y_hat - yy) # 5X3
        dW2 = (hidden.T @ dscore)/N + reg*(W2) # 10X3
        db2 = dscore.sum(axis=0)/N # 1X3

        def softplus_reverse(hidden,dhidden):
            if np.amax(np.amax(hidden))>300:
                return dhidden
            # using prevention on underflow made the CCR lower, so I comment it out
            # elif np.amax(np.amax(hidden))<-300:
            # 	return 0
            else:
                return ((np.exp(hidden)-1)/np.exp(hidden))*dhidden

        def tanh_reverse(hidden,dhidden):
            if np.amax(np.amax(hidden))>300:
                return np.zeros(())
            else:
                return (1-hidden**2)*dhidden

        # hidden layer
        dhidden = dscore @ W2.T # 
        C,D = (hidden*dhidden).shape 
        if act ==  'relu':
            dz1 = dhidden
            dz1[hidden <= 0] = 0
        elif act ==  'softplus':
            dz1 = softplus_reverse(hidden,dhidden)
        elif act ==  'sigmoid':
            dz1 = hidden*(1-hidden)*dhidden
        elif act ==  'tanh':
            dz1 = tanh_reverse(hidden,dhidden)
#             dz1 = (1-hidden**2)*dhidden
        elif act ==  'absrelu':
            dz1 = dhidden
            # dz1[hidden < 0] = - dz1[hidden < 0]
            np.where(dz1<0, -dz1, dz1)
            # np.where(dz1==0, 0.0000000001, dz1)
        elif act ==  'leakyrelu':
            dz1 = dhidden
            # dz1[hidden < 0] = - dz1[hidden < 0]
            np.where(dz1<0, -0.01*dz1, dz1)
            # np.where(dz1==0, 0.0000000001, dz1)
        else:
            raise ValueError('Unknown activation type')

        # first layer
        dW1 = (X.T @ dz1)/N + reg*W1 # 4X10
        db1 = dz1.sum(axis=0)/N  #1X10

        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1

        return loss, grads

    def train(self, X, y, X_val, y_val,learning_rate=1e-3, learning_rate_decay=0.95,reg=1e-5, num_epochs=10,batch_size=200, verbose=False):
        num_train = X.shape[0]
        # print(num_train)
        iterations_per_epoch = int(max(num_train / batch_size, 1))
        # print(iterations_per_epoch)
        epoch_num = 0

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        grad_magnitude_history = []
        train_acc_history = []
        val_acc_history = []

        np.random.seed(1)
        for epoch in range(num_epochs):
            # fixed permutation (within this epoch) of training data
            perm = np.random.permutation(num_train)

            # go through minibatches
            for it in range(iterations_per_epoch):
                X_batch = None
                y_batch = None

                # Create a random minibatch
                idx = perm[it*batch_size:(it+1)*batch_size]
                X_batch = X[idx, :]
                y_batch = y[idx]

                # Compute loss and gradients using the current minibatch
                loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
                loss_history.append(loss)

                # do gradient descent
                for param in self.params:
                    self.params[param] -= grads[param] * learning_rate

                # record gradient magnitude (Frobenius) for W1
                grad_magnitude_history.append(np.linalg.norm(grads['W1']))

            # Every epoch, check train and val accuracy and decay learning rate.
            # Check accuracy
            train_acc = (self.predict(X_batch) == y_batch).mean()
            val_acc = (self.predict(X_val) == y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            if verbose:
                print('Epoch %d: loss %f, train_acc %f, val_acc %f'%(
                    epoch+1, loss, train_acc, val_acc))

            # Decay learning rate
            learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'grad_magnitude_history': grad_magnitude_history, 
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        tmp_pred = np.exp(self.loss(X))/np.exp(self.loss(X)).sum(axis=1).reshape(-1,1)
        y_pred = np.argmax(tmp_pred,axis=1)

        return y_pred