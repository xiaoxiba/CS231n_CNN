import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    
    scores = X[i].dot(W)              # 1 x C 
    #print scores.shape
    
    #normalize the f
    scores -= np.max(scores)
    
    #calculate exponatial
    scores_exp = np.exp(scores)           # 1 x C
    sum_exp = np.sum(scores_exp)  
    log_sum_exp =np.log(sum_exp)
    
    p = scores_exp / sum_exp              # 1 x C
    
    loss += np.sum(-scores[y[i]] + log_sum_exp)               
   

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
    
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    
  f = X.dot(W)    # N by C
  f_max = np.reshape(np.max(f, axis=1), (num_train, 1))   # N by 1
  prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)
  y_trueClass = np.zeros_like(prob)
  y_trueClass[range(num_train), y] = 1.0    # N by C
  loss += -np.sum(y_trueClass * np.log(prob)) / num_train + 0.5 * reg * np.sum(W * W)
  dW += -np.dot(X.T, y_trueClass - prob) / num_train + reg * W       
            
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

