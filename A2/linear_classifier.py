"""
Implements linear classifeirs in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
import statistics
from abc import abstractmethod

def hello_linear_classifier():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from linear_classifier.py!')


# Template class modules that we will use later: Do not edit/modify this class
class LinearClassifier(object):
  """ An abstarct class for the linear classifiers """
  # Note: We will re-use `LinearClassifier' in both SVM and Softmax
  def __init__(self):
    random.seed(0)
    torch.manual_seed(0)
    self.W = None

  def train(self, X_train, y_train, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    train_args = (self.loss, self.W, X_train, y_train, learning_rate, reg,
                  num_iters, batch_size, verbose)
    self.W, loss_history = train_linear_classifier(*train_args)
    return loss_history

  def predict(self, X):
    return predict_linear_classifier(self.W, X)

  @abstractmethod
  def loss(self, W, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative.
    Subclasses will override this.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
    - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an tensor of the same shape as W
    """
    raise NotImplementedError

  def _loss(self, X_batch, y_batch, reg):
    self.loss(self.W, X_batch, y_batch, reg)

  def save(self, path):
    torch.save({'W': self.W}, path)
    print("Saved in {}".format(path))

  def load(self, path):
    W_dict = torch.load(path, map_location='cpu')
    self.W = W_dict['W']
    print("load checkpoint file: {}".format(path))



class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """
  def loss(self, W, X_batch, y_batch, reg):
    return svm_loss_vectorized(W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """
  def loss(self, W, X_batch, y_batch, reg):
    return softmax_loss_vectorized(W, X_batch, y_batch, reg)



#**************************************************#
################## Section 1: SVM ##################
#**************************************************#

def svm_loss_naive(W, X, y, reg):
  """
  結構化 SVM 損失函數，Naive實現（使用迴圈）
    有D維，C個classes
    輸入：
    - W: 形狀為 (D, C) 的 PyTorch 張量，包含權重
    - X: 形狀為 (N, D) 的 PyTorch 張量，包含小批量數據
    - y: 形狀為 (N,) 的 PyTorch 張量，包含訓練標籤；y[i] = c 表示 X[i] 的標籤為 c
    - reg: 浮點數，正則化強度
    
    返回：
    - loss: PyTorch 純量，表示損失值
    - dW: 形狀與 W 相同的張量，表示損失對 W 的梯度
  """
  dW = torch.zeros_like(W) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = W.t().mv(X[i]) # 
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:  # s_j - s_{y_i} + 1 > 0
        loss += margin
        #######################################################################
        # TODO:                                                               #
        # Compute the gradient of the loss function and store it dW. (part 1) #
        # Rather than first computing the loss and then computing the         #
        # derivative, it is simple to compute the derivative at the same time #
        # that the loss is being computed.                                    #
        #######################################################################
        # Replace "pass" statement with your code
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]
        #######################################################################
        #                       END OF YOUR CODE                              #
        #######################################################################


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * torch.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it in dW. (part 2)    #
  #############################################################################
  # Replace "pass" statement with your code
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  結構化 SVM 損失函數，向量化實現。在實現對權重 W 的正則化時，請不要將正則化項乘以 1/2（即無係數）。輸入和輸出的格式與 svm_loss_naive 相同。

  輸入：
  - W: 形狀為 (D, C) 的 PyTorch 張量，包含權重。
  - X: 形狀為 (N, D) 的 PyTorch 張量，包含小批量數據。
  - y: 形狀為 (N,) 的 PyTorch 張量，包含訓練標籤；y[i] = c 表示 X[i] 的標籤為 c，其中 0 <= c < C。
  - reg: 浮點數，正則化強度。

  返回：
  一個元組，包含：
  - 損失值（loss），為 PyTorch 純量。
  - 損失對權重 W 的梯度，形狀與 W 相同。
  """
  loss = 0.0
  dW = torch.zeros_like(W) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # Replace "pass" statement with your code
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X @ W        # shape(N, C)
  # 提取每個樣本正確類別的分數，然後 .view(num_train, 1) 轉為列向量，用於broadcasting
  correct_scores = scores[torch.arange(num_train), y].view(num_train, 1)  # shape (N, 1)
  margins = scores - correct_scores + 1               # 形狀 (N, C)
  margins[torch.arange(num_train), y] = 0             # 將正確類別的 margin 設為 0
  loss = torch.sum(torch.clamp(margins, min=0)) / num_train  # 平均 hinge loss
  loss += reg * torch.sum(W * W)                      # 加入正則化項

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # Replace "pass" statement with your code
  mask = (margins > 0).double()  # 形狀 (N, C)，指示 margin > 0 的位置
  row_sum = torch.sum(mask, dim=1)  # 形狀 (N,)，每個樣本的 positive margin 數
  mask[torch.arange(num_train), y] = -row_sum  # 正確類別的梯度貢獻
  dW = (X.t() @ mask) / num_train + 2 * reg * W  # 數據梯度 + 正則化梯度
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


def sample_batch(X, y, num_train, batch_size):
  """
  Sample batch_size elements from the training data and their
  corresponding labels to use in this round of gradient descent.
  """
  X_batch = None
  y_batch = None
  #########################################################################
  # TODO: Store the data in X_batch and their corresponding labels in     #
  # y_batch; after sampling, X_batch should have shape (batch_size, dim)  #
  # and y_batch should have shape (batch_size,)                           #
  #                                                                       #
  # Hint: Use torch.randint to generate indices.                          #
  #########################################################################
  # Replace "pass" statement with your code
  indices = torch.randint(0, num_train, (batch_size,))
  X_batch = X[indices]
  y_batch = y[indices]
  #########################################################################
  #                       END OF YOUR CODE                                #
  #########################################################################
  return X_batch, y_batch


def train_linear_classifier(loss_func, W, X, y, learning_rate=1e-3,
                            reg=1e-5, num_iters=100, batch_size=200,
                            verbose=False):
  """
  Train this linear classifier using stochastic gradient descent.

  Inputs:
  - loss_func: loss function to use when training. It should take W, X, y
    and reg as input, and output a tuple of (loss, dW)
  - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
    classifier. If W is None then it will be initialized here.
  - X: A PyTorch tensor of shape (N, D) containing training data; there are N
    training samples each of dimension D.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c
    means that X[i] has label 0 <= c < C for C classes.
  - learning_rate: (float) learning rate for optimization.
  - reg: (float) regularization strength.
  - num_iters: (integer) number of steps to take when optimizing
  - batch_size: (integer) number of training examples to use at each step.
  - verbose: (boolean) If true, print progress during optimization.

  Returns: A tuple of:
  - W: The final value of the weight matrix and the end of optimization
  - loss_history: A list of Python scalars giving the values of the loss at each
    training iteration.
  """
  # assume y takes values 0...K-1 where K is number of classes
  num_train, dim = X.shape
  if W is None:
    # lazily initialize W
    num_classes = torch.max(y) + 1
    W = 0.000001 * torch.randn(dim, num_classes, device=X.device, dtype=X.dtype)
  else:
    num_classes = W.shape[1]

  # Run stochastic gradient descent to optimize W
  loss_history = []
  for it in range(num_iters):
    # TODO: implement sample_batch function
    X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

    # evaluate loss and gradient
    loss, grad = loss_func(W, X_batch, y_batch, reg)
    loss_history.append(loss.item())

    # perform parameter update
    #########################################################################
    # TODO:                                                                 #
    # Update the weights using the gradient and the learning rate.          #
    #########################################################################
    # Replace "pass" statement with your code
    W -= learning_rate * grad
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    if verbose and it % 100 == 0:
      print('iteration %d / %d: loss %f' % (it, num_iters, loss))

  return W, loss_history


def predict_linear_classifier(W, X):
  """
  Use the trained weights of this linear classifier to predict labels for
  data points.

  Inputs:
  - W: A PyTorch tensor of shape (D, C), containing weights of a model
  - X: A PyTorch tensor of shape (N, D) containing training data; there are N
    training samples each of dimension D.

  Returns:
  - y_pred: PyTorch int64 tensor of shape (N,) giving predicted labels for each
    elemment of X. Each element of y_pred should be between 0 and C - 1.
  """
  y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
  ###########################################################################
  # TODO:                                                                   #
  # Implement this method. Store the predicted labels in y_pred.            #
  ###########################################################################
  # Replace "pass" statement with your code
  y_pred = (X @ W).max(dim=1).indices
  ###########################################################################
  #                           END OF YOUR CODE                              #
  ###########################################################################
  return y_pred


def svm_get_search_params():
  """
  Return candidate hyperparameters for the SVM model. You should provide
  at least two param for each, and total grid search combinations
  should be less than 25.

  Returns:
  - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
  - regularization_strengths: regularization strengths candidates
                              e.g. [1e0, 1e1, ...]
  """

  learning_rates = []
  regularization_strengths = []

  ###########################################################################
  # TODO:   add your own hyper parameter lists.                             #
  ###########################################################################
  # Replace "pass" statement with your code
  learning_rates = [1e-3, 5e-3, 1e-2]
  regularization_strengths = [5e-2, 1e-1, 2e-1]
  ###########################################################################
  #                           END OF YOUR CODE                              #
  ###########################################################################

  return learning_rates, regularization_strengths


def test_one_param_set(cls, data_dict, lr, reg, num_iters=2000):
  """
  Train a single LinearClassifier instance and return the learned instance
  with train/val accuracy.

  Inputs:
  - cls (LinearClassifier): a newly-created LinearClassifier instance.
                            Train/Validation should perform over this instance
  - data_dict (dict): a dictionary that includes
                      ['X_train', 'y_train', 'X_val', 'y_val']
                      as the keys for training a classifier
  - lr (float): learning rate parameter for training a SVM instance.
  - reg (float): a regularization weight for training a SVM instance.
  - num_iters (int, optional): a number of iterations to train

  Returns:
  - cls (LinearClassifier): a trained LinearClassifier instances with
                            (['X_train', 'y_train'], lr, reg)
                            for num_iter times.
  - train_acc (float): training accuracy of the svm_model
  - val_acc (float): validation accuracy of the svm_model
  """
  train_acc = 0.0 # The accuracy is simply the fraction of data points
  val_acc = 0.0   # that are correctly classified.
  ###########################################################################
  # TODO:                                                                   #
  # Write code that, train a linear SVM on the training set, compute its    #
  # accuracy on the training and validation sets                            #
  #                                                                         #
  # Hint: Once you are confident that your validation code works, you       #
  # should rerun the validation code with the final value for num_iters.    #
  # Before that, please test with small num_iters first                     #
  ###########################################################################
  # Feel free to uncomment this, at the very beginning,
  # and don't forget to remove this line before submitting your final version
  # num_iters = 100

  # Replace "pass" statement with your code
  X_train = data_dict['X_train']
  y_train = data_dict['y_train']
  X_val = data_dict['X_val']
  y_val = data_dict['y_val']

  loss_hist = cls.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=num_iters)
  y_train_pred = cls.predict(X_train)
  y_val_pred = cls.predict(X_val)
  train_acc = 100.0 * (y_train == y_train_pred).double().mean().item()
  val_acc = 100.0 * (y_val == y_val_pred).double().mean().item()
  ############################################################################
  #                            END OF YOUR CODE                              #
  ############################################################################

  return cls, train_acc, val_acc



#**************************************************#
################ Section 2: Softmax ################
#**************************************************#

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops).  When you implment
  the regularization over W, please DO NOT multiply the regularization term by
  1/2 (no coefficient).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A PyTorch tensor of shape (D, C) containing weights.
  - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an tensor of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = torch.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability (Check Numeric Stability #
  # in http://cs231n.github.io/linear-classify/). Plus, don't forget the      #
  # regularization!                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
    scores = W.t().mv(X[i])  # 計算 logits: f = W^T x_i (形狀: (C,))
    scores -= scores.max()   # 移位以確保數值穩定: f' = f - \max(f) (避免 exp() 溢位)
                             # 數學: 這不改變 softmax 結果，因為 softmax 是歸一化的，但防止數值問題
    scores = torch.exp(scores)      # 計算 exp(f'): \exp(f_j') (形狀: (C,))
    scores = scores / scores.sum()  # 計算 softmax 機率: p_j = \exp(f_j') / \sum_m \exp(f_m')
    loss += -torch.log(scores[y[i]])  # 累加單樣本損失: L_i = - \log(p_{y_i})
    
    e_y = torch.zeros_like(scores)  # 創建 one-hot 向量: e_y (形狀: (C,))
    e_y[y[i]] = 1                   # e_y[y_i] = 1，其他為 0
    dW += X[i].unsqueeze(dim=-1) @ (scores - e_y).unsqueeze(dim=0)
                                    # 計算單樣本梯度貢獻: \partial L_i / \partial W = x_i (p - e_y)^T
                                    # - X[i].unsqueeze(dim=-1): 將 x_i (D,) 轉為 (D,1)
                                    # - (scores - e_y).unsqueeze(dim=0): 將 (p - e_y) (C,) 轉為 (1,C)
                                    # \partial L_i / \partial f = p - e_y (對 logits 的梯度)
                                    # 由於 f = W^T x_i，所以 \partial L_i / \partial W = x_i (\partial L_i / \partial f)^T
                                    # (注意轉置，因為 W 是 (D,C)，梯度形狀相同)

  loss /= num_train   # 平均損失: L = (1/N) \sum L_i
                      
  loss += reg * torch.sum(W * W)  # 加入 L2 正則化: reg * ||W||^2_F
                                  # 數學: ||W||^2_F = \sum_{d,c} W_{d,c}^2 (Frobenius 範數平方)。
  dW /= num_train  # 平均梯度: (1/N) \sum \partial L_i / \partial W
                  # 數學: 批次平均。
  dW += 2 * reg * W  # 加入正則化梯度: 2 reg W
                      # 數學: \partial (reg ||W||^2) / \partial W = 2 reg W (L2 正則化的梯度)。
                      # 注意: 因為 ||W||^2 的導數是 2W，且無 1/2 係數。
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.  When you implment the
  regularization over W, please DO NOT multiply the regularization term by 1/2
  (no coefficient).

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = torch.zeros_like(W)  # shape(D, C)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability (Check Numeric Stability #
  # in http://cs231n.github.io/linear-classify/). Don't forget the            #
  # regularization!                                                           #
  #############################################################################
  # Replace "pass" statement with your code
  num_train = X.shape[0]

  scores = X @ W  #計算 f = W^T * x_i
  scores -= scores.max(dim=1, keepdim=True).values
  scores = torch.exp(scores)
  scores = scores/scores.sum(dim=1, keepdim=True)

  # One-hot encoding
  e_y = torch.zeros_like(scores)
  e_y[torch.arange(num_train), y] = 1

  dW = ( X.t() @ (scores - e_y) )/num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_get_search_params():
  """
  Return candidate hyperparameters for the Softmax model. You should provide
  at least two param for each, and total grid search combinations
  should be less than 25.

  Returns:
  - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
  - regularization_strengths: regularization strengths candidates
                              e.g. [1e0, 1e1, ...]
  """
  learning_rates = []
  regularization_strengths = []

  ###########################################################################
  # TODO: Add your own hyper parameter lists. This should be similar to the #
  # hyperparameters that you used for the SVM, but you may need to select   #
  # different hyperparameters to achieve good performance with the softmax  #
  # classifier.                                                             #
  ###########################################################################
  # Replace "pass" statement with your code
  pass
  ###########################################################################
  #                           END OF YOUR CODE                              #
  ###########################################################################

  return learning_rates, regularization_strengths
