"""
Implements a two-layer Neural Network classifier in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
from math import e
import torch
import random
import statistics
from linear_classifier import sample_batch


def hello_two_layer_net():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from two_layer_net.py!')


# Template class modules that we will use later: Do not edit/modify this class
class TwoLayerNet(object):
  def __init__(self, input_size, hidden_size, output_size,
               dtype=torch.float32, device='cuda', std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    - dtype: Optional, data type of each initial weight params
    - device: Optional, whether the weight params is on GPU or CPU
    - std: Optional, initial weight scaler.
    """
    # reset seed before start
    random.seed(0)
    torch.manual_seed(0)

    self.params = {}
    self.params['W1'] = std * torch.randn(input_size, hidden_size, dtype=dtype, device=device)
    self.params['b1'] = torch.zeros(hidden_size, dtype=dtype, device=device)
    self.params['W2'] = std * torch.randn(hidden_size, output_size, dtype=dtype, device=device)
    self.params['b2'] = torch.zeros(output_size, dtype=dtype, device=device)

  def loss(self, X, y=None, reg=0.0):
    return nn_forward_backward(self.params, X, y, reg)

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    return nn_train(
            self.params,
            nn_forward_backward,
            nn_predict,
            X, y, X_val, y_val,
            learning_rate, learning_rate_decay,
            reg, num_iters, batch_size, verbose)

  def predict(self, X):
    return nn_predict(self.params, nn_forward_backward, X)

  def save(self, path):
    torch.save(self.params, path)
    print("Saved in {}".format(path))

  def load(self, path):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint
    print("load checkpoint file: {}".format(path))



def nn_forward_pass(params, X):
    """
    我們的神經網絡實現的第一階段：執行網絡的向前傳遞，計算隱藏層特徵和分類分數
    網絡架構應為： 全連接層 -> ReLU（隱藏層） -> 全連接層（分數）
    作為練習，這次我們不允許使用 torch.relu 和 torch.nn 操作（從作業3開始可以使用）。
    - 輸入：
    params：一個儲存模型權重的 PyTorch 張量字典，應包含以下鍵和形狀：
    W1：第一層weight；形狀為 (D, H)
    b1：第一層bias；形狀為 (H,)
    W2：第二層weight；形狀為 (H, C)
    b2：第二層bias；形狀為 (C,)
    X：形狀為 (N, D) 的輸入數據。每個 X[i] 是一個訓練樣本。

    返回一個tuple，包含：

    scores：形狀為 (N, C) 的張量，表示 X 的分類分數
    hidden：形狀為 (N, H) 的張量，表示每個輸入值經過 ReLU 後的隱藏層表示
    """
    # Unpack variables from the params dictionary
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    N, D = X.shape

    # Compute the forward pass
    hidden = None
    scores = None
    ############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input.#
    # Store the result in the scores variable, which should be an tensor of    #
    # shape (N, C).                                                            #
    ############################################################################
    # Replace "pass" statement with your code
    Z1 = X @ W1 + b1

    hidden = Z1.clamp(min=0)  # 使用 clamp 實現 ReLU，替代 torch.relu

    scores = hidden @ W2 + b2  # 形狀: (N, C)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return scores, hidden


def nn_forward_backward(params, X, y = None, reg=0.0):
    """
    計算雙層全連接神經網絡的損失和梯度。在實現損失和梯度時，請不要忘記按批次大小對損失/梯度進行縮放。
    輸入：前兩個參數（params, X）與 nn_forward_pass 相同

    params：一個儲存模型權重的 PyTorch 張量字典，應包含以下鍵和形狀：
    - W1：第一層權重；形狀為 (D, H)
    - b1：第一層偏置；形狀為 (H,)
    - W2：第二層權重；形狀為 (H, C)
    - b2：第二層偏置；形狀為 (C,)
    - X：形狀為 (N, D) 的輸入數據。每個 X[i] 是一個訓練樣本
    - y：訓練標籤向量。y[i] 是 X[i] 的標籤，每個 y[i] 是範圍在 0 <= y[i] < C 的整數
      此參數是可選的；如果未傳入，則僅返回分數；如果傳入，則返回損失和梯度
    - reg：正則化強度

    返回：
    如果 y 為 None，返回形狀為 (N, C) 的張量 scores，其中 scores[i, c] 是輸入 X[i] 在類別 c 上的分數。
    如果 y 不為 None，則返回一個元組，包含：

    - loss：此批次訓練樣本的損失（數據損失和正則化損失）。
    - grads：將參數名稱映射到相對於損失函數的參數梯度的字典，與 self.params 具有相同的鍵。
    """
    # Unpack variables from the params dictionary
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    N, D = X.shape

    scores, h1 = nn_forward_pass(params, X)
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    ############################################################################
    # TODO: 根據 nn_forward_pass 的結果計算損失。這應包括數據損失和 W1 與 W2 的 L2 正則#
    # 化。將結果儲存在變數 loss 中，loss 應為一個標量。使用 Softmax 分類器損失          #
    # 在實現 W 的正則化時，請不要將正則化項乘以 1/2(無係數)                           #
    # 如果不小心，這裡很容易遇到數值不穩定問題                                        #
    #（請參考 http://cs231n.github.io/linear-classify/ 中的數值穩定性）。           #
    ############################################################################
    # Replace "pass" statement with your code
    num_train = X.shape[0]
    scores -= scores.max(dim=1, keepdim=True).values
    scores = torch.exp(scores)
    scores = scores/scores.sum(dim=1, keepdim=True)

    loss = (-torch.log(scores[torch.arange(num_train), y])).mean()

    loss += reg * torch.sum(W1*W1)
    loss += reg * torch.sum(W2*W2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    # Backward pass: compute gradients
    grads = {}
    ###########################################################################
    # TODO: Compute the backward pass, computing the derivatives of the       #
    # weights and biases. Store the results in the grads dictionary.          #
    # For example, grads['W1'] should store the gradient on W1, and be a      #
    # tensor of same size                                                     #
    ###########################################################################
    # 計算backward pass
    
    e_y = torch.zeros_like(scores)
    e_y[torch.arange(num_train), y] = 1
    # \gradient L_i / \grafdient S_i = P_i - e_{y_i}
    dscores = scores - e_y

    grads['b2'] = dscores.sum(dim=0)/num_train
    grads['W2'] = (h1.t() @ dscores) / num_train + 2 * reg * W2

    dh1 = dscores @ W2.t()
    drelu = (h1 > 0).to(X.dtype)

    grads['b1'] = (dh1 * drelu).sum(dim=0) / num_train
    grads['W1'] = (X.t() @ (dh1 * drelu)) / num_train + 2 * reg * W1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return loss, grads


def nn_train(params, loss_func, pred_func, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
  """
  Train this neural network using stochastic gradient descent.

  Inputs:
  - params: a dictionary of PyTorch Tensor that store the weights of a model.
    It should have following keys with shape
        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)
  - loss_func: a loss function that computes the loss and the gradients.
    It takes as input:
    - params: Same as input to nn_train
    - X_batch: A minibatch of inputs of shape (B, D)
    - y_batch: Ground-truth labels for X_batch
    - reg: Same as input to nn_train
    And it returns a tuple of:
      - loss: Scalar giving the loss on the minibatch
      - grads: Dictionary mapping parameter names to gradients of the loss with
        respect to the corresponding parameter.
  - pred_func: prediction function that im
  - X: A PyTorch tensor of shape (N, D) giving training data.
  - y: A PyTorch tensor f shape (N,) giving training labels; y[i] = c means that
    X[i] has label c, where 0 <= c < C.
  - X_val: A PyTorch tensor of shape (N_val, D) giving validation data.
  - y_val: A PyTorch tensor of shape (N_val,) giving validation labels.
  - learning_rate: Scalar giving learning rate for optimization.
  - learning_rate_decay: Scalar giving factor used to decay the learning rate
    after each epoch.
  - reg: Scalar giving regularization strength.
  - num_iters: Number of steps to take when optimizing.
  - batch_size: Number of training examples to use per step.
  - verbose: boolean; if true print progress during optimization.

  Returns: A dictionary giving statistics about the training process
  """
  num_train = X.shape[0]
  iterations_per_epoch = max(num_train // batch_size, 1)

  # Use SGD to optimize the parameters in self.model
  loss_history = []
  train_acc_history = []
  val_acc_history = []

  for it in range(num_iters):
    X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

    # Compute loss and gradients using the current minibatch
    loss, grads = loss_func(params, X_batch, y=y_batch, reg=reg)
    loss_history.append(loss.item())

    #########################################################################
    # TODO: Use the gradients in the grads dictionary to update the         #
    # parameters of the network (stored in the dictionary self.params)      #
    # using stochastic gradient descent. You'll need to use the gradients   #
    # stored in the grads dictionary defined above.                         #
    #########################################################################
    # Replace "pass" statement with your code
    for param_name, grad in grads.items():
          param = params[param_name]
          param -= learning_rate * grad
    #########################################################################
    #                             END OF YOUR CODE                          #
    #########################################################################

    if verbose and it % 100 == 0:
      print('iteration %d / %d: loss %f' % (it, num_iters, loss.item()))

    # Every epoch, check train and val accuracy and decay learning rate.
    if it % iterations_per_epoch == 0:
      # Check accuracy
      y_train_pred = pred_func(params, loss_func, X_batch)
      train_acc = (y_train_pred == y_batch).float().mean().item()
      y_val_pred = pred_func(params, loss_func, X_val)
      val_acc = (y_val_pred == y_val).float().mean().item()
      train_acc_history.append(train_acc)
      val_acc_history.append(val_acc)

      # Decay learning rate
      learning_rate *= learning_rate_decay

  return {
    'loss_history': loss_history,
    'train_acc_history': train_acc_history,
    'val_acc_history': val_acc_history,
  }


def nn_predict(params, loss_func, X):
  """
  使用這個雙層網絡的訓練權重來預測數據點的標籤。對於每個數據點，我們預測 C 個類別的得分，並將每個數據點分配到得分最高的類別。
  輸入：
  params：一個儲存模型權重的 PyTorch Tensor 字典，應包含以下鍵和形狀：
  - W1：第一層權重，形狀為 (D, H)
  - b1：第一層偏置，形狀為 (H,)
  - W2：第二層權重，形狀為 (H, C)
  - b2：第二層偏置，形狀為 (C,)
  - loss_func：一個計算損失和梯度的損失函數
  - X：一個形狀為 (N, D) 的 PyTorch Tensor，表示 N 個 D 維數據點進行分類。

  return：
  - y_pred：一個形狀為 (N,) 的 PyTorch Tensor，表示 X 中每個元素的預測標籤
  對於所有 i，y_pred[i] = c 表示 X[i] 被預測為類別 c，其中 0 <= c < C
  """
  y_pred = None

  ###########################################################################
  # TODO: Implement this function; it should be VERY simple!                #
  ###########################################################################
  # Replace "pass" statement with your code
  scores = loss_func(params, X)
  y_pred = scores.argmax(dim=1)
  ###########################################################################
  #                              END OF YOUR CODE                           #
  ###########################################################################

  return y_pred



def nn_get_search_params():
  """
  Return candidate hyperparameters for a TwoLayerNet model.
  You should provide at least two param for each, and total grid search
  combinations should be less than 256. If not, it will take
  too much time to train on such hyperparameter combinations.

  Returns:
  - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
  - hidden_sizes: hidden value sizes, e.g. [8, 16, ...]
  - regularization_strengths: regularization strengths candidates
                              e.g. [1e0, 1e1, ...]
  - learning_rate_decays: learning rate decay candidates
                              e.g. [1.0, 0.95, ...]
  """
  learning_rates = []
  hidden_sizes = []
  regularization_strengths = []
  learning_rate_decays = []
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

  return learning_rates, hidden_sizes, regularization_strengths, learning_rate_decays


def find_best_net(data_dict, get_param_set_fn):
  """
  Tune hyperparameters using the validation set.
  Store your best trained TwoLayerNet model in best_net, with the return value
  of ".train()" operation in best_stat and the validation accuracy of the
  trained best model in best_val_acc. Your hyperparameters should be received
  from in nn_get_search_params

  Inputs:
  - data_dict (dict): a dictionary that includes
                      ['X_train', 'y_train', 'X_val', 'y_val']
                      as the keys for training a classifier
  - get_param_set_fn (function): A function that provides the hyperparameters
                                 (e.g., nn_get_search_params)
                                 that gives (learning_rates, hidden_sizes,
                                 regularization_strengths, learning_rate_decays)
                                 You should get hyperparameters from
                                 get_param_set_fn.

  Returns:
  - best_net (instance): a trained TwoLayerNet instances with
                         (['X_train', 'y_train'], batch_size, learning_rate,
                         learning_rate_decay, reg)
                         for num_iter times.
  - best_stat (dict): return value of "best_net.train()" operation
  - best_val_acc (float): validation accuracy of the best_net
  """

  best_net = None
  best_stat = None
  best_val_acc = 0.0

  #############################################################################
  # TODO: Tune hyperparameters using the validation set. Store your best      #
  # trained model in best_net.                                                #
  #                                                                           #
  # To help debug your network, it may help to use visualizations similar to  #
  # the ones we used above; these visualizations will have significant        #
  # qualitative differences from the ones we saw above for the poorly tuned   #
  # network.                                                                  #
  #                                                                           #
  # Tweaking hyperparameters by hand can be fun, but you might find it useful #
  # to write code to sweep through possible combinations of hyperparameters   #
  # automatically like we did on the previous exercises.                      #
  #############################################################################
  # Replace "pass" statement with your code
  pass
  #############################################################################
  #                               END OF YOUR CODE                            #
  #############################################################################

  return best_net, best_stat, best_val_acc
