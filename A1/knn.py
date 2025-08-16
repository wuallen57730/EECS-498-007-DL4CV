"""
Implements a K-Nearest Neighbor classifier in PyTorch.
"""
import torch
from typing import Dict, List


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from knn.py!")


def compute_distances_two_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    計算訓練集和測試集中每個元素之間的平方歐幾里得距離。圖像應被展平並視為向量。
    此實現使用了一組簡單的嵌套迴圈來遍歷訓練和測試數據。
    輸入數據可以具有任意維度——例如，此函數應能計算向量之間的最近鄰距離，
    此時輸入形狀為 (num_{train, test}, D)；
    它也應能計算圖像之間的最近鄰距離，此時輸入形狀為 (num_{train, test}, C, H, W)。
    更一般地，輸入形狀為 (num_{train, test}, D1, D2, ..., Dn)；
    在計算距離之前，應將每個形狀為 (D1, D2, ..., Dn) 的元素展平為形狀為 (D1 * D2 * ... * Dn) 的向量。
    輸入張量不應被修改。
    注意：您的實現不得使用 torch.norm、torch.dist、torch.cdist 或它們的實例方法變體
    （x.norm、x.dist、x.cdist 等）
    不得使用來自 torch.nn 或 torch.nn.functional 模組的任何函數。
    
    參數：
    x_train: 形狀為 (num_train, D1, D2, ...) 的張量
    x_test: 形狀為 (num_test, D1, D2, ...) 的張量
    
    返回值：
    dists: 形狀為 (num_train, num_test) 的張量，
    其中 dists[i, j] 表示第 i 個訓練點與第 j 個測試點之間的平方歐幾里得距離。
    其數據類型應與 x_train 相同.
    """
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function using a pair of nested loops over the    #
    # training data and the test data.                                       #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    ##########################################################################
    # Replace "pass" statement with your code
    x_train_flatten = x_train.reshape(num_train, -1)
    x_test_flatten = x_test.reshape(num_test, -1)
    for i in range(num_train):
      for j in range(num_test):
        dists[i, j] = (x_train_flatten[i] - x_test_flatten[j]).pow(2).sum()
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def compute_distances_one_loop(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation uses only a single loop over the training data.

    Similar to `compute_distances_two_loops`, this should be able to handle
    inputs with any number of dimensions. The inputs should not be modified.

    NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
    `torch.cdist`, or their instance method variants (`x.norm`, `x.dist`,
    `x.cdist`, etc.). You may not use any functions from `torch.nn` or
    `torch.nn.functional` modules.

    Args:
        x_train: Tensor of shape (num_train, D1, D2, ...)
        x_test: Tensor of shape (num_test, D1, D2, ...)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j]
            is the squared Euclidean distance between the i-th training point
            and the j-th test point. It should have the same dtype as x_train.
    """
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function using only a single loop over x_train.   #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    ##########################################################################
    # Replace "pass" statement with your code
    x_train_flatten = x_train.reshape(num_train, -1)
    x_test_flatten = x_test.reshape(num_test, -1)
    print("Shape of x_train_flatten[i] = ", x_train_flatten[0].shape)
    print()
    print("Shape of x_test_flatten = ",x_test_flatten.shape)
    print()
    for i in range(num_train):
      dists[i] = (x_train_flatten[i] - x_test_flatten).pow(2).sum(dim=1)
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation should not use any Python loops. For memory-efficiency,
    it also should not create any large intermediate tensors; in particular you
    should not create any intermediate tensors with O(num_train * num_test)
    elements.

    Similar to `compute_distances_two_loops`, this should be able to handle
    inputs with any number of dimensions. The inputs should not be modified.

    NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
    `torch.cdist`, or their instance method variants (`x.norm`, `x.dist`,
    `x.cdist`, etc.). You may not use any functions from `torch.nn` or
    `torch.nn.functional` modules.

    Args:
        x_train: Tensor of shape (num_train, C, H, W)
        x_test: Tensor of shape (num_test, C, H, W)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j] is
            the squared Euclidean distance between the i-th training point and
            the j-th test point.
    """
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function without using any explicit loops and     #
    # without creating any intermediate tensors with O(num_train * num_test) #
    # elements.                                                              #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    #                                                                        #
    # HINT: Try to formulate the Euclidean distance using two broadcast sums #
    #       and a matrix multiply.                                           #
    ##########################################################################
    # Replace "pass" statement with your code
    x_train_flatten = x_train.reshape(num_train, -1)
    x_test_flatten = x_test.reshape(num_test, -1)
    print(x_train_flatten.shape)
    print(x_test_flatten.shape)

    # (a - b)^2 = a^2 + b^2 - 2ab
    x_train_flatten_squared_sum_by_row = x_train_flatten.pow(2).sum(dim=1).unsqueeze(1)
    x_test_flatten_squared_sum_by_row = x_test_flatten.pow(2).sum(dim=1).unsqueeze(0)
    print(x_train_flatten_squared_sum_by_row.shape)
    print(x_test_flatten_squared_sum_by_row.shape)
    dists = x_train_flatten_squared_sum_by_row + x_test_flatten_squared_sum_by_row - \
            2*(x_train_flatten @ x_test_flatten.t())
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def predict_labels(dists: torch.Tensor, y_train: torch.Tensor, k: int = 1):
    """
    Given distances between all pairs of training and test samples, predict a
    label for each test sample by taking a MAJORITY VOTE among its `k` nearest
    neighbors in the training set.

    In the event of a tie, this function SHOULD return the smallest label. For
    example, if k=5 and the 5 nearest neighbors to a test example have labels
    [1, 2, 1, 2, 3] then there is a tie between 1 and 2 (each have 2 votes),
    so we should return 1 since it is the smallest label.

    This function should not modify any of its inputs.

    Args:
        dists: Tensor of shape (num_train, num_test) where dists[i, j] is the
            squared Euclidean distance between the i-th training point and the
            j-th test point.
        y_train: Tensor of shape (num_train,) giving labels for all training
            samples. Each label is an integer in the range [0, num_classes - 1]
        k: The number of nearest neighbors to use for classification.

    Returns:
        y_pred: int64 Tensor of shape (num_test,) giving predicted labels for
            the test data, where y_pred[j] is the predicted label for the j-th
            test example. Each label should be an integer in the range
            [0, num_classes - 1].
    """
    num_train, num_test = dists.shape
    y_pred = torch.zeros(num_test, dtype=torch.int64)
    ##########################################################################
    # TODO: Implement this function. You may use an explicit loop over the   #
    # test samples.                                                          #
    #                                                                        #
    # HINT: Look up the function torch.topk                                  #
    ##########################################################################
    # Replace "pass" statement with your code
    for j in range(num_test):
      # 找每一column的距離
      distances = dists[:, j]

      # 找到 k 個最小距離的索引（最近鄰居）
      _, indices = torch.topk(distances, k, largest=False)
      # 獲取 k 個鄰居的標籤
      labels = y_train[indices]
      # 使用 bincount 計算每個標籤的投票數
      counts = torch.bincount(labels)

      # 找到最大投票數
      max_count = counts.max()
        
      # 找到所有有最大投票數的標籤（候選標籤）
      candidates = (counts == max_count).nonzero(as_tuple=False).squeeze(1)
        
      # 選擇候選中最小的一個標籤（處理平局）
      y_pred[j] = candidates.min()
      # print("y_pred:", y_pred)
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return y_pred


class KnnClassifier:

    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor):
        """
        Create a new K-Nearest Neighbor classifier with the specified training
        data. In the initializer we simply memorize the provided training data.

        Args:
            x_train: Tensor of shape (num_train, C, H, W) giving training data
            y_train: int64 Tensor of shape (num_train, ) giving training labels
        """
        ######################################################################
        # TODO: Implement the initializer for this class. It should perform  #
        # no computation and simply memorize the training data in            #
        # `self.x_train` and `self.y_train`, accordingly.                    #
        ######################################################################
        # Replace "pass" statement with your code
        self.x_train = x_train
        self.y_train = y_train
        ######################################################################
        #                         END OF YOUR CODE                           #
        ######################################################################

    def predict(self, x_test: torch.Tensor, k: int = 1):
        """
        Make predictions using the classifier.

        Args:
            x_test: Tensor of shape (num_test, C, H, W) giving test samples.
            k: The number of neighbors to use for predictions.

        Returns:
            y_test_pred: Tensor of shape (num_test,) giving predicted labels
                for the test samples.
        """
        y_test_pred = None
        ######################################################################
        # TODO: Implement this method. You should use the functions you      #
        # wrote above for computing distances (use the no-loop variant) and  #
        # to predict output labels.                                          #
        ######################################################################
        # Replace "pass" statement with your code
        dists = compute_distances_no_loops(self.x_train, x_test)
        y_test_pred = predict_labels(dists, self.y_train, k)
        ######################################################################
        #                         END OF YOUR CODE                           #
        ######################################################################
        return y_test_pred

    def check_accuracy(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        k: int = 1,
        quiet: bool = False
    ):
        """
        Utility method for checking the accuracy of this classifier on test
        data. Returns the accuracy of the classifier on the test data, and
        also prints a message giving the accuracy.

        Args:
            x_test: Tensor of shape (num_test, C, H, W) giving test samples.
            y_test: int64 Tensor of shape (num_test,) giving test labels.
            k: The number of neighbors to use for prediction.
            quiet: If True, don't print a message.

        Returns:
            accuracy: Accuracy of this classifier on the test data, as a
                percent. Python float in the range [0, 100]
        """
        y_test_pred = self.predict(x_test, k=k)
        num_samples = x_test.shape[0]
        num_correct = (y_test == y_test_pred).sum().item()
        accuracy = 100.0 * num_correct / num_samples
        msg = (
            f"Got {num_correct} / {num_samples} correct; "
            f"accuracy is {accuracy:.2f}%"
        )
        if not quiet:
            print(msg)
        return accuracy


def knn_cross_validate(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_folds: int = 5,
    k_choices: List[int] = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100],
):
    """
    Perform cross-validation for `KnnClassifier`.

    Args:
        x_train: Tensor of shape (num_train, C, H, W) giving all training data.
        y_train: int64 Tensor of shape (num_train,) giving labels for training
            data.
        num_folds: Integer giving the number of folds to use.
        k_choices: List of integers giving the values of k to try.

    Returns:
        k_to_accuracies: Dictionary mapping values of k to lists, where
            k_to_accuracies[k][i] is the accuracy on the i-th fold of a
            `KnnClassifier` that uses k nearest neighbors.
    """

    # First we divide the training data into num_folds equally-sized folds.
    x_train_folds = []
    y_train_folds = []
    ##########################################################################
    # TODO: Split the training data and images into folds. After splitting,  #
    # x_train_folds and y_train_folds should be lists of length num_folds,   #
    # where y_train_folds[i] is label vector for images inx_train_folds[i].  #
    #                                                                        #
    # HINT: torch.chunk                                                      #
    ##########################################################################
    # Replace "pass" statement with your code
    x_train_folds = list(torch.chunk(x_train, num_folds))
    y_train_folds = list(torch.chunk(y_train, num_folds))
    # print("x_train_folds: ", len(x_train_folds)) # 5
    # print("y_train_folds: ", len(y_train_folds)) # 5 
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################

    # A dictionary holding the accuracies for different values of k that we
    # find when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the
    # different accuracies we found trying `KnnClassifier`s using k neighbors.
    k_to_accuracies = {}

    ##########################################################################
    # TODO: Perform cross-validation to find the best value of k. For each   #
    # value of k in k_choices, run the k-NN algorithm `num_folds` times; in  #
    # each case you'll use all but one fold as training data, and use the    #
    # last fold as a validation set. Store the accuracies for all folds and  #
    # all values in k in k_to_accuracies.                                    #
    #                                                                        #
    # HINT: torch.cat                                                        #
    ##########################################################################
    # 先遍歷不同k
    for k in k_choices:
      accuracies = []
      # 然後對固定k 遍歷每個fold
      for i in range(num_folds):
        x_train_ = None
        y_train_ = None

        for j in range(num_folds):
          if i == j:                    #當前fold是要test的fold
            x_test_ = x_train_folds[j]
            y_test_ = y_train_folds[j]
          elif x_train_ is None:        # 若非test fold，且x_train_是第一次
            x_train_ = x_train_folds[j]
            y_train_ = y_train_folds[j]
          else:                         # 合併到訓練集
            x_train_ = torch.cat((x_train_, x_train_folds[j]))
            y_train_ = torch.cat((y_train_, y_train_folds[j]))

        classifier = KnnClassifier(x_train_, y_train_)
        accuracies.append(classifier.check_accuracy(x_test_, y_test_, k))

      k_to_accuracies[k] = accuracies
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################

    return k_to_accuracies


def knn_get_best_k(k_to_accuracies: Dict[int, List]):
    """
    Select the best value for k, from the cross-validation result from
    knn_cross_validate. If there are multiple k's available, then you SHOULD
    choose the smallest k among all possible answer.

    Args:
        k_to_accuracies: Dictionary mapping values of k to lists, where
            k_to_accuracies[k][i] is the accuracy on the i-th fold of a
            `KnnClassifier` that uses k nearest neighbors.

    Returns:
        best_k: best (and smallest if there is a conflict) k value based on
            the k_to_accuracies info.
    """
    best_k = 0
    ##########################################################################
    # TODO: Use the results of cross-validation stored in k_to_accuracies to #
    # choose the value of k, and store result in `best_k`. You should choose #
    # the value of k that has the highest mean accuracy accross all folds.   #
    ##########################################################################
    # Replace "pass" statement with your code
    best_accuracy = 0.0
    for k, accuracies in k_to_accuracies.items():
      accuracy = sum(accuracies) / len(accuracies)
      if accuracy > best_accuracy:
        best_k = k
        best_accuracy = accuracy
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return best_k
