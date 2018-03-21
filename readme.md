## CS231n Assignment #1

详情见 [Assignment #1][http://cs231n.github.io/assignments2017/assignment1/] 

### 环境配置

推荐使用 Anaconda 新建环境，使用 `pip3 install -r requirements.txt` 进行环境配置。

一些可能出现的错误：

- 在 jupyter notebook 中启用 conda 环境，使用如下命令可看到切换内核选项：

  ```
  conda install nb_conda
  conda install ipykernel
  ```

- ModuleNotFoundError: No module named 'past'

  ```
  pip3 install future
  ```

- TypeError: slice indices must be integers or None or have an __index__ method

  将报错的位置下标运算中的浮点除法 '/' 改成整除 '//' 即可。


### KNN

KNN 部分比较简单，主要内容是向量化编程以及熟悉 numpy，对 MATLAB 有了解的话上手会很快。除此之外还涉及到交叉验证等基础知识。

##### 首先实现使用双层循环计算距离矩阵

```python
  def compute_distances_two_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        dists[i][j] = np.sqrt(np.sum((X[i] - self.X_train[j])**2))
    return dists
```

##### 实现 label 的预测

```python
  def predict_labels(self, dists, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      closest_y = []
      for ind in np.argsort(dists[i])[:k]:
            closest_y.append(self.y_train[ind])
      y_pred[i] = np.argmax(np.bincount(closest_y))
    return y_pred
```

这里用到了 [np.argsort][https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.argsort.html]，获取前 k 个最接近的训练样本的 label，以及 [np.bincount][https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.bincount.html]，获取出现次数最多的 label。

##### 使用向量化减少一层循环

```python
  def compute_distances_one_loop(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      dists[i] = np.sqrt(np.sum((self.X_train - X[i])**2, axis=1))
    return dists
```

##### 使用向量化去除所有循环

```python
  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    testSqua = np.sum(X**2, axis=1)
    trainSqua = np.transpose(np.sum(self.X_train**2, axis=1))
    crossPoly = np.dot(X, np.transpose(self.X_train))
    dists = np.sqrt(testSqua[:,np.newaxis] + trainSqua[np.newaxis,:] - 2 * crossPoly)
    return dists
```

这里主要用到的是多项式的展开，三者比较说明向量化还是很强的

```
Two loop version took 30.033519 seconds
One loop version took 32.466723 seconds
No loop version took 0.308525 seconds
```

##### 交叉验证

```Python
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}

for k in k_choices:
    k_to_accuracies[k] = []
for i in range(num_folds):
    X_train_KFold = np.array([]).reshape(0, 3072)
    y_train_KFold = np.array([])
    for j in range(num_folds):
        if j != i:
            X_train_KFold = np.concatenate((X_train_KFold, X_train_folds[j]))
            y_train_KFold = np.concatenate((y_train_KFold, y_train_folds[j]))
    X_test_KFold = X_train_folds[i]
    y_test_KFold = y_train_folds[i]
    classifierKFold = KNearestNeighbor()
    classifierKFold.train(X_train_KFold, y_train_KFold)
    dists = classifierKFold.compute_distances_no_loops(X_test_KFold)
    for k in k_choices:
        y_test_pred_KFold = classifierKFold.predict_labels(dists, k=k)
        num_correct = np.sum(y_test_pred_KFold == y_test_KFold)
        accuracy = float(num_correct) / len(y_test_KFold)
        k_to_accuracies[k].append(accuracy)

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
```

这里用到 np.concatenate 实际上很不优雅，如果有更好的方法欢迎告知。

### SVM 的训练

SVM 这部分开始进行梯度的计算，需要一丢丢数学，向量化的部分容易出错。

##### 利用循环进行 loss 和梯度的计算

```Python
def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW
```

注意到我们计算的梯度是 loss 对权重 W 的梯度，因此与 loss 应该是同步的。在纯循环的代码里很容易看出，loss 的变化会有相应的梯度的变化。平均化和正则项比较容易遗漏。

##### 向量化计算 loss 和梯度

```Python
def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = np.dot(X, W)
  correct_class_score = scores[(np.arange(num_train), y)]
  margin = scores - correct_class_score[:,np.newaxis] + 1
  mask = margin > 0
  loss += np.sum(margin[mask])
  loss -= num_train
  loss /= num_train
  loss += reg * np.sum(W * W)

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

  res = np.zeros(mask.shape)
  res[mask] = 1.0
  row_sum = np.sum(res, axis=1)
  res[np.arange(num_train), y] -= row_sum
  dW = np.dot(X.T, res) / num_train + reg * 2 * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
```

向量化的操作与循环基本上是对应的，用到了 np.newaxis 等矩阵操作。

计算出梯度以后进行 SGD 就比较简单了。

##### 使用验证集进行调参

```python
learning_rates = [1e-7, 1e-6, 8e-8, 5e-8]
regularization_strengths = [2.5e4, 1e4, 1.5e4]

# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results = {}
best_val = -1   # The highest validation accuracy that we have seen so far.
best_svm = None # The LinearSVM object that achieved the highest validation rate.

for lr in learning_rates:
    for reg in regularization_strengths:
        svm = LinearSVM()
        tic = time.time()
        svm.train(X_train, y_train, learning_rate=lr, reg=reg,num_iters=1500, 
                  verbose=True)
        y_train_pred = svm.predict(X_train)
        y_val_pred = svm.predict(X_val)
        train_acc = np.mean(y_train == y_train_pred)
        val_acc = np.mean(y_val == y_val_pred)
        if val_acc > best_val:
            best_val = val_acc
            best_svm = svm
        results[(lr, reg)] = (train_acc, val_acc)
# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))
    
print('best validation accuracy achieved during cross-validation: %f' % best_val)
```

该部分属于看脸的玄学调参，我这边最好有 39.4% 左右。

最后的可视化结果说实话我啥也看不出来……

### Softmax 分类器

##### 使用循环计算 loss 和梯度

```Python
def softmax_loss_naive(W, X, y, reg):
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
      scores = np.dot(X[i], W)
      scores = np.exp(scores)
      scores = scores / np.sum(scores)
      loss -= np.log(scores[y[i]])
      for j in xrange(num_classes):
        dW[:,j] += X[i] * scores[j]
      dW[:,y[i]] -= X[i]

  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2.0 * reg * W

  return loss, dW
```

需要一些数学推导，这个计算过程能很好的体现出 Softmax 梯度容易计算的特点。

##### 向量化计算 loss 和梯度

```Python
def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = np.dot(X, W)
  scores = np.exp(scores)
  scores = scores / np.sum(scores, axis=1)[:, None]
  
  loss -= np.sum(np.log(scores[(np.arange(num_train), y)]))
  scores[(np.arange(num_train), y)] -= 1
  dW += np.dot(X.T, scores)

  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2.0 * reg * W

  return loss, dW
```

速度比较结果如下：

```
naive loss: 2.330105e+00 computed in 0.076748s
vectorized loss: 2.330105e+00 computed in 0.003300s
Loss difference: 0.000000
Gradient difference: 0.000000
```

##### 玄学调参部分

```Python
from cs231n.classifiers import Softmax
results = {}
best_val = -1
best_softmax = None
learning_rates = [5e-9, 8e-8, 5e-8, 1e-7]
regularization_strengths = [8e3, 1e4, 1.5e4, 5e4, 1e5]

for lr in learning_rates:
    for reg in regularization_strengths:
        softmax = Softmax()
        tic = time.time()
        softmax.train(X_train, y_train, learning_rate=lr, reg=reg,num_iters=1500, 
                  verbose=True)
        y_train_pred = softmax.predict(X_train)
        y_val_pred = softmax.predict(X_val)
        train_acc = np.mean(y_train == y_train_pred)
        val_acc = np.mean(y_val == y_val_pred)
        if val_acc > best_val:
            best_val = val_acc
            best_softmax = softmax
        results[(lr, reg)] = (train_acc, val_acc)
    
# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))
    
print('best validation accuracy achieved during cross-validation: %f' % best_val)
```

我得到的最好结果是 36.5%

### 双层神经网络

这次作业最复杂的部分就是双层神经网络梯度的计算。

##### 前向传播

```Python
    h1 = np.dot(X, W1) + b1
    h1[h1 < 0] = 0.0
    # dropout = np.ones(h1.shape[0] * h1.shape[1])
    # dropout[np.random.choice(h1.shape[0] * h1.shape[1], int(0.1 * h1.shape[0] * h1.shape[1]))] = 0.0
    # h1 = h1 * np.reshape(dropout, h1.shape)
    scores = np.dot(h1, W2) + b2
```

看到有些博客上写 ```h1<0``` 应该为 ```h1<1e5``` ，防止临界无梯度，但我在实际训练时没有发现特别明显的问题。dropout 用于后续优化，目前不加。

##### loss 和梯度的计算

```Python
    scores = np.exp(scores)
    scores = scores / np.sum(scores, axis=1)[:, None]
    loss -= np.sum(np.log(scores[(np.arange(N), y)]))
    loss /= N
    loss += reg * (np.sum(W2 * W2) + np.sum(b1 * b1) + np.sum(W1 * W1) + np.sum(b2 * b2))

    # Backward pass: compute gradients
    grads = {}
    
    scores[(np.arange(N), y)] -= 1
    grads['b2'] = np.sum(scores, axis=0) / N + 2.0 * reg * b2
    grads['W2'] = np.dot(h1.T, scores) / N + 2.0 *  reg * W2
    dh1 = np.dot(scores, W2.T) / N
    mask = h1 <= 0
    dhdb = dh1
    dhdb[mask] = 0.0
    grads['b1'] = np.sum(dhdb, axis=0) + 2.0 * reg * b1
    grads['W1'] = np.dot(X.T, dhdb) + 2.0 * reg * W1
```

其中梯度的计算比较复杂，强烈推荐手推一遍公式，想清楚了再写代码，相比直接上手试错 debug 效率要高得多。

##### 调参

调参的部分与前面类似，不再赘述。我用 ```hidden_size=150, learning_rate=1e-3, learning_rate_decay=0.95``` 得到的验证集准确率为 55.5%，加上 dropout 最好时能到 60%。

##### 进一步优化

在优化时可加上 dropout，我尝试着多加一层神经网络，但是遇到了梯度消失的问题，可能需要对输入进行比例放大。

### 特征提取

特征提取部分比较简单，基本上就是前述方法的运用，只不过输入变成预提取的特征。

##### SVM 方法

```python
from cs231n.classifiers.linear_classifier import LinearSVM

learning_rates = [1e-9, 1e-8, 1e-7, 5e-8]
regularization_strengths = [5e4, 5e5, 5e6]

results = {}
best_val = -1
best_svm = None

for lr in learning_rates:
    for reg in regularization_strengths:
        svm = LinearSVM()
        svm.train(X_train_feats, y_train, learning_rate=lr, reg=reg,num_iters=1500, 
                  verbose=True)
        y_train_pred = svm.predict(X_train_feats)
        y_val_pred = svm.predict(X_val_feats)
        train_acc = np.mean(y_train == y_train_pred)
        val_acc = np.mean(y_val == y_val_pred)
        if val_acc > best_val:
            best_val = val_acc
            best_svm = svm
        results[(lr, reg)] = (train_acc, val_acc)

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))
    
print('best validation accuracy achieved during cross-validation: %f' % best_val)
```

神经网络方法

```Python
from cs231n.classifiers.neural_net import TwoLayerNet

input_dim = X_train_feats.shape[1]
hidden_dim = 500
num_classes = 10

net = TwoLayerNet(input_dim, hidden_dim, num_classes)
best_net = None
stats = net.train(X_train_feats * 100, y_train, X_val_feats * 100, y_val,
            num_iters=20000, batch_size=200,
            learning_rate=1e-3, learning_rate_decay=0.99,
            reg=0.05, verbose=True)
val_acc = (net.predict(X_val_feats) == y_val).mean()
print('Validation accuracy: ', val_acc)
best_net = net
```

这里我没有进行参数调整。使用上面的参数得到的准确率接近 60%。

值得注意的是，如果使用默认的参数进行训练，loss 几乎不降，打印了```grad['W1']``` 的模值发现过小。因此对输入进行了 100 倍的放大，减少精度损失。

### 



