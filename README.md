# cs231n-assignment(2024)
[CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/) 课程是经典的CV领域入门课程之一，我主要按照[官方的课程顺序](https://cs231n.stanford.edu/schedule.html)，结合课程的[官方笔记](https://cs231n.github.io/) 进行学习，期间学习了一些[视频课程](https://www.bilibili.com/video/BV1K7411W7So?spm_id_from=333.788.videopod.episodes&vd_source=9b0bab44f379d04b6954be4ca93b4b5a) （视频会比笔记更容易理解），完成了3次作业，并在此记录作业的解答过程。

注：作业中的代码实现、公式推导均为本人所写，如有错误，欢迎指正。

## 课程笔记
[CS231n 官方笔记授权翻译总集篇发布](https://github.com/whyscience/CS231n-Note-Translation_CN/tree/master)

## 作业-1
图像分类、kNN、SVM、Softmax、全连接神经网络
### Q1: k-Nearest Neighbor classifier
1. kNN的作业引导 [knn.ipynb](https://github.com/ruip0729/cs231n/blob/main/assignment1/knn.ipynb)
2. kNN的实现 [k_nearest_neighbor.py](https://github.com/ruip0729/cs231n/blob/main/assignment1/cs231n/classifiers/k_nearest_neighbor.py)
### Q2: Training a Support Vector Machine
1. SVM的作业引导 [svm.ipynb](https://github.com/ruip0729/cs231n/blob/main/assignment1/svm.ipynb)
2. SVM损失函数实现 [linear_svm.py](https://github.com/ruip0729/cs231n/blob/main/assignment1/cs231n/classifiers/linear_svm.py)
3. 线性分类器实现 [linear_classifier.py](https://github.com/ruip0729/cs231n/blob/main/assignment1/cs231n/classifiers/linear_classifier.py)
### Q3: Implement a Softmax classifier
1. Softmax的作业引导 [softmax.ipynb](https://github.com/ruip0729/cs231n/blob/main/assignment1/softmax.ipynb)
2. Softmax损失实现 [softmax.py](https://github.com/ruip0729/cs231n/blob/main/assignment1/cs231n/classifiers/softmax.py)
3. 线性分类器实现 [linear_classifier.py](https://github.com/ruip0729/cs231n/blob/main/assignment1/cs231n/classifiers/linear_classifier.py)
4. Softmax函数求导 [softmax.md](https://github.com/ruip0729/cs231n/blob/main/%E8%A1%A5%E5%85%85%E5%86%85%E5%AE%B9/softmax%E5%87%BD%E6%95%B0%E6%B1%82%E5%AF%BC.md)
### Q4: Two-Layer Neural Network
1. TwoLayerNet的作业引导 [two_layer_net.ipynb](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment1/two_layer_net.ipynb)
2. 单层神经网络中前向传播和反向传播的实现 [layers.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment1/cs231n/layers.py)
3. 两层神经网络的实现 [fc_net.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment1/cs231n/classifiers/fc_net.py)
4. 训练模型的封装 [solver.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment1/cs231n/solver.py)
### Q5: Higher Level Representations: Image Features
1. 图像特征的作业引导 [features.ipynb](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment1/features.ipynb)
2. Color Histogram和Histogram of Oriented Gradients (HoG) [课程ppt-lecture5](https://cs231n.stanford.edu/slides/2024/lecture_5.pdf)

## 作业-2
全连接和卷积网络、批量归一化、Dropout、Pytorch和网络可视化
### Q1: Multi-Layer Fully Connected Neural Networks
1. 多层全连接神经网络的作业引导 [FullyConnectedNets.ipynb](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment2/FullyConnectedNets.ipynb)
2. FullyConnectedNet类的实现 [fc_net.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment2/cs231n/classifiers/fc_net.py)
3. 单层神经网络中前向传播和反向传播的实现 [layers.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment2/cs231n/layers.py)
4. 多种更新规则的实现（sgd、sgd_momentum、rmsprop和adam） [optim.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment2/cs231n/optim.py)
### Q2: Batch Normalization
1. 归一化（batch normalization和layer normalization）的作业引导 [BatchNormalization.ipynb](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment2/BatchNormalization.ipynb)
2. BN和LN的前向传播和反向传播的实现 [layers.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment2/cs231n/layers.py)
3. 具有BN或LN层的全连接神经网络的实现 [fc_net.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment2/cs231n/classifiers/fc_net.py)
4. BN层反向传播利用链式法则求导 [批标准化中反向传播的链式法则推导.md](https://github.com/ruip0729/cs231n-assignment/blob/main/%E8%A1%A5%E5%85%85%E5%86%85%E5%AE%B9/%E6%89%B9%E6%A0%87%E5%87%86%E5%8C%96%E4%B8%AD%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E7%9A%84%E9%93%BE%E5%BC%8F%E6%B3%95%E5%88%99%E6%8E%A8%E5%AF%BC.md)
### Q3: Dropout
1. Dropout的作业引导 [Dropout.ipynb](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment2/Dropout.ipynb)
2. Dropout层的前向传播与反向传播的实现 [layers.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment2/cs231n/layers.py)
3. 具有Dropout层的全连接神经网络的实现 [fc_net.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment2/cs231n/classifiers/fc_net.py)
### Q4: Convolutional Neural Networks
1. CNN的作业引导 [ConvolutionalNetworks.ipynb](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment2/ConvolutionalNetworks.ipynb)
2. conv和pool层的前向传播和反向传播的实现（使用循环） [layers.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment2/cs231n/layers.py)
3. conv和pool层快速计算的实现（使用了im2col和reshape） [fast_layers.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment2/cs231n/fast_layers.py)
4. 各层之间的封装 [layer_utils.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment2/cs231n/layer_utils.py)
5. 三层CNN的实现 [cnn.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment2/cs231n/classifiers/cnn.py)
6. 空间批归一化和空间组归一化的实现 [layers.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment2/cs231n/layers.py)
### Q5: PyTorch on CIFAR-10
1. pytorch教程 [pytorch-examples](https://github.com/jcjohnson/pytorch-examples)
2. pytorch的作业引导 [PyTorch.ipynb](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment2/PyTorch.ipynb)

## 作业-3
网络可视化、使用RNN和Transformer的图像描述、生成对抗网络、自监督对比学习
### Q1: Image Captioning with Vanilla RNNs
1. RNN的作业引导 [RNN_Captioning.ipynb](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment3/RNN_Captioning.ipynb)
2. RNN各层前向传播和反向传播的实现 [rnn_layers.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment3/cs231n/rnn_layers.py)
3. 使用RNN进行图像描述的流程解释 [Image Captioning流程解释.md](https://github.com/ruip0729/cs231n-assignment/blob/main/%E8%A1%A5%E5%85%85%E5%86%85%E5%AE%B9/Image%20Captioning%E6%B5%81%E7%A8%8B%E8%A7%A3%E9%87%8A.md)
4. 用于图像描述的RNN的实现 [rnn.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment3/cs231n/classifiers/rnn.py)
5. 图像描述模型的封装 [captioning_solver.py](https://github.com/ruip0729/cs231n-assignment/blob/main/assignment3/cs231n/captioning_solver.py)
### Q2: Image Captioning with Transformers
### Q3: Generative Adversarial Networks
### Q4: Self-Supervised Learning for Image Classification
### Extra Credit: Image Captioning with LSTMs
