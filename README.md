# cs231n-assignment(2024)
[CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/) 课程是经典的CV领域入门课程之一，我主要按照课程的[官方笔记](https://cs231n.github.io/) 进行学习，期间学习了一些[视频课程](https://www.bilibili.com/video/BV1K7411W7So?spm_id_from=333.788.videopod.episodes&vd_source=9b0bab44f379d04b6954be4ca93b4b5a) （视频会比笔记更容易理解），完成了3次作业，并在此记录作业的解答过程。

注：作业中的代码实现、公式推导均为本人所写，如有错误，欢迎指正。

# 课程笔记
[CS231n 官方笔记授权翻译总集篇发布](https://github.com/whyscience/CS231n-Note-Translation_CN/tree/master)

# 作业

## 作业-1
图像分类、kNN、SVM、Softmax、全连接神经网络 https://cs231n.github.io/assignments2024/assignment1/
### Q1:k-Nearest Neighbor classifier
1. kNN的作业引导 [knn.ipynb](https://github.com/ruip0729/cs231n/blob/main/assignment1/knn.ipynb)
2. kNN的实现 [k_nearest_neighbor.py](https://github.com/ruip0729/cs231n/blob/main/assignment1/cs231n/classifiers/k_nearest_neighbor.py)
### Q2:Training a Support Vector Machine
1. SVM的作业引导 [svm.ipynb](https://github.com/ruip0729/cs231n/blob/main/assignment1/svm.ipynb)
2. SVM损失函数实现 [linear_svm.py](https://github.com/ruip0729/cs231n/blob/main/assignment1/cs231n/classifiers/linear_svm.py)
3. 线性分类器实现 [linear_classifier.py](https://github.com/ruip0729/cs231n/blob/main/assignment1/cs231n/classifiers/linear_classifier.py)
### Q3:Implement a Softmax classifier
1. Softmax的作业引导 [softmax.ipynb](https://github.com/ruip0729/cs231n/blob/main/assignment1/softmax.ipynb)
2. Softmax损失实现 [softmax.py](https://github.com/ruip0729/cs231n/blob/main/assignment1/cs231n/classifiers/softmax.py)
3. 线性分类器实现 [linear_classifier.py](https://github.com/ruip0729/cs231n/blob/main/assignment1/cs231n/classifiers/linear_classifier.py)
4. Softmax函数求导 [softmax.md](https://github.com/ruip0729/cs231n/blob/main/%E8%A1%A5%E5%85%85%E5%86%85%E5%AE%B9/softmax%E5%87%BD%E6%95%B0%E6%B1%82%E5%AF%BC.md)
### Q4:Two-Layer Neural Network
1. TwoLayerNet的作业引导
2. 单层神经网络中前向传播和反向传播的实现
3. 两层神经网络的实现
4. 训练模型的封装
### Q5:Higher Level Representations: Image Features

## 作业-2
全连接和卷积网络、批量归一化、Dropout、Pytorch和网络可视化

## 作业-3
网络可视化、使用RNN和Transformer的图像描述、生成对抗网络、自监督对比学习
