# PyTorch Tutorials
 
This Repository contains notebooks on PyTorch tutorials.

## [1. Tensors and Operations:](https://github.com/VishnuK11/Image-Classification/tree/main/Simpsons%20Character%20Recognition)
The notebook contains fundamental commands on tensors such as creating a tensor, printing its shape. Also, tensor operations like tensor addition, multiplication, gradients are explored.

## [2. Linear Regression:](https://github.com/VishnuK11/Image-Classification/tree/main/Fashion%20MNIST)
The notebook walks through code for Linear Regression task. Steps to create a model, define a loss function, making predictions, optimizer step based on gradient and iterating through epochs to arrive at a final prediction are detailed in this notebook. The task is based on prediction yield of fruits based on weather features like rainful, temperature and humidity as inputs.

## [3. Logistic Regression:](https://github.com/VishnuK11/Image-Classification/tree/main/Digits%20MNIST)
In this notebook, I walk through the code for using neural network for logistic regression to classify the digits of the MNIST digits dataset. I explore the steps around creating model, setting up loss function, optimizer, tracking and storing validation and training metadata like loss and accuracy. For the final output, softmax activation is used to estimate the probabilities of each class and a final prediction is made. The model architecture is a single layer Neural Network of size equal to the flattened number of pixels in the image. The model performed at 85% accuracy. 

## [4. Deep NN:](https://github.com/VishnuK11/Image-Classification/tree/main/Digits%20MNIST)
In this notebook, I walk through the code for using deep neural network for logistic regression to classify the digits of the MNIST digits dataset. Apart from steps discussed in the previous excercises, I create model with 1 hidden layer. Additionally, I create necessary functions to enable computation on GPU instead of CPU. For the final output, softmax activation is used to estimate the probabilities of each class and a final prediction is made. The model performed at 95% accuracy while training on only 5 epochs. 

## [5. CNN CIFAR:](https://github.com/VishnuK11/Image-Classification/tree/main/Digits%20MNIST)
MNIST digits dataset is easier to classify and hence we use CIFAR dataset containing 10 classes. I use 4 blocks of Convolutional Neural Network where each block consists of Conv-Relu-Conv-Relu-MaxPool layers. After extracting featuers using these blocks, I utilize a 3 layer linear neural network for classification. The model performed at 75% accuracy. 

## [6. ResNet CIFAR:](https://github.com/VishnuK11/Image-Classification/tree/main/Digits%20MNIST)
In this notebook, to classify the CIFAR 10 dataset, I modify the conv block to imitate the resnet architecture. Resnet architecture is such that, the input for Kth block is the sum of input of K-1 block and output of K-1 block. This ensures that the layers maintain a smaller weights without overfitting. Also each block contains batch normalization to negate effects of mean deviation due to batch wise training. After extracting featuers using these blocks, I utilize a 3 layer linear neural network for classification. The model performed at 90% accuracy. 

## [7. GANs:](https://github.com/VishnuK11/Image-Classification/tree/main/Digits%20MNIST)
Generative Adversarial Networks are a pair of neural network architectures that are used to generate images similar to training data. In this notebook, I train the Generator - Adversarial network to generate new anime faces. The below picture illustrate the inputs used to train the network. The output generated is also depicted below.

![Input](https://user-images.githubusercontent.com/30376279/111671248-c3d81400-883e-11eb-9081-2b3baa393938.png)
![Output](https://user-images.githubusercontent.com/30376279/111671175-af941700-883e-11eb-85bf-cfaf9e0120ac.png)


