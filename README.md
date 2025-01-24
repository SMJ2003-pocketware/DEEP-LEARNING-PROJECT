# DEEP-LEARNING-PROJECT

*COMPANY*: CODTECH IT SOLUTIONS PVT.LTD

*NAME*: SOHAM MAJUMDER

*INTERN ID*: CT6WMIS

*DOMAIN*: DATA SCIENCE

*DURATION*: 6 WEEKS

*MENTOR*: NEELA SANTHOSH KUMAR

As part of this task, I implemented a deep learning model for image classification using TensorFlow. For this project, I worked in Google Colab, a cloud-based Jupyter Notebook environment. I chose Colab because it provides a hassle-free setup with pre-installed libraries like TensorFlow and the ability to use GPUs for faster training. This task aimed to build a functional deep learning pipeline that preprocesses data, trains a model, evaluates it, and visualizes the results, offering a complete end-to-end solution for image classification. The dataset I used was CIFAR-10, a widely-used benchmark dataset in computer vision. CIFAR-10 consists of 60,000 32x32 pixel color images divided into 10 categories, such as airplanes, automobiles, birds, and cats. This made it an excellent choice for demonstrating the capabilities of deep learning in image classification.
The project began with loading and preprocessing the dataset. CIFAR-10 images were normalized by scaling their pixel values to a range of 0 to 1, which is crucial for stable and efficient training of neural networks. The class labels, which were initially integers, were converted into one-hot encoded vectors to suit the categorical cross-entropy loss function used later in the model. These preprocessing steps ensured the data was in an optimal format for training the deep learning model.
For the model architecture, I built a Convolutional Neural Network (CNN) using TensorFlow's Keras API. CNNs are a standard choice for image-related tasks due to their ability to learn spatial hierarchies in the data. The model consisted of three convolutional layers, each followed by a max-pooling layer to downsample feature maps and reduce computational complexity. After the convolutional layers, the network flattened the features into a vector, which was then passed through a dense hidden layer with 64 neurons and a ReLU activation function. Finally, the output layer had 10 neurons with a softmax activation function to classify the images into one of the 10 categories. I used Adam as the optimizer due to its efficiency and popularity in training deep learning models.
The model was trained for 10 epochs using a batch size of 64, and both the training and validation accuracy and loss were monitored. After training, the model achieved a competitive accuracy on the test set, demonstrating its ability to generalize well to unseen data. To make the results more meaningful, I visualized the training process by plotting the accuracy and loss curves for both the training and validation sets. This gave me insights into how the model improved over epochs and whether it was overfitting or underfitting.
Lastly, I added a visualization step where the model made predictions on randomly selected test images. These predictions were displayed alongside the true labels, allowing me to assess the model's performance visually. This project showcased how deep learning can be applied to image classification and provided valuable hands-on experience with TensorFlow and CNNs. Such techniques are widely applicable in fields like autonomous vehicles, medical imaging, and retail for tasks like object detection, disease diagnosis, and product categorization. Overall, this task was a rewarding experience that reinforced my understanding of deep learning workflows and their real-world applications.

#OUTPUT

![Image](https://github.com/user-attachments/assets/30fb02b3-cfea-4249-989e-76dca6eeeda5)

![Image](https://github.com/user-attachments/assets/dacb6280-e5c0-4bbe-98c9-60896ce64658)
