This program aims to classify handwritten digits (from 0 to 9) using a Convolutional Neural Network (CNN) on the MNIST dataset. Essentially, it trains a model to recognize and predict the digit represented in an image.



How It Works:
	1. Data Loading:
		○ The MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits, is loaded. Each image is 28x28 pixels in grayscale.
	2. Data Preprocessing:
		○ The images are reshaped into a 4D format suitable for CNNs and normalized, meaning pixel values are scaled to range between 0 and 1. This helps the model learn more effectively.
	3. Model Architecture:
		○ The model consists of several layers:
			§ Convolutional Layers: These layers apply filters to the images to extract features such as edges and shapes.
			§ MaxPooling Layers: These layers reduce the size of the feature maps, making computations faster and reducing the risk of overfitting.
			§ Flatten Layer: Converts the 2D outputs from the convolutional layers into a 1D array, preparing them for the dense layers.
			§ Dense Layers: Fully connected layers where the final layer uses a softmax activation function to output probabilities for the 10 digit classes.
	4. Compilation:
		○ The model is compiled with the Adam optimizer for efficient weight updates and the sparse categorical cross-entropy loss function, which is suitable for multi-class classification problems.
	5. Training:
		○ The model is trained over 5 epochs using a batch size of 64. It utilizes 90% of the training data for learning and 10% for validation during training to monitor performance.
	6. Evaluation:
		○ After training, the model is evaluated on the test dataset to check its accuracy, which is then printed out.
	7. Visualization:
		○ Finally, the program plots a graph comparing training accuracy to validation accuracy over the epochs. This visual representation helps in understanding how well the model is performing during training.

