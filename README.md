![Header](/images/header.png)

## Name That Mushroom!
The world of mycology is a veritable universe unto itself, in the soil around all of us. Forest floors invite exploration and discovery for young and old, novices and experts. And the world of mushrooms is diverse enough that even a seasoned mushroom hunter will find mushrooms she can’t always identify. Mushrooms are at once alluring and anxiety-causing; the right mushroom can be delicious, but the wrong mushroom can be a toxic experience. Imagine if a computer could help you identify which mushroom you’re looking at? In this project, we will train a machine learning model on images of different common mushroom genus, to see if it can be like having a mushroom expert in your pocket!

## 1. Data
The series of mushroom images comes from the Kaggle website, link below:

[Kaggle Dataset](https://www.kaggle.com/maysee/mushrooms-classification-common-genuss-images)


## 2. Method

Neural networks, with their ability to identify and look for patterns in visual data, are particularly useful for image classification. Two neural network types are explored:

**1.	Convolutional Neural Network (CNN):** Convolutional methods “convolve” the individual pixel data, aggregating them in useful ways for pattern recognition. 
**2.	Residual Neural Network (ResNet):** Extremely deep learning (i.e., many layers) suffer from issues with vanishing gradients. ResNet algorithms have been in use for a while, but in the last several years have gained prominence for their ability to deal with vanishing gradients with techniques such as skip connections, allowing deeper learning and greater classification accuracy.

![Resnet](/images/resnet_model.png)

## 3. Data Cleaning
The Kaggle dataset is pre-organized into subfolders, each representing a common mushroom genus. While the majority of the images are perfectly fine for use, there are a couple data cleaning opportunities:
**•	Problem 1:** The original dataset contains images with writing, highlights, arrows and circles, probably gleaned from books and websites about the subject of mushroom classification. While a human can ignore these image features as clearly non-natural, a computer may not be able to. 

![Arrows](/images/arrows.png)
 
**•	Problem 2:**  The original dataset contains images, some microscopic, of spores of the genus types. 

![Micro](/images/micro.png)
 
**•	Problem 3:** Several images which show sliced mushrooms. While these are not a natural form for mushrooms in the forest, different mushrooms can have different discolorations when cut and exposed to air. Ultimately, I decided to leave these images in the dataset to preserve this potentially useful, if confusing, information.

![Sliced](/images/sliced.png)

## 4. EDA
[EDA Report](https://github.com/david-olivero/Mushrooms/blob/main/Mushroom_Image_Processing_EDA.ipynb)

Exploratory data analysis consisted of preprocessing the images to a common size format. All images in the dataset are already color images with RBG channels. The images came in a wide variety of image sizes and aspect ratios, however; some wider than tall, some taller than wide.
Looking at the images, mushrooms were fairly well centered, which presented an opportunity for cropping each image around the center, as shown below:

![original](/images/original_hygrocybe.png)
![cropped](/images/cropped_hygrocybe.png)

The dataset presented two significant challenges for any classification algorithm. (For human beings, too!)
First, **Variety Within Class:** Mushrooms within a single genus can vary significantly in how they present themselves. A great example of this is the enormous variation that can be found within the Hygrocybe genus:

![variety](/images/variety.png)

Second, **Similarity Across Class:**  Kind of a corollary to all the variation within a genus, is that two different genera can end up looking pretty similar to each other. Here’s a few left vs. right examples to illustrate this. Sometimes, a Suillus can look like a Boletus; a Lactarius can look like a Russula; and an Amanita can look like an Entoloma. Certainly, even a human would be confused by these similarities! 

![similarity](/images/similarity.png)

## 5. Splitting the Dataset
The various training, validation and hold-out test sets are constructed as follows:
1.	Data Augmentation: Create sibling images of each image in the dataset using flipping and rotation schemes from the Pillow library
2.	Use Python for-loops, lists and Numpy to read each image in sequence as a new row in an array (“training_data”). Alongside, a separate array (“training_labels”) encoding each row with its original category, i.e., the subfolder the image came from.
3.	Use SciKit-Learn train_test_split to randomly shuffle the labelled data and assign a certain percentage (in this case, 10% of the original data), which the model can never see during training, as a hold-out test set after model training. 
4.	One-Hot encode the label arrays in both the training and test sets. 
5.	Using TensorFlow, compile the model to be tested. 
6.	Finally, train the model on the training set. During training, set aside a certain percentage (in this case, 20% of the training set) of the training set as a validation set. Stop the algorithm when the validation set accuracy fail to improve for numerous iterations (i.e., plateaus), using EarlyStopping callbacks.

## 6. Machine Learning
To explore the kind of model required for a problem of this scope, we’ll take a look at three models, ranging from simple to complex: We’ll start with a dummy stratification regression, then use a Convolutional Neural Network, then finally the ResNet50 model already described. 

**Convolutional Neural Network:** This model uses convolutions of local groups of pixels in an image, in an attempt to look for patterns. This kind of model has demonstrated to be quite effective with simple multi-class problems like MNIST handwritten digits and two-class mushroom problems. The model results in about 130,000 trainable parameters:

![conv2D](/images/conv_summary.png)

**ResNet50:** For this model, the a ResNet50 model weighted for the ImageNet database was added to several Batch Normalization and trainable Dense layers. 
The resulting model has over 24 million parameters, and over half a million trainable parameters. The model summary is as follows:

![resnet](/images/resnet_summary.png)

## 7. Predictions
The chart blow compares the accuracy achieved with a dummy stratification model (i.e., randomly assigning one of seven categories ot each image), the CNN, and the ResNet50 model. 

![compare](/images/model_compare.png)

It is interesting that it takes a highly complex, pre-trained ResNet model to perform wel with this dataset. With a much larger quantity of high quality images, we should expect the CNN model to perform much closer to ResNet50 in accuracy (as it does with fewer categories of mushrooms).


To be sure, in many computer vision exercises, a significantly higher accuracy can be obtained, often well in excess of 90%. The mushroom image set presents some fairly unique challenges, as has been discussed above. Indeed, it’s fair to say that 88% accuracy might be as good as what any true human expert could be expected to achieve based on images of mushrooms alone (i.e., they don’t get to touch, smell, cut, or inspect the surroundings of a mushroom, which are clues a mycologist often uses in the identification process.) 
A **confusion matrix heatmap** for the hold-out test set is shown below. Rather than randomly distributed error, the confusion matrix shows particular identification issues; for example, thinking too many images are Hygrocybes, and having a hard time finding all the Cortinarius and Entoloma mushrooms.

![confusion](/images/confusion.png)

A breakdown of each mushroom genus looks like this. On an individual genus basis, accuracy looks quite good. The model seems especially good with Boletus and Amanita mushrooms. Which is good, because Boletus are delicious, while Amanita are poisonous!

![table](/images/table.png)


## 8.Future Improvements
a)	With more time, it may be possible to break through the 90% accuracy level. While the exact method that will accomplish this is not known, I suspect that preventing overfitting of ResNet50 on training data may prove fruitful.  
b)	Experimenting with further image manipulations, such as superimposing a sobel edge detect image onto a standard RGB image. This might possibly allow the model to better identify the mushrooms in an image. 
c)	As a follow on to note (b) above, it may be possible to count the number of mushrooms in an image using edge detect algorithms like sobel or canny. If possible, that might be a useful feature to add to the dataset. 
d)	And, as always, procuring more high-quality images data of these mushroom classes! This might enable the far simpler CNN model to perform well. 

## 9. Credits
Many thanks to Springboard and Paperspace for access to their Gradient GPU environments which allowed these highly complicated neural network models to train quickly, allowing the great deal of iteration that was required to tune the model performance.  Thanks especially to Branko Kovac for his encouragement and excellent counsel during this project! 




