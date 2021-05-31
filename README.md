# AI Explainability for Multi-label Image Classification
## Overview
A Program that explains how the input features of a machine learning model affect its predictions using multi-label image classification. It also explains what the **Inception_V3 model** extracts as features to classify images in a particular way. We highlight specific parts of the Image (**Superpixels**) which are used to extract the features of an image. This is done for the top predicted class by the model. The **Superpixels** are used to generate random **perturbations** of the image which are used as inputs for a **Linear Regression Model**. The weights for this regression models are calculated using the distance (relation) between the original image and the perturbed images. The model then displays the top four Superpixels that have maximum correlation with the label.

---

## Implementation

* Predicting the class of an image
    * Input Image: 
    * ![image](https://user-images.githubusercontent.com/77194307/120222003-ee8e1200-c25c-11eb-8fe3-d957bc0094cf.png)


    * We use a pretrained [Inception_V3 model](https://keras.io/api/applications/inceptionv3/) available in Keras to predict the top five classes of our input image. Firstly, we have to pre-process our image so that it can be fed into the model, this is done using [skimage.transform.resize()](https://scikit-image.org/docs/stable/auto_examples/transform/plot_rescale.html). 
    * After getting the top 5 predicted classes we would use the top predicted class to explain the features of the image. 

* Generating perturbations of the input image
    * We implement the Quickshift Segmentation Algorithm using [skimage.segmentation.quickshift](https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.quickshift) that creates [Superpixels](https://darshita1405.medium.com/superpixels-and-slic-6b2d8a6e4f08) of our image.
    * Superpixelated Image:
    * ![image](https://user-images.githubusercontent.com/77194307/120222117-11b8c180-c25d-11eb-9217-d4844008bf0e.png)

    * Then we find the unique superpixels using [numpy.unique()](https://numpy.org/doc/stable/reference/generated/numpy.unique.html).
    * After this, we generate perturbations of the original image by masking out random superpixels from the image using [numpy.random.binomial()](https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html) 
    * Examples of perturbed images:
    * ![image](https://user-images.githubusercontent.com/77194307/120222439-90adfa00-c25d-11eb-90f6-c1c02ada2b41.png) ![image](https://user-images.githubusercontent.com/77194307/120222483-a1f70680-c25d-11eb-8608-17ee10eb0f7b.png) ![image](https://user-images.githubusercontent.com/77194307/120222510-acb19b80-c25d-11eb-90e5-7a5040e84cf8.png)

* Predicting class of perturbed images
    * Now we would again use the [Inception_V3 model](https://keras.io/api/applications/inceptionv3/) to predict the class of the perturbed images.
    * We save our predictions in an array `predictions`, the model gives the probabilities of each of the 1000 classes on which it is pre-trained for. But we would only consider the top predicted class as we would explain the features of the image with that label.
* Calculating the importance (weight) of each perturbed image
    * We use the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) to calculate the distance between the perturbed and original image. For implementation we  use the function [sklearn.metrics.pairwise_distances](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html)
    * Then these distances are normalized between 0 and 1 using a kernel function.
    * Kernel Function used (x dentoes distance between perturbed and original image):
    * ![image](https://user-images.githubusercontent.com/77194307/120212407-97357500-c24f-11eb-8429-dea0b36d6a36.png)

* Fitting a Linear Regression Model 
    * We make a weighted linear model using `predictions`, `perturbations` and `weights`.
    * From this model we get a coeffecient that denotes which perturbed image has features that are used to recognise its class. Higher coeffecients means that those superpixels have major role in prediction of the class.
* At last, we only turn on the top 4 Superpixels of the original image , the parts highlighted show the features that were used by the Inception_V3 Model to predict the class of the image.
* Image Highlighting the predicted label:
    * ![image](https://user-images.githubusercontent.com/77194307/120222694-fbf7cc00-c25d-11eb-85b0-a3beb7e824ee.png)



## References
* [“Rethinking the Inception Architecture for Computer Vision” by Szegedy, et. al.](https://arxiv.org/abs/1512.00567v3)
* [Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. “Why should I trust you? : Explaining the predictions of any classifier.” ](https://arxiv.org/abs/1602.04938)
* [Classify any Object using pre-trained CNN Model](https://towardsdatascience.com/classify-any-object-using-pre-trained-cnn-model-77437d61e05f)
* [Interpretable Machine Learning for Image Classification with LIME](https://towardsdatascience.com/interpretable-machine-learning-for-image-classification-with-lime-ea947e82ca13)
* [Explanation behind superpixel Segmentation](https://medium.com/@darshita1405/superpixels-and-slic-6b2d8a6e4f08)


