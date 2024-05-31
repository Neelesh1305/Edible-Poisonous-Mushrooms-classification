## Poisonous and Edible Mushroom detection

#### Problem
The Mushrooms which are in the Northeastern Region of Thailand, include some species of edible mushrooms and some of poisonous mushrooms. The detection of poisonous and edible mushrooms is a difficult task for any newcomer in that region. Developing a classification model which distinguishes between edible and poisonous mushrooms would greatly help inexperienced locals and newcomers to this region in identifying edible mushrooms, which is one of the most important survival traits in that region.

#### Data collection/feature extraction
The image data of the mushrooms is taken from https://zenodo.org/record/6378474#.ZCLOsC-B1MQ , which includes images of resolution 227x227 pixels and there are in total 2000 images. The data here is downloaded into image directories, from which feature extraction is done using resnet50 as the base model.
We define a clean source function in order to identify just the image files and read them, preventing the reading of other non-image files. The images are converted into a readable format by feature extraction, which can be done in various methods. While in our problem we choose to extract the features using image_dataset_from_directory and the base model Resnet50. Principal component analysis is performed to obtain data with 3 features and a variance ratio of 0.767.

#### Modeling
Various classification models were used for this task, which included ML models like Logistic Regression, Random Forest Classifier, SVC Classifier, Adaboost classifier, KNN Classifier and many more. Pre-trained convolulonal neural network models like VGG16, MobileNetV2 and Resnet50 were also used. maong the models used Resnet50 model received a validation accuracy of around 98.6% which was the highest among them all.
Resnet50 is a pre-trained convolutional neural network mainly used for image classification. It uses a residual deep learning framework and has 50 layers, the network models the residual, as a layer has access to both the immediately prior layer and outputs further down the stack.

#### Evaluation and Results
Using the Resnet50 model for our classification task yielded a training accuracy of 0.99 and a validation accuracy of 0.98.
A test image set of 30 images is separated from the image dataset which is subject to testing our winning model. This yields the following metrics: loss: 0.1906 - accuracy: 0.9666. The edible and poisonous mushrooms in our model predictions are illustrated below:
<img width="551" alt="Screenshot 2024-05-31 at 2 09 58 PM" src="https://github.com/Neelesh1305/Edible-Poisonous-Mushrooms-classification/assets/113800036/d129147d-016a-46ee-a6ea-dfd0ff50a1aa">

#### Conclusion
The Mushrooms can now be classified into edible and poisonous with almost 99% accuracy, which is a pretty good prediction accuracy. Further developments would be collecting more images, like having more samples of the mushroom species and in different stages of their life cycle. This would broaden our approach in classification.

#### References
1. ResNet-50 convolutional neural network - MATLAB resnet50. (n.d.). Www.mathworks.com. Retrieved May16,2023, from https://www.mathworks.com/help/deeplearning/ref/resnet50.html#responsive_offcanvas 
2. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. https://arxiv.org/pdf/1512.03385v1.pdf
3. Ketwongsa, W., Boonlue, S., & Kokaew, U. (2022, March 23). Poisonous and Edible Mushrooms in the Northeastern Region of Thailand. Zenodo; Zenodo. https://zenodo.org/record/6378474#.ZCLOsC-B1MQ
