# Civita_Laurea_Video_Classification Project

## Partner Background

Civita-Laurea is an intelligent and gamified platform that allows for redesigning the existing courses at universities, secondary schools, and other educative institutions flexibly and engagingly. By using AI models and motivational design, our system classifies and customizes the courses to the different student's profiles. The system will care about evaluations and activities assigned to each student based on their profiles. In this way, the professors' time will be optimized, and they will be able to focus on analyzing their students' progress and guiding them in real-time.
Civita Laurea aims to focus on helping and advising the professors to redesign their current videos to make them more engaging by adapting to student profiling and requirements regarding the videos, for example, if they want them to be more exercise-oriented or theory-oriented for their study. This is one of the various ideologies that Civita Laurea focuses upon in their journey for a new educational approach.

## Project Background 

The project "Machine Learning for Classifying Educational Videos" is an internationally based project that aims to put the initial milestone for the future vision of redesigning online education teaching used by many students and professors. The idea behind this project is to provide models that can classify the videos over the features such as "Digital(Slides) approach," "Handwritten(Classroom) approach," "Visuals" / "Non-Visuals" that represents if the video had used any visualization in its teaching approach or is wholly based on plain text. These deciding features were classified and sorted out by the host during requirement gathering. On the basis of these features and model classifications, Civita Laurea can judge the quality of any video and provide relevant feedback to the professors notifying them of their teaching style's strengths and weaknesses.
This is a start-up project; hence the requirement was to design Minimum Viable Product(s)(models) that can help the client and provide a pathway of the implementation such that it can be used and revised in the future when the client will be having more goals for improvising the online education paradigm.

## Aim of this project and repository:

1.	Extraction of Images based on keyframe extraction concept from video using the Python library and tagging the features.
2.	Detect the variability by determining the clusters of images through K-mean clustering techniques(unsupervised learning) that can help understand the level of interactivity of     the videos as if they are using any props while teaching or any presence of a human in the videos.
3.	Classification of the extracted images using deep learning after tagging of images with the features such as :
     a)	Style of teaching: Handwritten (Classroom teaching) or Digital (Slides Presentation) 
     b)	Presence of Visuals: Visualizations and No Visualization approaches.

## Methodology and Approach:

* The main aim was to deliver a Viable product(s) (Classification Model(s)) that can be refined in the future with the new addition of data and evaluation metrics for a video, as defined by the host. Also, to judge the variation in the videos by an unsupervised approach. The requirements were given in phases such that one could proceed with the second requirement will be given and processed.
* Keyframe images were extracted as valid data from around 420 videos. The data consisted of a variant length of video timings that varied between 5 minutes to 70 minutes. The number of extracted images depends on various factors like the time length of the videos, the changes in the scenes that have been detected, and the features present in the videos. More images will be extracted for more variations or slight activity changes in the scenes. The video length was also a determining factor in consideration to the keyframe extraction as a video with gave comparatively less number of images as compared to the numbers of images extracted from a video of length 70 minutes.
The data was manually explored, and some of the irrelevant images that did not satisfy the host's feature categories were removed. For example, extremely blurred out images as a recording error, image of a person blocking the view of the camera recordings, or some images were present which did not fall under any of the categories of the features that was given by the host, such only the face of the human where any action/feature was not identified. These images were used to identify the features of a video.
* We experimented with three models apart from the Sequential model. The convolution models chosen were Resnet50, Vgg16, InceptionV3 for prediction analysis over the images. They are some of the good performings models of Keras that can be the best fit as an initiative for the requirements of Civita Laurea. Along with these models we were able to construct some of Sequential Layers over the pre-built models, for a better training and accuracy purpose. The appropriate and optimal model(s) were given with hypertuning.

## Files and Folders Description in the Repository

### Video Image Extractors:

This folder contains two codes(in different Jupyter Notebook") for extracting the images from the video. One of the code "image_clustering_with_filemovements (1)" extracts the images on basis of features using a model of Keras and forms clusters of images from on video at a time. It had been studied that more clusters had been a result of more changes or variation in the images( videos). Another Jupyter Notebook is a simple extractor using puthon libraries that extract images from the videos on basis of various changes in the scene. It can be used if there is no need to form the cluster in future. 
Once the all the images had been collected, the data had been divided into subfolder having the features as folders name. For the current features, this had already been done ,but it should be repeated for a new feature added.
* Note: The images that had been extracted have the video_name as its initial and a number attached to them so that it is easy to track as which images are related to which videos.
* The "Sample Image.zip" shows the example as how images are being formed and segregated.

### Dataset(Image)-Tagging-Code:

This folder contains the Jupyter Notebook that had been used after the step of data extraction, once the subfolders with feature names is created, this code can be used totag the respective feature with the images

### Model_Neural_Network: 

* This folder contains the Jupyter Notebook for each models , InceptionV3 being the highest and optimal performing model, while Resnet50 is also being recommended with higher volume of testdata. The VGG16 model did not perform that well in comparison to all other models. The approach for the models is taken keeping in mind to augment the data for better training by flipping,rotating,shearing,zooming, rotating with an angle such that they gives a ariatrion to the existing data and help in better training as the dataset bore mostly similarity with various videos and some imbalance was noted in the data sets for features.
* The approach is to divide the images into respective folders having features name , the folders name serves as the label during the augmentation of images and data preparation for the model during the training. Since the data is imbalance we chose this approach else there is a high chances that most data will be missed during training from the minority class during shuffling. 
* The flow_from_directory method of Image DataGenerator of Keras package helps in directly reading the image from the path of folders provided and augmenting them. The generators than are sent for fitting and prediction. The subfolders in the main folder are features folders containing the images. A glimpse can be seen in the file "Sample Images.zip"
* For each JuputerNotebook, four folders need to be created , this approach was done to understand  the robustness of the algotithm. 
* The train folder will have the images used  for training."trainPathMode" contains the path to the Train folders with images.
* The validation folder will have the images used for validating the model performance during training. "validatePathmode" contains the path to Validation folder for images to     be validated while training.
* The test folder will contain the images that will be tested against the trained model."testPathmode" contains the path to Test folders for images to be tested. The data that had been tested and predicted will be shown in a csv file as "Predictions.csv" . This file contains the image name , its actual feature and its predicted feature. In this way , we can see how the model had behaved.
* Here all the folders as Train,Test, Validation folder has the images in the subfolders as per the features defined as we wanted the model to perform an extensive data training and testing inspite of data imbalance. So in case to get the label for unknown images from the videos used 
* The "Unlabeled" data folder will contain all the random images that had been collected to classify unseen data in future. Here we only need to put the images and the model will start labeling them and give the output in the csv file " Prediction_Unlabeled.csv"
* The prediction merge code merge all the prediction that has been done over the unlabeled data to generate one excel with image name with all the incorporated predicted features. We need to put all the valid files in one folder as named in the code.

### Model and Dataset Maintenance 

#### Tools Used/Required
The tools that are being used for the deliverables are mentioned below:
a)	Jupyter Notebook(Anaconda 3 package)
b)	Tensorflow( Version 2.4)
c)	Scikit Learn package (Version 0.23)
The version needs to be checked and updated with time if necessary else there will be some issues in model training.
#### Datasets
With keeping in mind the future expansion of the project, there will be addition of more images from various videos  for which we have two scenarios in scope:
* For introduction of any new feature apart from the exsiting one done , create new data set as per the steps involved in creating the existing one , that is , segrgate the images and separate them into the subfolders named as the feature that needs to identified . This segregation of the images will involve manual intervention as per Civita Laurea scope of decision to assign the feature for the images. This will help in preparation of Training, Validation and Test data in future . Once done and trained the model can predict on any unseen, unlabeled data.
* For the existing feature analysis , The training data should be audited regularly because if we consider the millions of videos available or made  for educational purpose. There will be a need to update the training data  for existing feature with changing time to make the model incorporate and adapt to the variations of videos. The new data needs to extracted from the new videos and added to the trainig data folder.
#### Models
With the incorporation of new data in the future , there may be a scenario where the models needs to be tuned for efficiency. Such as the current model InceptionV3 behaves good with the current feature analysis of Teaching style and presence of Visualizations . But with upgrading of features or large data in future , the models need to retrained and audited , if it seems that the accuracy is decreasing or the model is showing more signs of overfitting and underfitting than the model should be adjusted by changing the below parameters:
* Sequential layers of the models used , they can be increased
* Changing the value od dropout parameter or adding more dropout parameters
* Adding more neurons in Sequential layers such as the value in the Dense layer can be changed from 512 to 1024 or so on.

All the changes may vary with the Models outcome , it is an iterative process of training the data, checking the outcome, hypertuning for a better outcome. Different model may behave differently in respect to data and features 
Data Tagging and Prediction Merging codes
The code for tagging the dataset is a simpler and does not need a frequent update until there is a significant or mandatory change in the Scikit Learn package espacially os library. Same condition goes with the merging code for predictions.




