# internship_assignment
Predicting score in a test and challenge student to beat it by using the MongoDB data given.


I have developed an ANN that can predict final test scores based on various features. 

I have first analyzed and preprocessed the dataset and then made the data ready for feeding into the neural network. 

The neural network is just 3 layer based and till gives an accuracy over 97.2%.

I have used RELU for adding non linearity and also mean squared error for loss estimation.

I have also plotted the loss for better visualization.

I have assumed a few things and hence excluded a few columns based on correlation matrix and other means. I found that using a batch size of 64 and validation split of 30% was providing me good results.

I have for now just worked with the most of the major parameters/features that determine the final marks and still got an accuracy of 97.2%.

On increasing the layers and tweaking the other hyperparameters can vary our results, thus leaving more room for improvement.
