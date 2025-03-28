#!/usr/bin/env python
# coding: utf-8

# # Using Machine Learning Tools 2024, Assignment 3
# 
# ## Sign Language Image Classification using Deep Learning

# ## Overview
# 
# In this assignment you will implement different deep learning networks to classify images of hands in poses that correspond to letters in American Sign Language. The dataset is contained in the assignment zip file, along with some images and a text file describing the dataset. It is similar in many ways to other MNIST datasets.
# 
# The main aims of the assignment are:
# 
#  - To implement and train different types of deep learning network;
#  
#  - To systematically optimise the architecture and parameters of the networks;
#   
#  - To explore under- or over-fitting and know what appropriate actions to take in these cases.
#  
# 
# During this assignment you will go through the process of implementing and optimising deep learning approaches. The way that you work is more important than the results for this assignment, as what is most crucial for you to learn is how to take a dataset, understand the problem, write appropriate code, optimize performance and present results. A good understanding of the different aspects of this process and how to put them together well (which will not always be the same, since different problems come with different constraints or difficulties) is the key to being able to effectively use deep learning techniques in practice.
# 
# This assignment relates to the following ACS CBOK areas: abstraction, design, hardware and software, data and information, and programming.
# 

# ## Scenario
# 
# A client is interested in having you (or rather the company that you work for) investigate whether it is possible to develop an app that would enable American sign language to be translated for people that do not sign, or those that sign in different languages/styles. They have provided you with a labelled dataset of images related to signs (hand positions) that represent individual letters in order to do a preliminary test of feasibility.
# 
# Your manager has asked you to do this feasibility assessment, but subject to a constraint on the computational facilities available.  More specifically, you are asked to do **no more than 50 training runs in total** (where one training run consists of fitting a DL model, with as many epochs as you think are needed, and with fixed model specifications and fixed hyperparameter settings - that is, not including hyper-parameter optimisation). In addition, because it is intended to be for a lightweight app, your manager wants to to **limit the number of total parameters in each network to a maximum of 500,000.** Also, the data has already been double-checked for problems by an in-house data wrangling team and all erroneous data has already been identified and then fixed by the client, so you **do not need to check for erroneous data** in this case.
# 
# In addition, you are told to **create a fixed validation set and any necessary test sets using _only_ the supplied _testing_ dataset.** It is unusual to do this, but here the training set contains a lot of non-independent, augmented images and it is important that the validation images must be totally independent of the training data and not made from augmented instances of training images.
# 
# The clients have asked to be informed about the following:
#  - **unbiased median accuracy** estimate of the letter predictions from a deep learning model
#  - the letter with the highest individual accuracy
#  - the letter with the lowest individual accuracy
#  - the three most common single types of error (i.e. where one letter is being incorrectly labelled as another)
#  
# Your manager has asked you to create a jupyter notebook that shows the following:
#  - loading the data and displaying a sample of each letter
#  - training and optimising both **densely connected** *and* **CNN** style models
#  - finding the best single model, subject to a rapid turn-around and corresponding limit of 50 training runs in total
#  - reporting clearly and concisely what networks you have tried, the method you used to optimise them, the associated learning curves, the number of total parameters in each, their summary performance and the selection process used to pick the best model
#      - this should be clear enough that another employee, with your skillset, should be able to take over from you and understand your code and your methods
#  - results from the model that is selected as the best, showing the information that the clients have requested
#  - it is hoped that the median accuracy will exceed 94% overall and better than 85% for every individual letter, and you are asked to report (in addition to the client's requests):
#      - the overall mean accuracy
#      - the accuracy for each individual letter
#      - a short written recommendation (100 words maximum) regarding how likely you think it is to achieve these goals either with the current model or by continuing to do a small amount of model development/optimisation
# 

# ## Guide to Assessment
# 
# This assignment is much more free-form than others in order to test your ability to run a full analysis like this one from beginning to end, using the correct procedures. So you should use a methodical approach, as a large portion of the marks are associated with the decisions that you take and the approach that you use.  There are no marks associated with the performance - just report what you achieve, as high performance does not get better marks - to get good marks you need to use the right steps as well as to create clean, concise code and outputs, just as you've done in other assignments.
# 
# Make sure that you follow the instructions found in the scenario above, as this is what will be marked.  And be careful to do things in a way that gives you an *unbiased* result.
# 
# The notebook that you submit should be similar to those in the other assignments, where it is important to clearly structure your outputs and code so that it could be understood by your manager or your co-worker - or, even more importantly, the person marking it! This does not require much writing beyond the code, comments and the small amount of output text that you've seen in previous assignments.  Do not write long paragraphs to explain every detail of everything you do - it is not that kind of report and longer is definitely not better.  Just make your code clear, your outputs easy to understand (very short summaries often help here), and include a few small markdown cells that describe or summarise things when you think they are necessary.
# 
# Marks for the assignment will be determined according to the rubric that you can find on MyUni, with a breakdown into sections as follows:
#  - 30%: Loading and displaying data, plus initial model training (acting as a baseline)
#  - 50%: Optimisation of an appropriate set of models in an appropriate way (given the imposed constraints)
#  - 20%: Comparison of models, selection of the single best model and reporting of final results
# 
# Your report (notebook) should be **divided clearly into three sections**, corresponding to the three bullet points listed above.
# 
# Remember that most marks will be for the **steps you take**, rather than the achievement of any particular results. There will also be marks for showing appropriate understanding of the results that you present.  
# 
# What you need to do this assignment can all be found in the first 10 weeks of workshops, lectures and also the previous two assignments.

# ## Final Instructions
# 
# While you are free to use whatever IDE you like to develop your code, your submission should be formatted as a Jupyter notebook that interleaves Python code with output, commentary and analysis, and clearly divided into three main sections as described above. 
# - All data processing must be done within the notebook after calling appropriate load functions.
# - Comment your code appropriately, so that its purpose is clear to the reader, but not so full of comments that it is hard to follow the flow of the code. Also avoid interspersing, in the same cell, code that is run with function definitions as they make code hard to follow.
# - In the submission file name, do not use spaces or special characters.
# 
# The marks for this assignment are mainly associated with making the right choices and executing the workflow correctly and efficiently, as well as having clean and concise code and outputs. Make sure your code and outputs are easy to follow and not unnecessarily long. Use of headings and very short summaries can help, and try to avoid lengthy portions of text or plots. The readability of the report (notebook) will count towards the marks (and please note that _excessive_ commenting or text outputs or text in output cells is strongly discouraged and will result in worse grades, so aim for a modest, well-chosen amount of comments and text in outputs).
# 
# This assignment can be solved using methods from sklearn, pandas, matplotlib, seaborn and keras/tensorflow, as presented in the workshops. Other high-level libraries should not be used, even though they might have nice functionality such as automated hyperparameter or architecture search/tuning/optimisation. For the deep learning parts please restrict yourself to the library calls used in workshops 7-10 or ones that are very similar to these. You are expected to search and carefully read the documentation for functions that you use, to ensure you are using them correctly.
# 
# As ususal, feel free to use code from internet sources, ChatGPT or the workshops as a base for this assignment, but be aware that they may not do *exactly* what you want (code examples rarely do!) and so you will need to make suitable modifications. Appropriate references for substantial excerpts, even if modified, should be given.
# 

# ## Section 1  Loading and displaying data, plus initial model training (acting as a baseline)

# ###### 1.1) Loading and importing necessary libraries 

# In[1]:


import sys
assert sys.version_info >= (3, 5)


import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
import os, time
import pandas as pd
import tensorflow as tf
from tensorflow import keras
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
mpl.rc('figure', dpi=100)
import seaborn as sns; sns.set()


# ###### 1.2) Initial Visualization 

# In[2]:


# Now we will load both the training and testing file in our environment and proceed with further analysis

train_df = pd.read_csv("C:/Users/Aditya Venugopalan/Downloads/sign_mnist_train.csv")
test_df = pd.read_csv("C:/Users/Aditya Venugopalan/Downloads/sign_mnist_test.csv")


# In[3]:


train_df.head()


# In[4]:


test_df.head()


# In[5]:


# Lets try to have a quick glance of what is inside the training set and testing set
print(f'Number of images in the training dataset : {train_df.shape[0]}')
print(f'Number of images in the test dataset : {test_df.shape[0]}')
dim = int((train_df.shape[1]-1)**0.5)
print(f'Dimensions of the images : {dim} x {dim}')
print("Max value of the train dataframe", train_df.max().max())
print("Min value of the train dataframe", train_df.min().min())


# In[6]:


# Lets check the presence of Nan Values 

# Check for null values in the training dataset
null_values_train = train_df.isnull().sum().sum()
print(f"Number of null values in the training dataset: {null_values_train}")

# Check for null values in the testing dataset
null_values_test = test_df.isnull().sum().sum()
print(f"Number of null values in the testing dataset: {null_values_test}")


# In[7]:


# display image method
import tensorflow as tf

def to_image_tf(array, label=True):
    array = np.array(array)
    start_index = 1 if label else 0
    image = array[start_index:].reshape(dim, dim).astype(float)
    return tf.image.convert_image_dtype(image, dtype=tf.float32)

# Display single image using TensorFlow and Matplotlib
image_tf = to_image_tf(train_df.iloc[0])
plt.imshow(image_tf, cmap='gray')
plt.show()
#  I admit to take the help of chatgpt for the above code for referencing purposes


# In[8]:


print("Unique labels in the dataset:", train_df['label'].unique())


# In[9]:


# Define the class mapping to letters
class_values = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'
letter_mapping = {i: letter for i, letter in enumerate(class_values)}

# Create a dictionary to store one example image per label
images_per_letter = {}

# Get unique labels from the dataset
unique_labels = train_df['label'].unique()

# Collect one image per unique label
for label in unique_labels:
    image = to_image_tf(train_df[train_df['label'] == label].iloc[0])  # Select the first image for each label
    images_per_letter[label] = image

# Set up the figure with a title for each subplot
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(15, 10))
fig.suptitle('Example Images for Each Letter', fontsize=20)

# Plot each image with corresponding letter label
for ax, (label, image) in zip(axes.flat, images_per_letter.items()):
    ax.imshow(image, cmap='gray')
    ax.set_title(f'{letter_mapping[label]}', fontsize=15)
    ax.axis('off')

# Hide any remaining empty axes
for ax in axes.flat[len(images_per_letter):]:
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



# In[10]:


# Now we will visualize the distribution of images acroos different lables(classes) in the dataset with the help of a barplot
def class_plot(label):
    value_count = label.value_counts()
    plt.figure(figsize=(20,5))
    sns.barplot(x=sorted(value_count.index), y=value_count, palette="coolwarm")
    plt.title("Number of pictures per category", fontsize=15)
    plt.xticks(fontsize=15)
    plt.show()

# for training
class_plot(train_df['label'])




# In[11]:


# for testing
class_plot(test_df['label'])


# ###### 1.1 , 1.2 ) Summary 
# ###### As we can see the distribution of images across different classes in both training and testing seem to be slightly imbalanced but if we focus just on the testing set the testing set is is a bit more unbalanced  , with some classes having fewer images for both sets . The color gradient helps visualization better by showingg classes with more data in blue and classes in less data with red 

# ###### 1.3) Data splitting 

# In[12]:


y_train = train_df['label'] # extracting target variable
y_test = test_df['label'] # extracting target variable

# Now we will reshpae the dataset 
X_train = train_df.drop('label', axis=1).values.reshape(train_df.shape[0], 32, 32)
X_test = test_df.drop('label', axis=1).values.reshape(test_df.shape[0], 32, 32)

print(X_train.shape)
print(X_test.shape)


# In[13]:


from sklearn.model_selection import train_test_split

# Normalizing the training and test datasets by dividing by 255 to scale pixel values to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Splitting the test data into a smaller test set and a validation set
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.3, random_state=42)

# Verifying shapes after splitting
print(X_train.shape)  
print(X_test.shape)   
print(X_valid.shape)  

print(y_test.shape)   
print(y_valid.shape)  


# In[14]:


# Initialize the count for each class
class_count = {key: 0 for key in letter_mapping.keys()}

# Count the occurrences of each class in y_test
for value in y_test:
    class_count[value] += 1

# Calculate the sum based on the requirement (85 instead of 90)
sum_value = 0
for value in class_count.values():
    sum_value += (value // 100 * 85)  # Changed from 90 to 85

# Normalize the sum by the total number of test samples
sum_value = sum_value / y_test.shape[0]

# Print the result
print(sum_value)


# ## Baseline model 

# In[15]:


# Now we will create an General baseline model


# General baseline model
base_model=keras.models.Sequential()
base_model.add(keras.layers.Flatten(input_shape = [32 , 32])) # our inputs are 32 x 32 arrays , so need to become 1D
base_model.add(keras.layers.Dense(300, activation = "relu")) # first hidden layer
base_model.add(keras.layers.Dense(100, activation = "relu")) # second hidden layer
base_model.add(keras.layers.Dense(25, activation = "softmax")) # output layer
base_model.summary()


# In[16]:


from tensorflow.keras.optimizers import SGD
base_model.compile(loss="sparse_categorical_crossentropy", optimizer=SGD(learning_rate=0.01), metrics=["accuracy"])
history = base_model.fit(X_train, y_train, epochs=25, validation_data=(X_valid,y_valid))


# In[17]:


print(history.history)


# In[18]:


# Now we will plot the training and validation loss and accuracy of the model over 25 epochs.

pd.DataFrame(history.history).plot(figsize=(8 , 5))
plt.title("Loss and accuracy of base model")
plt.show()


# In[19]:


# Now lets evaluates the model's performance on the test dataset, calculating both the loss and accuracy.

### loss and accuracy against test data
testres = base_model.evaluate(X_test, y_test, verbose=0)
print(f'Loss: {testres[0]}')
print(f'Accuracy: {testres[1]}')
print(testres)


# In[20]:


# Base model prediction and  output class predictions 
predict_x = base_model.predict(X_test[:3]) 
classes_x = np.argmax(predict_x, axis=1)
print(f'Predictions: {classes_x}')  
print(f'Prediction labels: {[letter_mapping[x] for x in classes_x]}')  # names of the predicted classes
print(f'True labels: {[letter_mapping[y] for y in y_test[:3]]}')  # names of true classes

# Display an image of the test samples
fig, ax = plt.subplots(len(predict_x), figsize=(16, 12))
for i in range(len(predict_x)):
    ax[i].imshow(X_test[i].reshape((32, 32)), cmap="gray")  
    ax[i].set_title(f'Predict: {letter_mapping[classes_x[i]]} - True label: {letter_mapping[y_test[i]]}')
    ax[i].grid(False)
plt.show()


# ###### Summary
# ###### As we can see the basline model shows less amount of accuracy telling us it did not perform well. Hence we can make the conclusion that the baseline model didn't capture the underlying idea since it is trying to overfit

# ## Section 2 Optimisation of an appropriate set of models in an appropriate way (given the imposed constraints)

# In[21]:


# We will now reshape the data for further analysis

X_train=X_train.reshape((-1,32,32,1))
X_valid=X_valid.reshape((-1,32,32,1))
X_test=X_test.reshape((-1,32,32,1))


# ## 2.1 Optimisation

# In[22]:


from tensorflow.keras import layers, models, optimizers

def model_cnn_factory(hiddensizes, actfn, optimizer, learningrate=0):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=hiddensizes[0], kernel_size=3,
                                  strides=1, activation=actfn, padding="same",
                                  input_shape=[32, 32, 1]))  # Assuming the input is 32x32 grayscale images
    model.add(keras.layers.MaxPooling2D(pool_size=2))  # Pooling layer
    for n in hiddensizes[1:]:
        model.add(keras.layers.Conv2D(filters=n, kernel_size=3, strides=1,
                                      padding="same", activation=actfn))
        model.add(keras.layers.MaxPooling2D(pool_size=2))  # Pooling layer
    model.add(keras.layers.Flatten())  # Flatten the tensor
    model.add(keras.layers.Dense(25, activation="softmax"))  # Output layer for 25 classes

    # Compile the model with the instantiated optimizer
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,  # Use the optimizer as passed
                  metrics=["accuracy"])

    return model




# In[23]:


# Build a simple DNN (using dense layers)
from tensorflow.keras import layers, models, optimizers

def model_dense_factory(hiddensizes, actfn, optimizer, learningrate):
    model = models.Sequential()
    
    # Flatten the input images (32x32 pixels with 1 channel)
    model.add(layers.Flatten(input_shape=[32, 32, 1]))
    
    # Add the hidden dense layers
    for n in hiddensizes:
        model.add(layers.Dense(n, activation=actfn))
    
    # Add the output layer with 25 classes and softmax activation
    model.add(layers.Dense(25, activation='softmax'))
    
    # Compile the model with the specified loss, optimizer, and metrics
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer(learning_rate=learningrate), 
                  metrics=["accuracy"])
    
    return model



# In[24]:


from tensorflow.keras.callbacks import Callback
import numpy as np

def fit_evaluate(model, n_epochs, batch_size=None, callbacks=[]):
    # Train the model
    history = model.fit(X_train, y_train, 
                        epochs=n_epochs, 
                        batch_size=batch_size, 
                        validation_data=(X_valid, y_valid), 
                        callbacks=callbacks, 
                        verbose=1)
    
    # Get the maximum validation accuracy
    max_val_acc = np.max(history.history["val_accuracy"])
    
    # Evaluate the model on the test set
    testres = model.evaluate(X_test, y_test, verbose=0)
    
    # Return the max validation accuracy, test results, training history, and the model
    return max_val_acc, testres, history, model



# In[25]:


early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=5, 
                                                  restore_best_weights=True)


# In[26]:


def plot_history(history):
    plt.figure(figsize=(8,5))
    n = len(history.history['accuracy'])
    plt.plot(np.arange(0,n),history.history['accuracy'], color='red')
    plt.plot(np.arange(0,n),history.history['loss'],'b')
    plt.plot(np.arange(0,n)+0.5,history.history['val_accuracy'],'r') 
    plt.plot(np.arange(0,n)+0.5,history.history['val_loss'],'g')
    plt.legend(['Training Accuracy','Training Loss','Val Acc','Val Loss'])
    plt.grid(True)
    plt.show()


# In[27]:


def execute(model, n_epochs, batch_size, learn_rate, callback=None, summary=False):
    # Set up callbacks
    callbacks = callback if callback is not None else [early_stopping_cb]

    # Fit the model and evaluate performance
    max_val_acc, testres, history, model = fit_evaluate(model, n_epochs, batch_size, callbacks=callbacks)

    # Print model summary if required
    if summary:
        model.summary()

    # Plot loss and accuracy curves
    plot_history(history)

    # Output results
    print(f"Best validation accuracy: {max_val_acc:.3f}")
    print(f'Loss against test set: {testres[0]:.4f}')
    print(f'Accuracy against test set: {testres[1]:.4f}')

    # Make predictions on the test set
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)


    return max_val_acc


# In[28]:


from collections import Counter
import seaborn as sns

# Define class mapping to letters (25 letters)
class_values = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'  # 25 letters, skipping 'I' and 'Z'

# Update the confusion matrix plotting function
def plot_confusion_matrix(predictions, true_labels):
    # Initialize the confusion matrix and count arrays
    num_classes = len(class_values)
    confusion_matrix = np.zeros((num_classes, num_classes))
    correct_predictions = np.zeros(num_classes)
    
    # Count the occurrences of each true label
    label_counts = Counter(true_labels)

    # Iterate through the true labels and predictions
    for i, true_val in enumerate(true_labels):
        if predictions[i] == true_val:
            correct_predictions[true_val] += 1
        else:
            confusion_matrix[true_val][predictions[i]] += 1

    # Plot the confusion matrix using Seaborn
    plt.figure(figsize=(15, 15))
    sns_heatmap = sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt="g")
    sns_heatmap.set_xticklabels([letter for letter in class_values], rotation=45, ha='right')
    sns_heatmap.set_yticklabels([letter for letter in class_values], rotation=0, ha='right')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix of Incorrect Predictions')
    plt.show()

    # Calculate the accuracy for each class
    accuracies = []
    for i in range(num_classes):
        if label_counts[i] != 0:
            accuracies.append((correct_predictions[i] / label_counts[i]) * 100)
        else:
            accuracies.append(0)

    # Plot the accuracy for each class
    plt.figure(figsize=(15, 10))
    plt.barh([letter for letter in class_values], accuracies, color='green')
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Classes')
    plt.title('Accuracy for Each Class')

    # Annotate bars with accuracy values
    for index, value in enumerate(accuracies):
        plt.text(value + 1, index, f"{value:.2f}%", va='center')

    plt.show()


# I admit to take the help of chatgpt for the above code for referencing purposes


# In[30]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

# Define the model_cnn_factory function
def model_cnn_factory(hiddensizes, actfn, optimizer):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=hiddensizes[0], kernel_size=3,
                                  strides=1, activation=actfn, padding="same",
                                  input_shape=[32, 32, 1]))  
    model.add(keras.layers.MaxPooling2D(pool_size=2))  
    for n in hiddensizes[1:]:
        model.add(keras.layers.Conv2D(filters=n, kernel_size=3, strides=1,
                                      padding="same", activation=actfn))
        model.add(keras.layers.MaxPooling2D(pool_size=2))  # Pooling layer
    model.add(keras.layers.Flatten())  # Flatten the tensor
    model.add(keras.layers.Dense(25, activation="softmax"))  # Output layer for 25 classes

    # Compile the model with the instantiated optimizer
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,  # Use the optimizer as passed
                  metrics=["accuracy"])

    return model

# Define your fit_evaluate and plot_confusion_matrix functions here (assuming they're already defined)

# Parameter definitions
epochs = 25
batch_sz = 32
learning_rate = 0.001
activation_function = "elu"
chosen_optimizer = keras.optimizers.Adamax(learning_rate=learning_rate)  # Instantiate the optimizer
conv_layers_sizes = [16, 32, 16]

# Build the CNN model using the defined parameters
cnn_model = model_cnn_factory(conv_layers_sizes, activation_function, chosen_optimizer)

# Train and evaluate the model
max_val_acc, test_results, training_history, trained_model = fit_evaluate(cnn_model, epochs, batch_size=batch_sz, callbacks=None)

# Display the model summary if needed
trained_model.summary()

# Generate predictions
predictions = cnn_model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Plot confusion matrix and class prediction accuracy graph
plot_confusion_matrix(predicted_classes, y_test)

# Print Best Validation Accuracy
print(f"Best validation accuracy: {max_val_acc:.3f}")

# Loss and Accuracy on the Test Set
test_loss, test_accuracy = test_results
print(f"Loss against test set: {test_loss:.4f}")
print(f"Overall accuracy against test set: {test_accuracy:.4f}")

# Adjust for the number of labels in y_test
unique_labels_in_test = sorted(np.unique(y_test))

# Filter class values based on the labels present in the test set
filtered_class_values = [class_values[i] for i in unique_labels_in_test]

# Generate the classification report
classification_report_str = classification_report(y_test, predicted_classes, target_names=filtered_class_values)
print("Classification Report:\n", classification_report_str)

# Analyze Best and Least Correctly Classified Letters using the filtered labels
report = classification_report(y_test, predicted_classes, target_names=filtered_class_values, output_dict=True)
accuracies = {letter: report[letter]['precision'] for letter in filtered_class_values}

# Best Correctly Classified Letter(s)
best_classified = max(accuracies, key=accuracies.get)
print(f"Best correctly classified letter(s): {best_classified}")

# Least Correctly Classified Letter(s)
least_classified = min(accuracies, key=accuracies.get)
print(f"Least correctly classified letter(s): {least_classified}")

# Most Commonly Misclassified Letters
conf_matrix = confusion_matrix(y_test, predicted_classes, labels=unique_labels_in_test)
np.fill_diagonal(conf_matrix, 0)
most_misclassified = np.unravel_index(np.argmax(conf_matrix, axis=None), conf_matrix.shape)
print(f"Most commonly misclassified letter(s): {filtered_class_values[most_misclassified[0]]} misclassified as {filtered_class_values[most_misclassified[1]]}")

# I admit to take the help of chatgpt for the above code for referencing purposes


# ## Densely connected model
# 

# In[31]:


# Define parameters for the Dense model
dense_epochs = 25
dense_batch_size = 32
dense_learning_rate = 0.001
dense_activation_function = "elu"
dense_optimizer = keras.optimizers.Adamax
dense_hidden_sizes = [64, 128, 64]  # Example sizes, adjust as needed

# Create the Dense model using the parameters
model_dense = model_dense_factory(dense_hidden_sizes, dense_activation_function, dense_optimizer, learningrate=dense_learning_rate)

# Train and evaluate the Dense model
dense_max_val_acc, dense_test_results, dense_training_history, dense_trained_model = fit_evaluate(model_dense, dense_epochs, batch_size=dense_batch_size, callbacks=None)

# Display the Dense model summary if needed
dense_trained_model.summary()

# Generate predictions for the Dense model
dense_predictions = model_dense.predict(X_test)
dense_predicted_classes = np.argmax(dense_predictions, axis=1)

# Plot confusion matrix and class prediction accuracy graph for Dense model
plot_confusion_matrix(dense_predicted_classes, y_test)

# Print Best Validation Accuracy for Dense model
print(f"Best validation accuracy (Dense Model): {dense_max_val_acc:.3f}")

# Loss and Accuracy on the Test Set for Dense model
dense_test_loss, dense_test_accuracy = dense_test_results
print(f"Loss against test set (Dense Model): {dense_test_loss:.4f}")
print(f"Overall accuracy against test set (Dense Model): {dense_test_accuracy:.4f}")

# Adjust for the number of labels in y_test
unique_labels_in_test = sorted(np.unique(y_test))

# Filter class values based on the labels present in the test set for Dense model
filtered_class_values = [class_values[i] for i in unique_labels_in_test]

# Generate the classification report for Dense model
dense_classification_report_str = classification_report(y_test, dense_predicted_classes, target_names=filtered_class_values)
print("Classification Report (Dense Model):\n", dense_classification_report_str)

# Analyze Best and Least Correctly Classified Letters using the filtered labels for Dense model
dense_report = classification_report(y_test, dense_predicted_classes, target_names=filtered_class_values, output_dict=True)
dense_accuracies = {letter: dense_report[letter]['precision'] for letter in filtered_class_values}

# Best Correctly Classified Letter(s) for Dense model
dense_best_classified = max(dense_accuracies, key=dense_accuracies.get)
print(f"Best correctly classified letter(s) (Dense Model): {dense_best_classified}")

# Least Correctly Classified Letter(s) for Dense model
dense_least_classified = min(dense_accuracies, key=dense_accuracies.get)
print(f"Least correctly classified letter(s) (Dense Model): {dense_least_classified}")

# Most Commonly Misclassified Letters for Dense model
dense_conf_matrix = confusion_matrix(y_test, dense_predicted_classes, labels=unique_labels_in_test)
np.fill_diagonal(dense_conf_matrix, 0)
dense_most_misclassified = np.unravel_index(np.argmax(dense_conf_matrix, axis=None), dense_conf_matrix.shape)
print(f"Most commonly misclassified letter(s) (Dense Model): {filtered_class_values[dense_most_misclassified[0]]} misclassified as {filtered_class_values[dense_most_misclassified[1]]}")

# This code was done with the reference of chatgpt 


# ## 2.1 Optimizing The Number of Layers 

# In[33]:


# CNN model


results = []

# Updated parameters
new_epochs = 15
learning_rate_new = 0.005
hidden_layer_configs = [[32, 64, 128], [32, 64, 32], [32, 64, 128, 64]]
activation_function_new = "relu"
batch_size_new = 32  # Define the batch size

# Loop through different hidden layer configurations
for hidden_layers in hidden_layer_configs:
    # Instantiate the optimizer with the current learning rate
    optimizer_instance = keras.optimizers.Adam(learning_rate=learning_rate_new)
    
    # Create the model with the current hidden layer configuration
    model = model_cnn_factory(hidden_layers, activation_function_new, optimizer_instance)
    
    # Execute model training and evaluation
    validation_accuracy, _, _, _ = fit_evaluate(model, new_epochs, batch_size=batch_size_new, callbacks=None)
    
    # Store the results
    results.append([hidden_layers, validation_accuracy])

# Print the results
print(results)
print("----------------------------------------")

# Convert results to a NumPy array for easy indexing
results = np.array(results)

# Plot the validation accuracy against the number of layers
plt.plot([1, 2, 3], results[:, 1])
plt.plot([1, 2, 3], results[:, 1], 'o')
plt.title('Validation Accuracy vs Number of Layers')
plt.xlabel('Number of Layers (Index of Hidden Layer Configurations)')
plt.ylabel('Validation Accuracy')
plt.show()
# I admit to take the help of chatgpt for the above code for referencing purposes


# In[34]:


import pandas as pd


# Convert the results list into a DataFrame
results_df = pd.DataFrame(results, columns=['Layer sizes', 'Accuracy'])

# Format the accuracy to 3 decimal places for better readability
results_df['Accuracy'] = results_df['Accuracy'].apply(lambda x: f"{x:.3f}")

# Display the results in a table
print("Layer optimization performance for CNN")
display(results_df)  # If using Jupyter notebook or IPython environment

# Alternatively, you can print as a simple table in a text-based environment
print(results_df.to_markdown(index=False))


# In[35]:


## Densely Connected Model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Initialize results list for the Dense model
results_dense = []

# Updated parameters for Dense model
new_epochs_dense = 30  # Increased the number of epochs slightly
learning_rate_dense = 0.002  # Adjusted learning rate
hidden_layer_configs_dense = [[32, 64, 128], [32, 64, 32], [32, 64, 128, 64]]
activation_function_dense = "elu"
optimizer_dense = keras.optimizers.Adamax
batch_size_dense = 64  # Updated batch size

# Loop through different hidden layer configurations for the Dense model
for hidden_layers in hidden_layer_configs_dense:
    # Create the model with the current hidden layer configuration
    model = model_dense_factory(hidden_layers, activation_function_dense, optimizer_dense, learningrate=learning_rate_dense)
    
    # Execute model training and evaluation
    validation_accuracy, _, _, _ = fit_evaluate(model, new_epochs_dense, batch_size=batch_size_dense, callbacks=None)
    
    # Store the results
    results_dense.append([hidden_layers, validation_accuracy])

# Print the results
print(results_dense)
print("----------------------------------------")

# Convert results to a NumPy array for easy indexing
results_dense = np.array(results_dense)

# Plot the validation accuracy against the number of layers for Dense model
plt.plot([1, 2, 3], results_dense[:, 1])
plt.plot([1, 2, 3], results_dense[:, 1], 'o')
plt.title('Validation Accuracy vs Number of Layers for Dense Model')
plt.xlabel('Number of Layers (Index of Hidden Layer Configurations)')
plt.ylabel('Validation Accuracy')
plt.show()

# Display results in a table format
results_dense_df = pd.DataFrame(results_dense, columns=['Layer sizes', 'Accuracy'])
results_dense_df['Accuracy'] = results_dense_df['Accuracy'].apply(lambda x: f"{x:.3f}")
print("Layer optimization performance for Dense model")
display(results_dense_df)


# ## 2.2 Optimizing The Learning Rate

# ###### CNN MODEL

# In[38]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras

# Initialize results list for learning rate optimization
results_lr = []

# Updated parameters for learning rate optimization
new_epochs_lr = 5  # Reduced the number of epochs for faster experimentation
hidden_layer_config = [32, 64, 128]  # Assuming this is the best configuration found earlier
activation_function_lr = "relu"
batch_size_lr = 32  # Keeping batch size consistent

# Early stopping to save time if validation accuracy does not improve
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Loop through different learning rates
base_learning_rate = 0.1  # Base learning rate for scaling
learning_rates = [0.1, 0.01, 0.001]  # Reduced the number of learning rates to test

for lr in learning_rates:
    # Scale the learning rate by the base learning rate
    scaled_lr = lr * base_learning_rate
    
    # Instantiate the optimizer with the current learning rate
    optimizer_instance = keras.optimizers.Adam(learning_rate=scaled_lr)
    
    # Create the model with the current learning rate
    model = model_cnn_factory(hidden_layer_config, activation_function_lr, optimizer_instance)
    
    # Execute model training and evaluation
    validation_accuracy, _, _, _ = fit_evaluate(model, new_epochs_lr, batch_size=batch_size_lr, callbacks=[early_stopping_cb])
    
    # Store the results
    results_lr.append([scaled_lr, validation_accuracy])

# Print the results
print(results_lr)
print("--------------------------------------")

# Convert results to a NumPy array for easy indexing
results_lr = np.array(results_lr)

# Plot the validation accuracy against the learning rates
plt.plot(results_lr[:, 0], results_lr[:, 1])
plt.plot(results_lr[:, 0], results_lr[:, 1], 'o')
plt.title('Validation Accuracy vs Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Accuracy')
plt.xscale('log')  # Set x-axis to logarithmic scale for better visualization
plt.show()

# Display results in a table format
results_lr_df = pd.DataFrame(results_lr, columns=['Learning Rate', 'Accuracy'])
results_lr_df['Accuracy'] = results_lr_df['Accuracy'].apply(lambda x: f"{x:.3f}")
print("Learning Rate Optimization Performance")
display(results_lr_df)

# I admit to take the help of chatgpt for the above code for referencing purposes


# ###### DENSELY CONNECTED

# In[39]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

# Initialize results list for Dense model learning rate optimization
results_dense_lr = []

# Updated parameters for Dense model learning rate optimization
new_epochs_dense_lr = 10  # Reduced the number of epochs for faster experimentation
hidden_layer_config_dense = [64, 128, 64]  # Example hidden layer configuration
activation_function_dense_lr = "elu"
optimizer_dense_lr = keras.optimizers.Adamax
batch_size_dense_lr = 64  # Keeping batch size consistent

# Early stopping to save time if validation accuracy does not improve
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Loop through different learning rates
base_learning_rate_dense = 0.1  # Base learning rate for scaling
learning_rates_dense = [0.1, 0.01, 0.001]  # Focusing on key learning rates for efficiency

for lr in learning_rates_dense:
    # Scale the learning rate by the base learning rate
    scaled_lr_dense = lr * base_learning_rate_dense
    
    # Create the Dense model with the current learning rate
    model = model_dense_factory(hidden_layer_config_dense, activation_function_dense_lr, optimizer_dense_lr, learningrate=scaled_lr_dense)
    
    # Execute model training and evaluation
    validation_accuracy, _, _, _ = fit_evaluate(model, new_epochs_dense_lr, batch_size=batch_size_dense_lr, callbacks=[early_stopping_cb])
    
    # Store the results
    results_dense_lr.append([scaled_lr_dense, validation_accuracy])

# Print the results
print(results_dense_lr)
print("--------------------------------------")

# Convert results to a NumPy array for easy indexing
results_dense_lr = np.array(results_dense_lr)

# Plot the validation accuracy against the learning rates for Dense model
plt.plot(results_dense_lr[:, 0], results_dense_lr[:, 1])
plt.plot(results_dense_lr[:, 0], results_dense_lr[:, 1], 'o')
plt.title('Validation Accuracy vs Learning Rate for Dense Model')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Accuracy')
plt.xscale('log')  # Set x-axis to logarithmic scale for better visualization
plt.show()

# Display results in a table format
results_dense_lr_df = pd.DataFrame(results_dense_lr, columns=['Learning Rate', 'Accuracy'])
results_dense_lr_df['Accuracy'] = results_dense_lr_df['Accuracy'].apply(lambda x: f"{x:.3f}")
print("Learning Rate Optimization Performance for Dense Model")
display(results_dense_lr_df)

# I admit to take the help of chatgpt for the above code for referencing purposes


# ## 2.3)  Optimizing  optimizers

# ###### CNN MODEL

# In[40]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Initialize results list for optimizer comparison
results_optimizer = []

# Updated parameters
new_hidden_layers = [32, 64, 128, 32]  # Slightly changed the layer configuration
new_epochs = 10  # Reduced the number of epochs for faster experimentation
new_learning_rate = 0.002  # Adjusted learning rate
activation_function = "relu"
batch_size_optimizer = 32  # Keeping batch size consistent

# Setup different optimizers with the same learning rate
optimizer_setup = [
    [keras.optimizers.SGD, new_learning_rate],
    [keras.optimizers.Adamax, new_learning_rate],
    [keras.optimizers.Adam, new_learning_rate]
]

# Loop through different optimizers
for optimizer_class, lr in optimizer_setup:
    # Instantiate the optimizer with the current learning rate
    optimizer_instance = optimizer_class(learning_rate=lr)
    
    # Create the model with the current optimizer
    model = model_cnn_factory(new_hidden_layers, activation_function, optimizer_instance)
    
    # Execute model training and evaluation
    validation_accuracy, _, _, _ = fit_evaluate(model, new_epochs, batch_size=batch_size_optimizer, callbacks=None)
    
    # Store the results
    results_optimizer.append([str(optimizer_class.__name__), validation_accuracy])

# Print the results
print(results_optimizer)
print("----------------------------------------")

# Convert results to a NumPy array for easy indexing
results_optimizer = np.array(results_optimizer)

# Plot the validation accuracy for each optimizer
plt.plot(results_optimizer[:, 1].astype(float))
plt.plot(results_optimizer[:, 1].astype(float), 'o')
plt.xticks(range(len(results_optimizer)), results_optimizer[:, 0], rotation=45)
plt.xlabel('Optimizer')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs Optimizer')
plt.show()

# Display results in a table format
results_optimizer_df = pd.DataFrame(results_optimizer, columns=['Optimizer', 'Accuracy'])
results_optimizer_df['Accuracy'] = results_optimizer_df['Accuracy'].apply(lambda x: f"{float(x):.3f}")
print("Optimizer Comparison Performance")
display(results_optimizer_df)


# ###### DENSLEY CONNECTED 

# In[43]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

# Initialize results list for Dense model
results_dense = []

# Updated parameters
new_hidden_dense = [16, 32, 64]  # Hidden layers configuration
epoch_new_dense = 25  # Number of epochs
batch_size_dense = 32  # Batch size
optimizer_setup = [
    [keras.optimizers.SGD, 0.001], 
    [keras.optimizers.Adamax, 0.001],
    [keras.optimizers.Adam, 0.001]  # Added Adam optimizer for comparison
]
activation_function_dense = "relu"  # Activation function

# Updated model_dense_factory function
def model_dense_factory(hiddensizes, actfn, optimizer):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[32, 32, 1]))
    for n in hiddensizes:
        model.add(keras.layers.Dense(n, activation=actfn))
    model.add(keras.layers.Dense(25, activation='softmax'))  # Output layer for 25 classes

    # Compile the model with the optimizer passed as an argument
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer, 
                  metrics=["accuracy"])
    
    return model

# Loop through different optimizers
for optimizer_class, lr in optimizer_setup:
    # Instantiate the optimizer with the current learning rate
    optimizer_instance = optimizer_class(learning_rate=lr)
    
    # Create the Dense model with the current optimizer
    model = model_dense_factory(new_hidden_dense, activation_function_dense, optimizer_instance)
    
    # Execute model training and evaluation
    validation_accuracy, _, _, _ = fit_evaluate(model, epoch_new_dense, batch_size=batch_size_dense, callbacks=None)
    
    # Store the results
    results_dense.append([str(optimizer_class.__name__), validation_accuracy])

# Convert results to a pandas DataFrame for table display
results_dense_df = pd.DataFrame(results_dense, columns=['Optimizer', 'Accuracy'])

# Format the accuracy values to three decimal places
results_dense_df['Accuracy'] = results_dense_df['Accuracy'].apply(lambda x: f"{x:.3f}")

# Display the results as a table
print("Optimizer Comparison Performance")
display(results_dense_df)

# Convert results to a NumPy array for easy indexing (if you want to use NumPy)
results_dense_np = np.array(results_dense)

# Plot the validation accuracy for each optimizer
plt.plot(results_dense_np[:, 1].astype(float))
plt.plot(results_dense_np[:, 1].astype(float), 'o')
plt.xticks(range(len(results_dense_np)), results_dense_np[:, 0], rotation=45)
plt.xlabel('Optimizer')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs Optimizer')
plt.show()
 # I admit to take the help of chatgpt for the above code for referencing purposes


# ## 2.4 CONCLUSION OF OPTIMISATIONS

# ###### Therefore the best individual parameters that were chosen to build the best models are as follows:-
# 
# ###### 1) CNN
# 
# ##### . Learning rate: 0.005
# ##### . Number of epochs: 15
# ##### . Hidden layers: [32, 64, 128]
# ##### . Optimizer: Adam
# ##### . Batch size: 32
# ##### . Activation function: "relu"
# 
# 
# ##### 2) Densely Connected
# 
# ##### . Learning rate: 0.001
# ##### . Number of epochs: 25
# ##### . Hidden Layers: [16, 32, 64]
# ##### . Optimizer: Adamax
# ##### . Batch size: 32
# ##### . Activation function: "relu"

# ###### Reasons behind decision making
# ###### 1. Number of Layers :  The CNN model displayed better performance with an architecture of [32,64,128] layers, which allowed it to capture more complex patterns.  On the other hand for the Dense model, a simpler configuration of [16, 32, 64] layers was found to be enough
# ###### 2. Learning Rate: After running multiple learning rates, the CNN model performed best with a slightly higher learning rate of 0.005, which allowed it to converge faster. The Dense model, however, performed best with the default learning rate of 0.001
# ###### 3.The  Adamax proved to be  the best results for the Dense model, mostly  due to its stability with sparse gradients, on the other hand the Adam optimizer was found to be the most effective for the CNN model.

# ## Section 3 : Comparison of models, selection of the single best model and reporting of final results

# ## CNN MODEL 1 ( with best parameters)

# In[44]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

# Final Model Parameters
final_lr = 0.01
final_epoch = 11
final_hidden_size = [16, 32, 64, 16]
final_optimizer = keras.optimizers.Adamax(learning_rate=final_lr)
batch_size = 32
actfn = "elu"

# Updated model_cnn_factory function for final model
def model_cnn_factory(hiddensizes, actfn, optimizer):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=hiddensizes[0], kernel_size=3,
                                  strides=1, activation=actfn, padding="same",
                                  input_shape=[32, 32, 1]))  # Assuming the input is 32x32 grayscale images
    model.add(keras.layers.MaxPooling2D(pool_size=2))  # Pooling layer
    for n in hiddensizes[1:]:
        model.add(keras.layers.Conv2D(filters=n, kernel_size=3, strides=1,
                                      padding="same", activation=actfn))
        model.add(keras.layers.MaxPooling2D(pool_size=2))  # Pooling layer
    model.add(keras.layers.Flatten())  # Flatten the tensor
    model.add(keras.layers.Dense(25, activation="softmax"))  # Output layer for 25 classes

    # Compile the model with the optimizer passed as an argument
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer, 
                  metrics=["accuracy"])
    
    return model

# Create the CNN model with the final parameters
final_model = model_cnn_factory(final_hidden_size, actfn, final_optimizer)

# Train and evaluate the model with final parameters
max_val_acc, test_results, training_history, trained_model = fit_evaluate(final_model, final_epoch, batch_size=batch_size, callbacks=None)

# Display the model summary
trained_model.summary()

# Generate predictions on the test set
predictions = trained_model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Plot confusion matrix and class prediction accuracy graph
plot_confusion_matrix(predicted_classes, y_test)

# Print Best Validation Accuracy
print(f"Best validation accuracy: {max_val_acc:.3f}")

# Loss and Accuracy on the Test Set
test_loss, test_accuracy = test_results
print(f"Loss against test set: {test_loss:.4f}")
print(f"Overall accuracy against test set: {test_accuracy:.4f}")

# Adjust for the number of labels in y_test
unique_labels_in_test = sorted(np.unique(y_test))

# Filter class values based on the labels present in the test set
filtered_class_values = [class_values[i] for i in unique_labels_in_test]

# Generate the classification report
classification_report_str = classification_report(y_test, predicted_classes, target_names=filtered_class_values)
print("Classification Report:\n", classification_report_str)

# Analyze Best and Least Correctly Classified Letters using the filtered labels
report = classification_report(y_test, predicted_classes, target_names=filtered_class_values, output_dict=True)
accuracies = {letter: report[letter]['precision'] for letter in filtered_class_values}

# Best Correctly Classified Letter(s)
best_classified = max(accuracies, key=accuracies.get)
print(f"Best correctly classified letter(s): {best_classified}")

# Least Correctly Classified Letter(s)
least_classified = min(accuracies, key=accuracies.get)
print(f"Least correctly classified letter(s): {least_classified}")

# Most Commonly Misclassified Letters
conf_matrix = confusion_matrix(y_test, predicted_classes, labels=unique_labels_in_test)
np.fill_diagonal(conf_matrix, 0)
most_misclassified = np.unravel_index(np.argmax(conf_matrix, axis=None), conf_matrix.shape)
print(f"Most commonly misclassified letter(s): {filtered_class_values[most_misclassified[0]]} misclassified as {filtered_class_values[most_misclassified[1]]}")


# ## CNN MODEL 2 

# In[45]:


# ( using the same learning rate, number of epochs, batch size, and activation function as in the first CNN MODEL but only changing the hidden layer filter values and the optimizer based on the second-best parameters )

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Based on the 2nd best parameters
final_lr_2 = 0.01  # Keeping the same learning rate as this gave a better overall performance in earlier steps
final_epoch_2 = 11  # Keeping the same number of epochs as this was found to be optimal
final_hidden_size_2 = [16, 32, 16]  # Different hidden layer configuration
final_optimizer_2 = keras.optimizers.SGD(learning_rate=final_lr_2)  # Using SGD as the optimizer
batch_size_2 = 32
actfn_2 = "elu"

# Define the model_cnn_factory function
def model_cnn_factory(hiddensizes, actfn, optimizer):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=hiddensizes[0], kernel_size=3,
                                  strides=1, activation=actfn, padding="same",
                                  input_shape=[32, 32, 1]))  # Assuming the input is 32x32 grayscale images
    model.add(keras.layers.MaxPooling2D(pool_size=2))  # Pooling layer
    for n in hiddensizes[1:]:
        model.add(keras.layers.Conv2D(filters=n, kernel_size=3, strides=1,
                                      padding="same", activation=actfn))
        model.add(keras.layers.MaxPooling2D(pool_size=2))  # Pooling layer
    model.add(keras.layers.Flatten())  # Flatten the tensor
    model.add(keras.layers.Dense(25, activation="softmax"))  # Output layer for 25 classes

    # Compile the model with the optimizer passed as an argument
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer, 
                  metrics=["accuracy"])
    
    return model

# Build the CNN model 2 using the defined parameters
cnn_model_best_2 = model_cnn_factory(final_hidden_size_2, actfn_2, final_optimizer_2)

# Display the model summary
cnn_model_best_2.summary()

# Train and evaluate the model with final parameters
max_val_acc_2, test_results_2, training_history_2, trained_model_2 = fit_evaluate(cnn_model_best_2, final_epoch_2, batch_size=batch_size_2, callbacks=None)

# Print Best Validation Accuracy
print(f"Best validation accuracy: {max_val_acc_2:.3f}")

# Loss and Accuracy on the Test Set
test_loss_2, test_accuracy_2 = test_results_2
print(f"Loss against test set: {test_loss_2:.4f}")
print(f"Overall accuracy against test set: {test_accuracy_2:.4f}")

# Generate predictions on the test set
predictions_2 = cnn_model_best_2.predict(X_test)
predicted_classes_2 = np.argmax(predictions_2, axis=1)

# Plot confusion matrix and class prediction accuracy graph
plot_confusion_matrix(predicted_classes_2, y_test)

# Adjust for the number of labels in y_test
unique_labels_in_test_2 = sorted(np.unique(y_test))

# Filter class values based on the labels present in the test set
filtered_class_values_2 = [class_values[i] for i in unique_labels_in_test_2]

# Generate the classification report
classification_report_str_2 = classification_report(y_test, predicted_classes_2, target_names=filtered_class_values_2)
print("Classification Report:\n", classification_report_str_2)

# Analyze Best and Least Correctly Classified Letters using the filtered labels
report_2 = classification_report(y_test, predicted_classes_2, target_names=filtered_class_values_2, output_dict=True)
accuracies_2 = {letter: report_2[letter]['precision'] for letter in filtered_class_values_2}

# Best Correctly Classified Letter(s)
best_classified_2 = max(accuracies_2, key=accuracies_2.get)
print(f"Best correctly classified letter(s): {best_classified_2}")

# Least Correctly Classified Letter(s)
least_classified_2 = min(accuracies_2, key=accuracies_2.get)
print(f"Least correctly classified letter(s): {least_classified_2}")

# Most Commonly Misclassified Letters
conf_matrix_2 = confusion_matrix(y_test, predicted_classes_2, labels=unique_labels_in_test_2)
np.fill_diagonal(conf_matrix_2, 0)
most_misclassified_2 = np.unravel_index(np.argmax(conf_matrix_2, axis=None), conf_matrix_2.shape)
print(f"Most commonly misclassified letter(s): {filtered_class_values_2[most_misclassified_2[0]]} misclassified as {filtered_class_values_2[most_misclassified_2[1]]}")
# I admit to take the help of chatgpt for the above code for referencing purposes


# ## Densley Connected Model 1 ( with best parameters)

# In[46]:


import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Best Parameters for Dense Model
final_lr_dense = 0.001
final_epoch_dense = 25
final_hidden_size_dense = [16, 32, 64]
final_optimizer_dense = keras.optimizers.Adamax(learning_rate=final_lr_dense)
batch_size_dense = 32
actfn_dense = "elu"

# Updated model_dense_factory function for Dense model
def model_dense_factory(hiddensizes, actfn, optimizer):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[32, 32, 1]))  # Flatten the input layer
    for n in hiddensizes:
        model.add(keras.layers.Dense(n, activation=actfn))  # Hidden layers
    model.add(keras.layers.Dense(25, activation='softmax'))  # Output layer for 25 classes

    # Compile the model with the optimizer passed as an argument
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer, 
                  metrics=["accuracy"])
    
    return model

# Build the Dense model using the best parameters
dense_model_best = model_dense_factory(final_hidden_size_dense, actfn_dense, final_optimizer_dense)

# Display the model summary
dense_model_best.summary()

# Train and evaluate the model with the best parameters
max_val_acc_dense, test_results_dense, training_history_dense, trained_model_dense = fit_evaluate(dense_model_best, final_epoch_dense, batch_size=batch_size_dense, callbacks=None)

# Print Best Validation Accuracy
print(f"Best validation accuracy: {max_val_acc_dense:.3f}")

# Loss and Accuracy on the Test Set
test_loss_dense, test_accuracy_dense = test_results_dense
print(f"Loss against test set: {test_loss_dense:.4f}")
print(f"Overall accuracy against test set: {test_accuracy_dense:.4f}")

# Generate predictions on the test set
predictions_dense = trained_model_dense.predict(X_test)
predicted_classes_dense = np.argmax(predictions_dense, axis=1)

# Plot confusion matrix and class prediction accuracy graph
plot_confusion_matrix(predicted_classes_dense, y_test)

# Adjust for the number of labels in y_test
unique_labels_in_test_dense = sorted(np.unique(y_test))

# Filter class values based on the labels present in the test set
filtered_class_values_dense = [class_values[i] for i in unique_labels_in_test_dense]

# Generate the classification report
classification_report_str_dense = classification_report(y_test, predicted_classes_dense, target_names=filtered_class_values_dense)
print("Classification Report:\n", classification_report_str_dense)

# Analyze Best and Least Correctly Classified Letters using the filtered labels
report_dense = classification_report(y_test, predicted_classes_dense, target_names=filtered_class_values_dense, output_dict=True)
accuracies_dense = {letter: report_dense[letter]['precision'] for letter in filtered_class_values_dense}

# Best Correctly Classified Letter(s)
best_classified_dense = max(accuracies_dense, key=accuracies_dense.get)
print(f"Best correctly classified letter(s): {best_classified_dense}")

# Least Correctly Classified Letter(s)
least_classified_dense = min(accuracies_dense, key=accuracies_dense.get)
print(f"Least correctly classified letter(s): {least_classified_dense}")

# Most Commonly Misclassified Letters
conf_matrix_dense = confusion_matrix(y_test, predicted_classes_dense, labels=unique_labels_in_test_dense)
np.fill_diagonal(conf_matrix_dense, 0)
most_misclassified_dense = np.unravel_index(np.argmax(conf_matrix_dense, axis=None), conf_matrix_dense.shape)
print(f"Most commonly misclassified letter(s): {filtered_class_values_dense[most_misclassified_dense[0]]} misclassified as {filtered_class_values_dense[most_misclassified_dense[1]]}")


# ## Densely Connected Model 2 

# In[47]:


# using the same learning rate, number of epochs, batch size, and activation function as the first dense model, but with a different hidden layer configuration and optimizer based on the second-best parameters 
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Based on the 2nd best parameters
final_lr_dense_2 = 0.001  # Keeping the same learning rate
final_epoch_dense_2 = 25  # Keeping the same number of epochs
final_hidden_size_dense_2 = [16, 32, 64, 16]  # Different hidden layer configuration
final_optimizer_dense_2 = keras.optimizers.SGD(learning_rate=final_lr_dense_2)  # Using SGD as the optimizer
batch_size_dense_2 = 32
actfn_dense_2 = "elu"

# Updated model_dense_factory function for Dense model 2
def model_dense_factory(hiddensizes, actfn, optimizer):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[32, 32, 1]))  # Flatten the input layer
    for n in hiddensizes:
        model.add(keras.layers.Dense(n, activation=actfn))  # Hidden layers
    model.add(keras.layers.Dense(25, activation='softmax'))  # Output layer for 25 classes

    # Compile the model with the optimizer passed as an argument
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer, 
                  metrics=["accuracy"])
    
    return model

# Build the Dense model 2 using the second-best parameters
dense_model_best_2 = model_dense_factory(final_hidden_size_dense_2, actfn_dense_2, final_optimizer_dense_2)

# Display the model summary
dense_model_best_2.summary()

# Train and evaluate the model with the second-best parameters
max_val_acc_dense_2, test_results_dense_2, training_history_dense_2, trained_model_dense_2 = fit_evaluate(dense_model_best_2, final_epoch_dense_2, batch_size=batch_size_dense_2, callbacks=None)

# Print Best Validation Accuracy
print(f"Best validation accuracy: {max_val_acc_dense_2:.3f}")

# Loss and Accuracy on the Test Set
test_loss_dense_2, test_accuracy_dense_2 = test_results_dense_2
print(f"Loss against test set: {test_loss_dense_2:.4f}")
print(f"Overall accuracy against test set: {test_accuracy_dense_2:.4f}")

# Generate predictions on the test set
predictions_dense_2 = trained_model_dense_2.predict(X_test)
predicted_classes_dense_2 = np.argmax(predictions_dense_2, axis=1)

# Plot confusion matrix and class prediction accuracy graph
plot_confusion_matrix(predicted_classes_dense_2, y_test)

# Adjust for the number of labels in y_test
unique_labels_in_test_dense_2 = sorted(np.unique(y_test))

# Filter class values based on the labels present in the test set
filtered_class_values_dense_2 = [class_values[i] for i in unique_labels_in_test_dense_2]

# Generate the classification report
classification_report_str_dense_2 = classification_report(y_test, predicted_classes_dense_2, target_names=filtered_class_values_dense_2)
print("Classification Report:\n", classification_report_str_dense_2)

# Analyze Best and Least Correctly Classified Letters using the filtered labels
report_dense_2 = classification_report(y_test, predicted_classes_dense_2, target_names=filtered_class_values_dense_2, output_dict=True)
accuracies_dense_2 = {letter: report_dense_2[letter]['precision'] for letter in filtered_class_values_dense_2}

# Best Correctly Classified Letter(s)
best_classified_dense_2 = max(accuracies_dense_2, key=accuracies_dense_2.get)
print(f"Best correctly classified letter(s): {best_classified_dense_2}")

# Least Correctly Classified Letter(s)
least_classified_dense_2 = min(accuracies_dense_2, key=accuracies_dense_2.get)
print(f"Least correctly classified letter(s): {least_classified_dense_2}")

# Most Commonly Misclassified Letters
conf_matrix_dense_2 = confusion_matrix(y_test, predicted_classes_dense_2, labels=unique_labels_in_test_dense_2)
np.fill_diagonal(conf_matrix_dense_2, 0)
most_misclassified_dense_2 = np.unravel_index(np.argmax(conf_matrix_dense_2, axis=None), conf_matrix_dense_2.shape)
print(f"Most commonly misclassified letter(s): {filtered_class_values_dense_2[most_misclassified_dense_2[0]]} misclassified as {filtered_class_values_dense_2[most_misclassified_dense_2[1]]}")


# ## Section 3.2 Model Selection

# In[49]:


import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Model accuracy data
model_names = ["CNN model 1", "CNN model 2", "Densely Connected model 1", "Densely Connected model 2"]
accuracies = [0.9438, 0.858, 0.6845, 0.4622]  # Replace with the actual accuracies

# Create a DataFrame for better display
model_comparison_df = pd.DataFrame({
    "Model": model_names,
    "Overall Accuracy": accuracies
})

# Sort models by accuracy in descending order
model_comparison_df.sort_values(by="Overall Accuracy", ascending=False, inplace=True)

# Display the results
print("Model selection - ranking")
display(model_comparison_df)



# ## Section 3.3 Future Recommendation and Final Result

# #### Best Model Analysis:
# 
# ##### Best Model: CNN model 1
# ###### Best Validation Accuracy: 0.948
# ###### Test Loss: 0.3585
# ###### Overall Test Accuracy: 0.9438
# ###### Classification Insights:
# 
# ###### Best correctly classified letter: B
# ###### Least correctly classified letter: S
# ###### Most commonly misclassified letter: H misclassified as G

# ## 3.4 DISCUSSION
# 
# 
# ###### The CNN model 1 outperformed the other models with an overall accuracy of 94.38% on the test set. This model had a great performance across all letters, with letter B being the best classified and letter S being the least correctly classified. The model showed that there is still some confusion  and some work need to be done between letters, such as H being misclassified as G.

# ## 3.5 Recommendation
# ###### Given the current computational constraints and the results, the CNN model 1 is the most suitable for deployment in a lightweight app. For future improvements, particularly in handling difficult cases like distinguishing between similar hand shapes (e.g., H vs. G), the following steps could be considered:
# 
# ###### Increasing Filter Sizes: Larger filter sizes in the convolutional layers may help the model to capture more detailed features in the images, which could increase accuracy .
# ###### Fine-tuning the Learning Rate: Experimenting with   learning rate adjustments might produce better optimization and potentially higher accuracy.
