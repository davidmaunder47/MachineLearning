import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier


def load_data():
    # Open for file for reading only
    file_fake_dev = open("C:/Users/johnd/Documents/hw1/question2/data/dev/clean_fake_dev.txt", "r")
    file_real_dev = open("C:/Users/johnd/Documents/hw1/question2/data/dev/clean_real_dev.txt", "r")
    file_fake_train = open("C:/Users/johnd/Documents/hw1/question2/data/train/clean_fake_train.txt", "r")
    file_real_train = open("C:/Users/johnd/Documents/hw1/question2/data/train/clean_real_train.txt", "r")

    # temp dev array to pass
    dev_list_array = []

    # train array to pass 
    train_list_array = []

    # We will use these four count_lines to measure how many documents 
    # are in the real dev, fake dev, real train and fake train documents
    count_lines = 0
    count_lines2 = 0
    count_lines3 = 0
    count_lines4 = 0

    # These first two loops will append each document to the dev training list
    # the count_lines for each will be passed so we make can a label training set
    for line in file_fake_dev:
        dev_list_array.append(line)
        count_lines += 1

    for line2 in file_real_dev:
        dev_list_array.append(line2)
        count_lines2 += 1

    for line3 in file_fake_train:
        train_list_array.append(line3)
        count_lines3 += 1

    for line4 in file_real_train:
        train_list_array.append(line4)
        count_lines4 += 1

    # Setting up labels for the training data set
    # We will use zeros for fake, and ones for true documents
    fake_train_array = np.zeros(count_lines3, dtype=int)
    real_train_array = np.ones(count_lines4, dtype=int)
    labels_training_array = np.concatenate((fake_train_array, real_train_array))

    # setting up labels for the developmental/testing set
    fake_dev_array = np.zeros(count_lines, dtype=int)
    real_dev_array = np.ones(count_lines2, dtype=int)
    labels_dev_array = np.concatenate((fake_dev_array, real_dev_array))

    # Here we vectorize our training data set to 1 and zeros and transform our
    # testing data set using transform. 
    vectorizer = CountVectorizer(analyzer='word')
    train_array1 = vectorizer.fit_transform(train_list_array)
    test_array2 = vectorizer.transform(dev_list_array)

    file_fake_dev.close()
    file_real_dev.close()
    file_fake_train.close()
    file_real_train.close()

    return train_array1, test_array2, labels_training_array, labels_dev_array


def select_knn_model():
    train_array1, test_array2, labels_training_array, labels_dev_array = load_data()

    # Temp arrays to store results of our KNN classifier 
    knn_validation = []
    knn_training = []
    temp_array = []

    """
    Below we are testing our KNN classifier not using cosine. Here we will fit our 
    model and then compare these results to our labels. 
    """
    for i in range(1, 21):
        temp_array.append(i)
        knn_classifier = KNeighborsClassifier(n_neighbors=i)
        knn_classifier.fit(train_array1, labels_training_array)
        prediction = knn_classifier.predict(test_array2)
        prediction2 = knn_classifier.predict(train_array1)
        knn_training.append(sklearn.metrics.accuracy_score(labels_training_array, prediction2))
        knn_validation.append(sklearn.metrics.accuracy_score(labels_dev_array, prediction))

    # Below is boiler plat code to plot our data
    plt.Figure(figsize=(12, 6))
    plt.plot(temp_array, knn_training, label='training', color='red', linestyle='solid', marker='o',
             markerfacecolor='blue',
             markersize=10)
    plt.plot(temp_array, knn_validation, label='validation', color='green', linestyle='solid', marker='o',
             markerfacecolor='green',
             markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.legend()
    plt.ylabel('Mean Error')
    plt.show()

    knn_validation_cosine = []
    knn_training_cosine = []
    temp_array_cosine = []

    """
       We are conducting the same process as above expect we are using cosine instead.  
       """
    for j in range(1, 21):
        temp_array_cosine.append(j)
        knn_classifier_cosine = KNeighborsClassifier(n_neighbors=j, metric="cosine")
        knn_classifier_cosine.fit(train_array1, labels_training_array)
        prediction_cosine = knn_classifier_cosine.predict(test_array2)
        prediction2_cosine = knn_classifier_cosine.predict(train_array1)
        knn_training_cosine.append(sklearn.metrics.accuracy_score(labels_training_array, prediction2_cosine))
        knn_validation_cosine.append(sklearn.metrics.accuracy_score(labels_dev_array, prediction_cosine))

    # Below is boiler plat code to plot our data
    plt.Figure(figsize=(12, 6))
    plt.plot(temp_array_cosine, knn_training_cosine, label='training', color='red', linestyle='solid', marker='o',
             markerfacecolor='blue',
             markersize=10)
    plt.plot(temp_array_cosine, knn_validation_cosine, label='validation', color='green', linestyle='solid', marker='o',
             markerfacecolor='green',
             markersize=10)
    plt.title('Error Rate K Value for Cosine')
    plt.xlabel('K Value')
    plt.legend()
    plt.ylabel('Mean Error')
    plt.show()


# used to run our code
select_knn_model()
