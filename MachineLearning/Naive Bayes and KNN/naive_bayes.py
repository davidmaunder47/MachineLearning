import os
import math
from collections import Counter


class NaiveBayesClassifier:
    """Code for a bag-of-words Naive Bayes classifier.
    """

    def __init__(self, train_dir='data/train', REMOVE_STOPWORDS=False):
        self.REMOVE_STOPWORDS = REMOVE_STOPWORDS
        self.stopwords = set([l.strip() for l in open('english.stop')])
        self.classes = os.listdir(train_dir)
        self.train_data = {c: os.path.join(train_dir, c) for c in self.classes}
        for value in self.train_data:
            self.train_data[value] = self.train_data[value].replace("\\", "/")
        self.vocabulary = set()
        self.logprior = {}
        self.loglikelihood = {}  # keys should be tuples in the form (w, c)


    def train(self):

        """

        Note that self.train_data contains the paths to training data files. 
        To get all the documents for a given training class c in a list, you can use:
            c_docs = open(self.train_data[c]).readlines()

        You can get words with simply `words = doc.split()`

        Parameters
        ----------
        None (reads training data from self.train_data)
        
        Returns
        -------
        None (updates class attributes self.vocabulary, self.logprior, self.loglikelihood)
        """

        # We will initialize two empty arrays and two int variables.
        # The two int variables will be used to track how many training documents we have for each class
        # The two arrays will be used to count how many words we have for each class
        real_array = []
        fake_array = []
        prior_of_real = 0
        prior_of_fake = 0

        # >>> YOUR ANSWER HERE
        """This code will be used when stop loss is set to false
        If stop loss is set to false we will open both documents and append the words to 
        self.vocabulary and the array of that class. This will allow us to get |V| and the W,C.
        Also will looping we can track the amount of documents we have for each class so we can 
        calculate our priors.
        
        For Else option, this is if we want to remove stop words. The only difference is we will check to see 
        if the intended word is in the stop word document. If it is, we wont add it to either 
        the vocabulary of the length of either class
        """
        if not self.REMOVE_STOPWORDS:
            # this is iterating through the fake training set
            with open(self.train_data[self.classes[0]]) as fake_train:
                for line in fake_train:
                    prior_of_fake += 1
                    for word in line.split():
                        self.vocabulary.add(word)
                        fake_array.append(word)
            fake_train.close()

            with open(self.train_data[self.classes[1]]) as real_train:
                for line2 in real_train:
                    prior_of_real += 1
                    for word2 in line2.split():
                        self.vocabulary.add(word2)
                        real_array.append(word2)
            # calculate the log prior not using the remove stop word option. We will use log for smoothing purposes
            self.logprior[self.classes[0]] = math.log(prior_of_fake / (prior_of_fake + prior_of_real))
            self.logprior[self.classes[1]] = math.log(prior_of_real / (prior_of_fake + prior_of_real))
            real_train.close()

        else:
            with open(self.train_data[self.classes[0]]) as fake_train:
                for line in fake_train:
                    prior_of_fake += 1
                    for word in line.split():
                        if word not in self.stopwords:
                            self.vocabulary.add(word)
                            fake_array.append(word)
            fake_train.close()

            with open(self.train_data[self.classes[1]]) as real_train:
                for line2 in real_train:
                    prior_of_real += 1
                    for word2 in line2.split():
                        if word2 not in self.stopwords:
                            self.vocabulary.add(word2)
                            real_array.append(word2)
            real_train.close()
            # calculate the log prior of the remove stop word option. We will use log for smoothing purposes
            self.logprior[self.classes[0]] = math.log(prior_of_fake / (prior_of_fake + prior_of_real))
            self.logprior[self.classes[1]] = math.log(prior_of_real / (prior_of_fake + prior_of_real))


        """
        this for loop will build our
        here we will add each count of the word in the class plus one for lapace smoothing
        Lastly we will divide this amount by the sum of the self.vocabulary and the length of the words in the class
        """
        for word in self.vocabulary:
            word_count = fake_array.count(word)
            word2_count = real_array.count(word)
            self.loglikelihood[(word, 'fake.txt')] = math.log(
                (word_count + 1) / (len(self.vocabulary) + len(fake_array)))
            self.loglikelihood[(word, 'real.txt')] = math.log(
                (word2_count + 1) / (len(self.vocabulary) + len(real_array)))

        pass
        # >>> END YOUR ANSWER

    def score(self, doc, c):
        """Return the log-probability of a given document for a given class,
        using the trained Naive Bayes classifier. 

        This is analogous to the inside of the for loop in the TestNaiveBayes
        pseudocode in Figure 4.2, SLP Chapter 4.

        Parameters
        ----------
        doc : str
            The text of a document to score.
        c : str
            The name of the class to score it against.

        Returns
        -------
        float
            The log-probability of the document under the model for class c.
        """
        """
        First thing we will do us check to see which class we are using. After we will take the running sum 
        of the word if it appears in our vocabulary. If it does not appear in our vocab, we will not add it. 
        """
        if c == self.classes[0]:
            ending_value = self.logprior[self.classes[0]]
            doc = doc.split()
            for word in doc:
                if word in self.vocabulary:
                    ending_value += self.loglikelihood[(word, "fake.txt")]

        elif c == self.classes[1]:
            ending_value = self.logprior[self.classes[1]]
            doc = doc.split()
            for word2 in doc:
                if word2 in self.vocabulary:
                    ending_value += self.loglikelihood[(word2, "real.txt")]

        # >>> YOUR ANSWER HERE
        return ending_value
        # >>> END YOUR ANSWER

    def predict(self, doc):
        """Return the most likely class for a given document under the trained classifier model.
        This should be only a few lines of code, and should make use of your self.score function.


        Parameters
        ----------
        doc : str
            A text representation of a document to score.
        
        Returns
        -------
        str
            The most likely class as predicted by the model.
        """
        # >>> YOUR ANSWER HERE
        # simple function that calls predict for each class. It will then return the amount which is larger
        real_score = self.score(doc, self.classes[1])
        fake_score = self.score(doc, self.classes[0])

        if real_score < fake_score:
            return self.classes[0]

        return self.classes[1]
        # >>> END YOUR ANSWER

    def evaluate(self, test_dir='data/dev', target='relevant'):
        """Calculate a precision, recall, and F1 score for the model
        on a given test set.

        Not the 'target' parameter here, giving the name of the class
        to calculate relative to. So you can consider a True Positive
        to be an instance where the gold label for the document is the
        target and the model also predicts that label; a False Positive
        to be an instance where the gold label is *not* the target, but
        the model predicts that it is; and so on.

        Parameters
        ----------
        test_dir : str
            The path to a directory containing the test data.
        target : str
            The name of the class to calculate relative to. 

        Returns
        -------
        (float, float, float)
            The model's precision, recall, and F1 score relative to the
            target class.
        """
        outcomes = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        tempdirection = os.listdir(test_dir)


        test_data = {c: os.path.join(test_dir, c) for c in tempdirection}
        for value in test_data:
            test_data[value] = test_data[value].replace("\\", "/")



        """
        We will open both documents and see if it matches our expectations. If we say the document is a real
        document for our testing document class we will get a True Positive amount, else we will get a False positive. 
        Then we will do the same thing for the fake testing set. If we guess right it will be a 
        true negative amount, else it will be a false negative. 
        """
        with open(test_data["clean_fake_dev.txt"]) as fake_test:
            for document in fake_test:
                results = self.predict(document)
                if results == self.classes[0]:
                    outcomes['TN'] += 1

                elif results == self.classes[1]:
                    outcomes['FN'] += 1
        fake_test.close()

        with open(test_data["clean_real_dev.txt"]) as real_test:
            for document2 in real_test:
                results = self.predict(document2)
                if results == self.classes[1]:
                    outcomes['TP'] += 1

                elif results == self.classes[0]:
                    outcomes['FP'] += 1
        real_test.close()

        # >>> YOUR ANSWER HERE

        precision = outcomes['TP'] / (outcomes['TP'] + outcomes['FP'])
        recall = outcomes['TP'] / (outcomes['TP'] + outcomes['FN'])
        f1_score = 2 * ((precision * recall) / (precision + recall))

        # this is just for visualization purposes, can be taken out if needed.
        print("TP:", outcomes['TP'])
        print("FP:", outcomes['FP'])
        print("TN:", outcomes['TN'])
        print("FN:", outcomes['FN'])

        # >>> END YOUR ANSWER
        return precision, recall, f1_score




if __name__ == '__main__':
    # following code was provided and is used for testing and implementing Naive Bayes
    target = 'relevant'

    clf = NaiveBayesClassifier(train_dir='data/train')

    clf.train()
    print(f'Performance on class <{target.upper()}>, keeping stopwords')
    precision, recall, f1_score = clf.evaluate(test_dir='data/dev', target="true")
    print(f'\tPrecision: {precision}\t Recall: {recall}\t F1: {f1_score}\n')

    clf = NaiveBayesClassifier(train_dir='data/train', REMOVE_STOPWORDS=True)
    clf.train()
    print(f'Performance on class <{target.upper()}>, removing stopwords')
    precision, recall, f1_score = clf.evaluate(test_dir='data/dev', target="true")
    print(f'\tPrecision: {precision}\t Recall: {recall}\t F1: {f1_score}\n')



