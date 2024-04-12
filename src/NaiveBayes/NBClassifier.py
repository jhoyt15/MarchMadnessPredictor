import numpy as np
import scipy.stats

class NBClassifier:
    def __init__(self,train_data:np.array,train_labels:np.array):
        self.train_data = train_data
        self.train_labels = train_labels
        self.prior = dict()
        self.conditional_means = []
        self.conditional_stds = []

    """
    Get the number of occurences of each label.

    Args: 
        self (NBClassifier) : The classifer to be trained
    Returns:
        dict: The labels and their corresponding number of occurences
    """
    def getOccurences(self)->dict:
        occurences = dict()
        for label in self.train_labels:
            if label in occurences:
                occurences[label] += 1
            else:
                occurences[label] = 1
        return occurences
    
    """
    Trains the Naive Bayes Classifier by computing the prior and the conditional probabilities.

    Args: 
        self (NBClassifier) : The classifer to be trained
    Returns:
        None
    """
    def train(self)->None:
        #For this, I am assuming that the kenpom data follows a normal (Gaussian Distribution) based on the Central Limit Theorem with N > 1000

        #First find the mean and std for the training set
        train_means = np.zeros((len(self.train_data),7)) #7 is the number of possible labels (64,32,16,8,4,2,1)
        train_stds = np.zeros((len(self.train_data),7))

        label_occurences = self.getOccurences()
        print(label_occurences)
        
        #Finding mean for each label type
        for i in range(0,len(self.train_data)):
            for j in range(0,len(self.train_data[i])):
                label_index = int(np.log2(int(self.train_labels[j])))
                train_means[i][label_index] += float(self.train_data[i][j])
        
        for i in range(0,len(train_means)):
            for j in range(0,len(train_means[i])):
                train_means[i][j] /= float(label_occurences[str(2**j)])

        #Finding std for each label type
        for i in range(0,len(self.train_data)):
            for j in range(0,len(self.train_data[i])):
                label_index = int(np.log2(int(self.train_labels[j])))
                train_stds[i][label_index] += (float(self.train_data[i][j])-train_means[i][label_index])**2

        for i in range(0,len(train_stds)):
            for j in range(0,len(train_stds[i])):
                train_stds[i][j] = np.sqrt(train_stds[i][j] / float(label_occurences[str(2**j)]-1))

        self.prior = {label: occurences / len(self.train_data[0]) for label,occurences in label_occurences.items()}
        self.conditional_means = train_means
        self.conditional_stds = train_stds


    """
    Predicts the label for a test data point.

    Args: 
        self (NBClassifier) : The classifer to be trained
        test_data_instance (np.array) : The test data instance to predict
    Returns:
        dict: A dictionary containing the probability of each result.
    """
    def predict(self, test_data_instance:np.array)->dict:
        label_prob = {'16': 0, '64': 0, '32': 0, '1': 0, '8': 0, '4': 0, '68': 0, '2': 0} #Stores probability of each label

        for label in label_prob:
            log_prior = np.log(self.prior[label])
            log_sum = 0
            log_conditional = []
            for i in range(0,len(test_data_instance)):
                label_index = int(np.log2(int(label)))
                prob_value = scipy.stats.norm.pdf(float(test_data_instance[i]),loc=self.conditional_means[i][label_index],scale=self.conditional_stds[i][label_index])
                log_conditional.append(prob_value)
            log_sum = np.sum(np.log(log_conditional))

            label_prob[label] = log_prior + log_sum
        
        return label_prob
    
    """
    Gets the prediction with the highest probability given the label results

    Args: 
        self (NBClassifier) : The classifer to be trained
        label_prob (dict) : The probability of each label
    Returns:
        int: The label with the highest probability and its corresponding probability
    """
    def getPrediction(self,label_prob:dict)->tuple:
        prediction_value = float('-inf')
        prediction = 0
        for label in label_prob:
            if label_prob[label] > prediction_value:
                prediction = int(label)
                prediction_value = label_prob[label]
        return (prediction,prediction_value)
            

        
        
