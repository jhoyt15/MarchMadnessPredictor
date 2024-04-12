from dataHandling.importData import importTrainData, importTestData
from NaiveBayes.NBClassifier import NBClassifier

train_data, train_labels = importTrainData()
test_data,team_names = importTestData()
nb_classifier = NBClassifier(train_data,train_labels)
nb_classifier.train()

for i in range(0,len(test_data)):
    prediction = nb_classifier.predict(test_data[i])

    print(team_names[i],":",nb_classifier.getPrediction(prediction))
