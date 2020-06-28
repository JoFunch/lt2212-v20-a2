import argparse
import random

from sklearn.datasets import fetch_20newsgroups
from sklearn.base import is_classifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import is_classifier
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS


import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#for bonus
from sklearn.decomposition import PCA

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



random.seed(42)

###### PART 1
#DONT CHANGE THIS FUNCTION
def part1(samples):
    #extract features
    X = extract_features(samples)
    assert type(X) == np.ndarray
    print("Example sample feature vec: ", X[0])
    print("Data shape: ", X.shape)
    return X

#Lav tokenizer sep.

dataone = [""" was wondering if anyone out there could enlighten me on this car I saw
the other day. It was a 2-door sports car, looked to be from the late 60s/
early 70s. It was called a Bricklin. The doors were really small. In addition,
the front bumper was separate from the rest of the body. This is
all I know. If anyone can tellme a model name, engine specs, years
of production, where this car is made, history, or whatever info you
have on this funky looking car, please e-mail."""]

data = fetch_20newsgroups(subset ='all', shuffle=True, random_state=42)

    
#add preproccessing


def tokenisation(text):
    # lower -> tokenization/divide on whitespace (meaning no n-grams) ->
    # alphabeticals -> stopwords
    lower_tokens = text.lower() 
    tokens = lower_tokens.split(' ')
    processed = [word for word in tokens if word.isalpha()]
    #stop_words = [word for word in processed if word not in ENGLISH_STOP_WORDS]
    #counts = [word for word in stop_words if stop_words.count(word) > 5]

    return processed



#print(tokenisation(dataone))

def extract_features(samples):
    # all articles are tokenised, put in dict and added to list
    # the list of dicts are made vectors and transformed into arrays
    print("Extracting features ...")
    list_of_articles = []
    for i in samples:
        article = dict()
        words = tokenisation(i)
        for word in words:
            if word in article:
                article[word] += 1
            else:
                article[word] = 1
        
        list_of_articles.append(article)
    vector = DictVectorizer()
    array = vector.fit_transform(list_of_articles).toarray()
    return array
        
        
        
        
#print(extract_features(data.data))
#print(extract_features(data.data))


##### PART 2
#DONT CHANGE THIS FUNCTION
def part2(X, n_dim):
    #Reduce Dimension
    print("Reducing dimensions ... ")
    X_dr = reduce_dim(X, n=n_dim)
    assert X_dr.shape != X.shape
    assert X_dr.shape[1] == n_dim
    print("Example sample dim. reduced feature vec: ", X[0])
    print("Dim reduced data shape: ", X_dr.shape)
    return X_dr


def reduce_dim(X,n=10):
    #fill this in
    svd = PCA(n_components = n)
    svd.fit(X)
    return svd.transform(X)

#print(reduce_dim(extract_features(data.data)))

##### PART 3
#DONT CHANGE THIS FUNCTION EXCEPT WHERE INSTRUCTED
def get_classifier(clf_id):
    if clf_id == 1:
        clf = KNeighborsClassifier() # <--- REPLACE THIS WITH A SKLEARN MODEL
    elif clf_id == 2:
        clf = GaussianNB() # <--- REPLACE THIS WITH A SKLEARN MODEL
    else:
        raise KeyError("No clf with id {}".format(clf_id))

    assert is_classifier(clf)
    print("Getting clf {} ...".format(clf.__class__.__name__))
    return clf

#DONT CHANGE THIS FUNCTION
def part3(X, y, clf_id):
    #PART 3
    X_train, X_test, y_train, y_test = shuffle_split(X,y)

    #get the model
    clf = get_classifier(clf_id)

    #printing some stats
    print()
    print("Train example: ", X_train[0])
    print("Test example: ", X_test[0])
    print("Train label example: ",y_train[0])
    print("Test label example: ",y_test[0])
    print()


    #train model
    print("Training classifier ...")
    train_classifer(clf, X_train, y_train)


    # evalute model
    print("Evaluating classcifier ...")
    evalute_classifier(clf, X_test, y_test)


def shuffle_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    return X_train, X_test, y_train, y_test


def train_classifer(clf, X, y):
    assert is_classifier(clf)
    trained_model = clf.fit(X,y)
    return trained_model

   

def evalute_classifier(clf, X, y):
    assert is_classifier(clf)
    #Fill this in
    predict = clf.predict(X)
    accuracy = accuracy_score(y, predict, normalize=True)
    precision = precision_score(y, predict, average = "macro")
    recall = recall_score(y, predict, average = "macro")
    f_measure = f1_score(y, predict, average = "macro")
    classification = classification_report(y, predict)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("recall: ", recall)
    print("F-Measure: ", f_measure)
    print("Classification: ", classification)

    
    
    
#print(part3(part2(part1(data.data), 500), data.target, 2))
#print(part3(part2(part1(data.data), 100), data.target, 2))
#print(part3(part2(part1(data.data), 50), data.target, 2))
#print(part3(part2(part1(data.data), 25), data.target, 2))
#print(part3(part2(part1(data.data), 10), data.target, 2))
print(part3(part1(data.data), data.target, 2))  




######
#DONT CHANGE THIS FUNCTION
def load_data():
    print("------------Loading Data-----------")
    data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
    print("Example data sample:\n\n", data.data[0])
    print("Example label id: ", data.target[0])
    print("Example label name: ", data.target_names[data.target[0]])
    print("Number of possible labels: ", len(data.target_names))
    return data.data, data.target, data.target_names


#DONT CHANGE THIS FUNCTION
def main(model_id=None, n_dim=False):

    # load data
    samples, labels, label_names = load_data()


    #PART 1
    print("\n------------PART 1-----------")
    X = part1(samples)

    #part 2
    if n_dim:
        print("\n------------PART 2-----------")
        X = part2(X, n_dim)

    #part 3
    if model_id:
        print("\n------------PART 3-----------")
        part3(X, labels, model_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_dim",
                        "--number_dim_reduce",
                        default=False,
                        type=int,
                        required=False,
                        help="int for number of dimension you want to reduce the features for")

    parser.add_argument("-m",
                        "--model_id",
                        default=False,
                        type=int,
                        required=False,
                        help="id of the classifier you want to use")

    args = parser.parse_args()
    main(   
            model_id=args.model_id, 
            n_dim=args.number_dim_reduce
            )
