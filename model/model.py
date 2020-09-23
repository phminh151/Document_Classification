import pickle
from sklearn.naive_bayes import MultinomialNB

def load_NB_model():
    with open('model\document_classifier.pkl','rb') as model:
        NB_classifier= pickle.load(model)

    return NB_classifier


