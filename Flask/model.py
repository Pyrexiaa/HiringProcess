import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import spacy
import re
import matplotlib.pyplot as plt 
import pickle
import joblib

def train_model():
    # loading the data from csv file to a pandas Dataframe
    data = pd.read_csv("Datasets/data.csv")
    others = pd.read_csv("Datasets/training_data.csv")
    others = others.loc[others['class']==0]
    others['word_count'] = others['text_join'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    other_data = others[others['word_count'] > 10]

    def split_sentences(text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences

    data['Answers'] = data['Answers'].apply(split_sentences)
    data = data.explode('Answers').reset_index(drop=True)

    other_data['normal'] = other_data['text_join'].astype(str)
    other_data.replace('', np.nan, inplace=True)
    other_data.replace('nan', np.nan, inplace=True)
    other_data = other_data.dropna()

    num_rows = int(len(other_data['normal']) * 0.05)
    selected_indices = np.random.choice(other_data.index, size=num_rows, replace=True)
    reduced_data = other_data['normal'].loc[selected_indices].reset_index(drop=True)

    d_length = len(data['Answers'])
    c_length = len(reduced_data)
    samples = d_length - c_length
    oversampled_indices = np.random.choice(other_data['normal'].index, size=samples, replace=True)
    oversampled_df2 = other_data['normal'].loc[oversampled_indices].reset_index(drop=True)
    # Concatenate 
    new_df = pd.concat([data['Answers'], oversampled_df2], axis=1)

    df_new = pd.melt(new_df, value_vars=['normal', 'Answers'], var_name='Source', value_name='Combined')
    df_new.replace('', np.nan, inplace=True)
    df_new.replace('nan', np.nan, inplace=True)
    df = df_new.dropna()

    df.loc[df['Source'] == 'normal', 'Source'] = 0
    df.loc[df['Source'] == 'Answers', 'Source'] = 1

    X = df['Combined']
    Y = df['Source']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=3)

    feature_extraction = TfidfVectorizer(lowercase=True, stop_words='english')
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
    ])        
    X_train_features = pipeline.fit_transform(X_train).todense()
    pca = PCA(n_components=2).fit(np.array(X_train_features))
    data2D = pca.transform(np.array(X_train_features))
    plt.scatter(data2D[:,0], data2D[:,1])
    plt.show()

    X_test_features = pipeline.transform(X_test).todense()

    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')

    model = IsolationForest(contamination= 0.1, max_features= 1, n_estimators= 100, max_samples=0.5)
    model.fit(np.array(X_train_features))
    predictions = model.predict(np.array(X_train_features))

    X_train_features_dense = X_train_features
    print(predictions)
    outlier_index = np.where(predictions == -1)[0] 
    values = X_train_features_dense[outlier_index]
    
    plt.scatter([X_train_features_dense[:,0]], [X_train_features_dense[:,1]])
    plt.scatter([values[:,0]], [values[:,1]], color='y')
    plt.show()

    # prediction on training data
    prediction_on_training_data = model.predict(np.array(X_train_features))
    print(prediction_on_training_data)
    accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
    print('Accuracy on training data : ', accuracy_on_training_data)

    # prediction on test data
    prediction_on_test_data = model.predict(np.array(X_test_features))
    accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
    print('Accuracy on test data : ', accuracy_on_test_data)

    filename = 'trained_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    joblib.dump(pipeline, 'pipeline.pkl')

# def predictions(model, input_stuff):
    # pipeline = Pipeline([
    #     ('vect', CountVectorizer()),
    #     ('tfidf', TfidfTransformer()),
    # ])  
    # input_stuffs = [input_stuff]
    # nlp = spacy.load("en_core_web_sm")
    # string = ' '.join(input_stuffs)
    # doc = nlp(string)
    # anomaly_results = []  
    # normal_results = []
    # for sent in doc.sents:
    #     input = [str(sent)]
    #     print(input)
    #     # convert text to feature vectors
    #     input_data_features = pipeline.fit_transform(input)
    #     # # making prediction
    #     prediction = model.predict(input_data_features)
    #     # If the input is detected to be anomalous
    #     if (prediction[0]==1):
    #         normal_results.append(str(sent))
    #     else:
    #         anomaly_results.append(str(sent))

    # return normal_results, anomaly_results

if __name__ == "__main__":
    train_model()
   