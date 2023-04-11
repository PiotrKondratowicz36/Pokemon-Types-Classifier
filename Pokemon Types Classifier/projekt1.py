import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

# wczytanie danych
data = pd.read_csv("pokemon.csv", sep=";", names=["text", "type"])

# podział na zbiór uczący i testowy
data_train, data_test = train_test_split(data, test_size=0.10)

# wydzielenie X i y
X_train = data_train['text']
y_train = data_train['type']
X_test = data_test['text']
y_expected = data_test['type']

# użycie tf-idf
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# przygotowanie modelu, czyli naiwnego klasyfikatora Bayesa
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# użycie modelu do przewidzenia wyników zbioru testowego
y_predicted = model.predict(X_test_tfidf)

# ewaluacja
accuracy = metrics.accuracy_score(y_expected, y_predicted)
precision, recall, f_score, support = precision_recall_fscore_support(y_expected, y_predicted, average='micro')
print(f"Accuracy:   {accuracy}")
print(f"Precision:  {precision}")
print(f"Recall:     {recall}")
print(f"F-score:    {f_score}"'\n')
print(metrics.classification_report(y_expected, y_predicted,
                                    target_names=['FIRE', 'WATER', 'GRASS', 'ELECTRIC', 'GROUND']))

# testowanie własnych zdań
while True:
    x = input("Podaj tekst do klasyfikacji: ")
    text = [x]
    if text == ["exit"]:
        exit()
    text2 = tfidf_vectorizer.transform(text)
    prediction = model.predict(text2)
    print("Prediction: " + str(prediction))
