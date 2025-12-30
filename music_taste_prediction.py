import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree
import graphviz
import os
# Importa o arquivo CSV com dados de vendas de videogames
# caminho absoluto usado para evitar problemas com diretório atual
music_data = pd.read_csv(r"D:\02_Machine-Learning\Mosh_ PythonMachineLearningTutorial(Data Science)\dataset\music.csv")
print(music_data.shape)
print(music_data.head())
print(music_data.describe())


X = music_data.drop(columns=['genre']) #quero tudo menos o genre
Y = music_data['genre'] #quero selecionanr o genre

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions= model.predict(X_test)

#print(predictions) 
#score = accuracy_score(Y_test, predictions)
#print(score)

#temos o modelo guardado e podemos carregá-lo quando quisermos para fazer previsões
joblib.dump(model, r"D:\02_Machine-Learning\Mosh_ PythonMachineLearningTutorial(Data Science)\dataset\music_recomender.joblib")
model = joblib.load(r"D:\02_Machine-Learning\Mosh_ PythonMachineLearningTutorial(Data Science)\dataset\music_recomender.joblib")


tree.export_graphviz(model, out_file=r"D:\02_Machine-Learning\Mosh_ PythonMachineLearningTutorial(Data Science)\dataset\music_recommender.dot",
                        feature_names=['age', 'gender'],
                        class_names=sorted(Y.unique()),
                        label='all',
                        rounded=True,
                        filled=True)

dot_data = tree.export_graphviz(model, out_file=None)
graph = graphviz.Source(dot_data)
import os
os.environ["PATH"] += os.pathsep + r"C:\Users\Utilizador\windows_10_cmake_Release_Graphviz-14.1.1-win64\Graphviz-14.1.1-win64\bin"
graph.render("music_recommender")

