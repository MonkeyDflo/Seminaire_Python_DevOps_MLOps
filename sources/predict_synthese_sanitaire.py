#steps : 
#1. Charger en mémoire les data sets => avoir un padas data frame
import pandas
#df_alim = pandas.read_csv("C:\\Users\\flore\\OneDrive\\Documents\\M2 DATA MINING\\Séminaire Python DevOps MLops pavel Sorano\\Projet\\data\\export_alimconfiance.csv", sep=";")
#df_alim = pandas.read_csv("C:\\Users\\flore\\Desktop\\export_alimconfiance.csv", sep=";")
df_alim = pandas.read_csv("/home/florent/Documents/Seminaire_Python_DevOps_MLOps/data/export_alimconfiance.csv", sep=";")
df_alim.columns

#concaténation de variables 

#df_alim["texte"] = df_alim["APP_Libelle_etablissement"] + df_alim.filtre +df_alim.ods_type_activite
#df_alim["texte"].loc[0]
#df_alim["texte"] = df_alim["APP_Libelle_etablissement"] + " " + df_alim.filtre + " " + df_alim.ods_type_activite
#y = df_alim["Synthere_eval_sanit"]
#X = df_alim["texte"]

#X.shape
#Y.shape

#df_alim.dropna() 
df_alim.shape

df_alim2 = df_alim.dropna()
df_alim2["texte"] = df_alim2["APP_Libelle_etablissement"] + " " + df_alim2.filtre + " " + df_alim2.ods_type_activite
y = df_alim2["Synthese_eval_sanit"]
X = df_alim2["texte"]

print(X.shape, y.shape)

#data science one o one 
#diviser le jeu de données 
 
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#matrice creuse car bcp de 0 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

vectorizer = CountVectorizer()

#? vectorizer 

X_train_text = vectorizer.fit_transform(X_train)
X_train_text.shape
X_train_text
X_train_text.A

X_train_test = vectorizer.fit_transform(X_train)
X_test_text = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=10000)

clf.fit(X_train_test, y_train)
y_prod = clf.predict(X_test_text)
print(classification_report(y_test, y_prod))

#2. entrainer un modèle scikit learn
#3. 

