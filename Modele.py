#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from unidecode import unidecode


# In[2]:


data = pd.read_excel('/Users/nour-elmi/Desktop/Stage2A_Casa/Downtime-ttr_MTBF.xlsx')


# In[10]:


dt = data.copy()
desired_order = ['Stop group', 'Stop','Stop Location', 'Extra text','Stop type', 'Stop start time','Stop end time', 'Stop duration (min) (TTR)','Product','MTBF']
dt = dt[desired_order]
print(dt)


# 
# ## 1- Supprimer les accents et les apostrophes & convertir en minuscules & replace dons with dans

# In[11]:


# Convertir les valeurs de la colonne "Extra text" en minuscules
dt['Extra text'] = dt['Extra text'].str.lower()

# Supprimer les accents en utilisant la fonction unidecode
from unidecode import unidecode
dt['Extra text'] = dt['Extra text'].apply(unidecode)

# Supprimer les apostrophes
dt['Extra text'] = dt['Extra text'].str.replace("'", "")

# Remplacer dons par dans
dt['Extra text'] = dt['Extra text'].str.replace("dons", "dans")

# Supprimer les lignes contenant le caractère '-' dans les colonnes "Stop Location" et "Extra text"
dt = dt[dt['Stop Location'] != '-']
dt = dt[dt['Extra text'] != '-']
desired_order = ['Stop group', 'Stop','Stop Location', 'Extra text','Stop type', 'Stop start time','Stop end time', 'Stop duration (min) (TTR)','Product','MTBF']
dt = dt[desired_order]
print(dt)


# 
# ## 2- Supprimmer les lignes avec des val manquantes: Extra text & Stop location
# 
# 
# 

# In[6]:


# Supprimer les lignes contenant le caractère '-' dans les colonnes "Stop Location" et "Extra text"
dt = dt[dt['Stop Location'] != '-']
dt = dt[dt['Extra text'] != '-']



# In[7]:


# Regrouper les valeurs de la colonne "Extra text" en comptant leur fréquence dans chaque groupe
#le nombre de répétitions de chaque valeur de la colonne 'Extra text' 
#dans chaque groupe formé par les colonnes 'Stop group', 'Stop' et 'Stop Location'
grouped_counts = dt.groupby(['Stop group', 'Stop', 'Stop Location'])['Extra text'].value_counts()

# Créer un DataFrame à partir du résultat regroupé et compté
df_counts = grouped_counts.reset_index(name='Count')

# Calculer le total des occurrences dans chaque groupe
df_totals = df_counts.groupby(['Stop group', 'Stop', 'Stop Location'])['Count'].sum().reset_index(name='Total')

# Fusionner les DataFrames pour obtenir les pourcentages de chaque valeur dans chaque groupe
df_merged = df_counts.merge(df_totals, on=['Stop group', 'Stop', 'Stop Location'])
df_merged['Percentage'] = (df_merged['Count'] / df_merged['Total']) * 100

# Trier les valeurs par ordre décroissant des occurrences
df_merged = df_merged.sort_values(by='Count', ascending=False)

# Sélectionner les 20 premières lignes
df_top20 = df_merged.head(20)

# Créer une liste de labels pour le diagramme circulaire
labels = [f"{text} ({count}, {percentage:.1f}%)" for text, count, percentage in zip(df_top20['Extra text'], df_top20['Count'], df_top20['Percentage'])]

# Créer une liste de valeurs pour le diagramme circulaire
sizes = df_top20['Count']

# Afficher le diagramme circulaire
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("Répartition des valeurs de 'Extra text' (20 premières lignes)")
plt.show()


# ## 3- diviser dataset: X_train, X_test

# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Separate features (X) from the target variable (y)
X = dt[['Stop group', 'Stop', 'Stop Location']]
y = dt['Extra text']

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X['Stop group'] + ' ' + X['Stop'] + ' ' + X['Stop Location'])

# Divide the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, random_state=np.random, train_size=0.7)


# ## A- Modele affiche juste plus frequente valeur de Extra text: 

# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Train a classification model (e.g., logistic regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model performance on the test set
accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)

# Use the trained model to predict the value of 'Extra text'
stop_group_input = input("Please enter the value of 'Stop group': ")
stop_input = input("Please enter the value of 'Stop': ")
stop_location_input = input("Please enter the value of 'Stop Location': ")

input_vec = vectorizer.transform([stop_group_input + ' ' + stop_input + ' ' + stop_location_input])
predicted_extra_text = model.predict(input_vec)
print("Predicted Extra text:", predicted_extra_text)


# ## B- Model affiche les 10 frequentes val de extra text -liste-

# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Train a classification model (e.g., logistic regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model performance on the test set
accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)

# Use the trained model to predict the value of 'Extra text'
stop_group_input = input("Please enter the value of 'Stop group': ")
stop_input = input("Please enter the value of 'Stop': ")
stop_location_input = input("Please enter the value of 'Stop Location': ")

input_vec = vectorizer.transform([stop_group_input + ' ' + stop_input + ' ' + stop_location_input])
predicted_extra_text_probs = model.predict_proba(input_vec)[0]
top_10_indices = np.argsort(predicted_extra_text_probs)[::-1][:10]
top_10_values = model.classes_[top_10_indices]
print("Top 10 Predicted Extra text:")
for value in top_10_values:
    print(value)


# ## C- Model  LogisticRegression affiche les 10 frequente val de extra text -diagramme-

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from unidecode import unidecode
from sklearn import metrics

# Train a classification model (e.g., logistic regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model performance on the test set
accuracy = model.score(X_test, y_test)
#print("Model accuracy:", accuracy)

# Use the trained model to predict the value of 'Extra text'
stop_group_input = input("Please enter the value of 'Stop group': ")
stop_input = input("Please enter the value of 'Stop': ")
stop_location_input = input("Please enter the value of 'Stop Location': ")

input_vec = vectorizer.transform([stop_group_input + ' ' + stop_input + ' ' + stop_location_input])
predicted_extra_text_probs = model.predict_proba(input_vec)[0]
top_10_indices = np.argsort(predicted_extra_text_probs)[::-1][:10]
top_10_values = model.classes_[top_10_indices]

# Filter out unavailable values
top_10_values = top_10_values[:len(np.unique(y))]

# Pie chart visualization
plt.figure(figsize=(8, 8))
plt.pie(predicted_extra_text_probs[top_10_indices], labels=top_10_values, autopct='%1.1f%%')
plt.title("Top 10 Predicted Extra text")
plt.axis('equal')
plt.show()


# ## D- Model  RandomForestClassifier affiche les 10 frequente val de extra text -diagramme-

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Change the random_state to an integer value, e.g., 42
model.fit(X_train, y_train)

# Evaluate the model performance on the test set
accuracy = model.score(X_test, y_test)
#print("Model accuracy:", accuracy)

# Use the trained model to predict the value of 'Extra text'
stop_group_input = input("Please enter the value of 'Stop group': ")
stop_input = input("Please enter the value of 'Stop': ")
stop_location_input = input("Please enter the value of 'Stop Location': ")

input_vec = vectorizer.transform([stop_group_input + ' ' + stop_input + ' ' + stop_location_input])
predicted_extra_text_probs = model.predict_proba(input_vec)[0]
top_10_indices = np.argsort(predicted_extra_text_probs)[::-1][:10]
top_10_values = model.classes_[top_10_indices]

# Filter out unavailable values
top_10_values = top_10_values[:len(np.unique(y))]

# Pie chart visualization
plt.figure(figsize=(8, 8))
plt.pie(predicted_extra_text_probs[top_10_indices], labels=top_10_values, autopct='%1.1f%%')
plt.title("Top 10 Predicted Extra text")
plt.axis('equal')
plt.show()


# ## D- Model  MLPClassifier affiche les 10 frequente val de extra text -diagramme-

# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Train a Multi-Layer Perceptron classifier (Neural Network)
model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42)  # Change the random_state to an integer value, e.g., 42
model.fit(X_train, y_train)

# Evaluate the model performance on the test set
accuracy = model.score(X_test, y_test)
#print("Model accuracy:", accuracy)

# Use the trained model to predict the value of 'Extra text'
stop_group_input = input("Please enter the value of 'Stop group': ")
stop_input = input("Please enter the value of 'Stop': ")
stop_location_input = input("Please enter the value of 'Stop Location': ")

input_vec = vectorizer.transform([stop_group_input + ' ' + stop_input + ' ' + stop_location_input])
predicted_extra_text_probs = model.predict_proba(input_vec)[0]
predicted_extra_text = model.classes_
# Convert the predicted probabilities and values to a dictionary
predicted_values = dict(zip(predicted_extra_text, predicted_extra_text_probs))

# Sort the predicted values based on probabilities in descending order
sorted_values = sorted(predicted_values.items(), key=lambda x: x[1], reverse=True)

# Separate the top 10 predicted values and probabilities
top_10_values = [value[0] for value in sorted_values[:10]]
top_10_probs = [value[1] for value in sorted_values[:10]]

# Pie chart visualization
plt.figure(figsize=(8, 8))
plt.pie(top_10_probs, labels=top_10_values, autopct='%1.1f%%')
plt.title("Top 10 Predicted Extra text")
plt.axis('equal')
plt.show()


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

# Assuming you have loaded your dataset in the variable 'dt'
# Replace any missing values in the 'MTBF' column with the mean value
#imputer = SimpleImputer(strategy='mean')
#dt['MTBF'] = imputer.fit_transform(dt[['MTBF']])

# Separate features (X) from the target variable (y)
X = dt[['Stop group', 'Stop', 'Stop Location', 'MTBF']]  # Add the 'MTBF' column
y = dt['Extra text']

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X['Stop group'] + ' ' + X['Stop'] + ' ' + X['Stop Location'])

# Add the 'MTBF' column to the vectorized data
X_vec = np.hstack((X_vec.toarray(), X['MTBF'].values.reshape(-1, 1)))

# Divide the data into training and test sets while preserving the order
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, random_state=np.random, train_size=0.7)



# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


# Preprocess the Data: Handle missing values
imputer = SimpleImputer(strategy='mean')  # You can choose other strategies as well
X_train_imputed = imputer.fit_transform(X_train)

# Train a Multi-Layer Perceptron classifier (Neural Network)
model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42)
model.fit(X_train_imputed, y_train)

# Now you can proceed with the rest of your code
stop_group_input = input("Please enter the value of 'Stop group': ")
stop_input = input("Please enter the value of 'Stop': ")
stop_location_input = input("Please enter the value of 'Stop Location': ")

input_vec = vectorizer.transform([stop_group_input + ' ' + stop_input + ' ' + stop_location_input])
predicted_extra_text_probs = model.predict_proba(input_vec)[0]
predicted_extra_text = model.classes_

# Convert the predicted probabilities and values to a dictionary
predicted_values = dict(zip(predicted_extra_text, predicted_extra_text_probs))

# Sort the predicted values based on probabilities in descending order
sorted_values = sorted(predicted_values.items(), key=lambda x: x[1], reverse=True)

# Separate the top 10 predicted values and probabilities
top_10_values = [value[0] for value in sorted_values[:10]]
top_10_probs = [value[1] for value in sorted_values[:10]]

# Pie chart visualization
plt.figure(figsize=(8, 8))
plt.pie(top_10_probs, labels=top_10_values, autopct='%1.1f%%')
plt.title("Top 10 Predicted Extra text")
plt.axis('equal')
plt.show()


# In[ ]:




