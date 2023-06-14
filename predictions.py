import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import pickle
import plotly

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV


#model_path = r"C:\Users\kervi\OneDrive\Documents\GitHub\kervindholah\streamlit_app\ridge_model.joblib"
#ridge_model = load(model_path)


#with open(r"C:\Users\kervi\Documents\DataScientest - PROJETS\PROJET BIEN-ETRE MACHINE LEARNING\ridge_model.pkl", 'rb') as f:
#    model = pickle.load(f)

fp = r"C:\Users\kervi\PycharmProjects\ProjetStreamlit\projet_world_happiness_pour_machine_learning.csv"
df = pd.read_csv(fp)

score_df = df["Life_Ladder"]

# Style personnalisé pour la colonne "Life_Ladder"
def highlight_target(s):
    if s.name == "Life_Ladder":
        return ['background-color: yellow'] * len(s)
    else:
        return [''] * len(s)


# Affichage du dataframe
# Affichage du DataFrame avec style personnalisé
st.dataframe(df.style.apply(highlight_target, axis=0))

# Options de tri
sort_options = ['Ordre croissant', 'Ordre décroissant']






# Liste déroulante pour choisir le Country_name
country_name = st.selectbox('Choisir un Country_name:', df['Country_name'].unique())

# Affichage du dataframe correspondant au Country_name choisi
filtered_df = df[df['Country_name'] == country_name]

# Affichage du DataFrame filtré avec style personnalisé
st.dataframe(filtered_df.style.apply(highlight_target, axis=0))




# -----------------------------------------------------------
# intéractivité

filepath = r"C:\Users\kervi\PycharmProjects\ProjetStreamlit\df_encoded_streamlit.csv"
df_encoded = pd.read_csv(filepath)


st.write("Choisissez un pays pour pouvoir changer les paramètres :")
selected_country = st.selectbox('Country_name', df['Country_name'].unique())

# Filtrer le DataFrame pour obtenir la ligne correspondante à l'année 2021 ou la plus élevée si non disponible
filtered_row = df[df['Country_name'] == selected_country].nlargest(1, 'year')
score = filtered_row["Life_Ladder"]
filtered_row = filtered_row.drop(['Life_Ladder'], axis=1)
# filtered_row = filtered_row.drop(['year', 'Country_name'], axis=1)

selected_index = filtered_row.index[0]  # Obtenir l'index de la ligne filtrée

# Obtenir la ligne correspondante dans df_encoded
#encoded_row = df_encoded.loc[selected_index]
encoded_row = df_encoded[df_encoded.index == selected_index]

# Afficher la ligne correspondante
st.write(encoded_row)
#st.dataframe(encoded_row)
# st.dataframe(encoded_row.style.apply(highlight_target, axis=0))

# Affichage de la valeur de "Life_Ladder" séparée
st.write("Valeur de Life_Ladder : ", score)


# Séparer les caractéristiques en deux groupes
features_slider = encoded_row.columns[:6]
features_range = encoded_row.columns[6:]

if 'Log_GDP_per_capita' in features_slider:
    current_value1 = encoded_row['Log_GDP_per_capita']
    new_value1 = st.slider("Entrez une nouvelle valeur pour Log_GDP_per_capita", min_value=0.0, max_value=20.0, value=float(current_value1))
    encoded_row['Log_GDP_per_capita'] = new_value1

if 'Social_support' in features_slider:
    current_value2 = encoded_row['Social_support']
    new_value2 = st.slider("Entrez une nouvelle valeur pour Social_support", min_value=0.0, max_value=1.0, value=float(current_value2))
    encoded_row['Social_support'] = new_value2

if 'Healthy_life_expectancy_at_birth' in features_slider:
    current_value3 = encoded_row['Healthy_life_expectancy_at_birth']
    new_value3 = st.slider("Entrez une nouvelle valeur pour Healthy_life_expectancy_at_birth", min_value=20.0, max_value=100.0, value=float(current_value3))
    encoded_row['Healthy_life_expectancy_at_birth'] = new_value3

if 'Freedom_to_make_life_choices' in features_slider:
    current_value4 = encoded_row['Freedom_to_make_life_choices']
    new_value4 = st.slider("Entrez une nouvelle valeur pour Freedom_to_make_life_choices", min_value=0.0, max_value=1.0, value=float(current_value4))
    encoded_row['Freedom_to_make_life_choices'] = new_value4

if 'Generosity' in features_slider:
    current_value5 = encoded_row['Generosity']
    new_value5 = st.slider("Entrez une nouvelle valeur pour Generosity", min_value=-1.0, max_value=1.0, value=float(current_value5))
    encoded_row['Generosity'] = new_value5

if 'Perceptions_of_corruption' in features_slider:
    current_value6 = encoded_row['Perceptions_of_corruption']
    new_value6 = st.slider("Entrez une nouvelle valeur pour Perceptions_of_corruption", min_value=0.0, max_value=1.0, value=float(current_value6))
    encoded_row['Perceptions_of_corruption'] = new_value6


for feature in features_range:
    current_value = encoded_row[feature]
    new_value = st.slider(f"Entrez une nouvelle valeur pour {feature}", min_value=0.0, max_value=1.0, value=float(current_value))
    encoded_row[feature] = new_value


# Afficher la ligne correspondante mise à jour
#st.write(encoded_row)
encoded_row_modified = pd.DataFrame(encoded_row)
st.write(encoded_row_modified)

# Afficher un message pour rafraîchir la page
st.info("Rafraîchissez la page pour récupérer les valeurs d'origine")

# Afficher un message pour obtenir la prédiction
st.info("Cliquer sur 'prédiction' pour obtenir le nouveau score du bonheur")

# Afficher le bouton "Prédiction"
ok = st.button("Prédiction")

# st.write(encoded_row_modified.index[0])
index_b = encoded_row_modified.index[0]
# st.write(df_encoded[df_encoded.index == index_b])

if ok:
    df_encoded.loc[index_b] = encoded_row_modified.iloc[0]
    scaler = StandardScaler()
    df_encoded_scaled = scaler.fit_transform(df_encoded)
    choice_scaled = scaler.transform(encoded_row_modified)

    ridge_model = Ridge()
    ridge_model.fit(df_encoded_scaled, score_df)

    # Définir la grille d'hyperparamètres
    alpha_grid = [0.1, 1.0, 10.0]

    # Créer l'estimateur Ridge
    ridge = Ridge()

    # Créer l'objet GridSearchCV
    grid_search = GridSearchCV(estimator=ridge, param_grid={'alpha': alpha_grid}, cv=5)

    # Effectuer la recherche des hyperparamètres
    grid_search.fit(df_encoded_scaled, score_df)

    # Obtenir les meilleurs hyperparamètres
    best_alpha = grid_search.best_params_['alpha']

    # Réentraîner le modèle Ridge avec les meilleurs hyperparamètres
    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(df_encoded_scaled, score_df)


    prediction = ridge_model.predict(choice_scaled)
#    prediction = model.predict(choice_scaled)
    st.write("Résultat de la prédiction de Life Ladder :", prediction)





