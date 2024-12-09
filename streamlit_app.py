import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuration de la page Streamlit
st.set_page_config(page_title="Prédiction des prix immobiliers", layout="wide")

# Fonction pour charger le modèle Ridge (mise en cache)
@st.cache_resource
def load_ridge_model():
    return joblib.load('code/ridge_model.pkl')  # Remplacez par le chemin réel de votre modèle

ridge_model = load_ridge_model()

pipeline = joblib.load('code/pipeline.pkl')

# Fonction pour charger les données (mise en cache)
@st.cache_data
def load_data():
    return pd.read_csv("data/train_df.csv")  # Remplacez par le chemin réel de vos données

data = load_data()
labels = {
    "MSSubClass": "Type de logement impliqué dans la vente",
    "MSZoning": "Classification générale de zonage de la vente",
    "LotFrontage": "Longueur linéaire de la rue connectée à la propriété (en pieds)",
    "LotArea": "Surface du lot (en pieds carrés)",
    "Street": "Type d'accès routier à la propriété",
    "LotShape": "Forme générale de la propriété",
    "LandContour": "Niveau de platitude de la propriété",
    "Utilities": "Type de services publics disponibles",
    "LotConfig": "Configuration du lot",
    "LandSlope": "Pente de la propriété",
    "Neighborhood": "Localisation physique à l'intérieur des limites de la ville d'Ames",
    "Condition1": "Proximité à différentes conditions",
    "Condition2": "Proximité à d'autres conditions (si plus d'une est présente)",
    "BldgType": "Type de logement",
    "HouseStyle": "Style du logement",
    "OverallQual": "Qualité générale des matériaux et finitions de la maison",
    "OverallCond": "État général de la maison",
    "RoofStyle": "Style de toit",
    "RoofMatl": "Matériau du toit",
    "Exterior1st": "Revêtement extérieur sur le premier étage",
    "Exterior2nd": "Revêtement extérieur secondaire (si présent)",
    "MasVnrType": "Type de revêtement de maçonnerie",
    "MasVnrArea": "Surface du revêtement de maçonnerie (en pieds carrés)",
    "ExterQual": "Qualité générale des matériaux extérieurs",
    "ExterCond": "État général des matériaux extérieurs",
    "Foundation": "Type de fondation",
    "BsmtQual": "Hauteur du sous-sol",
    "BsmtCond": "État général du sous-sol",
    "BsmtExposure": "Degré d'exposition du sous-sol au niveau du sol",
    "BsmtFinType1": "Qualité du sous-sol aménagé",
    "BsmtUnfSF": "Surface non aménagée du sous-sol (en pieds carrés)",
    "Heating": "Type de système de chauffage",
    "HeatingQC": "Qualité et état du système de chauffage",
    "CentralAir": "Climatisation centrale (Y: Oui, N: Non)",
    "Electrical": "Type de système électrique",
    "LowQualFinSF": "Surface finie de faible qualité (en pieds carrés)",
    "BedroomAbvGr": "Nombre de chambres au-dessus du niveau du sol",
    "KitchenAbvGr": "Nombre de cuisines au-dessus du niveau du sol",
    "KitchenQual": "Qualité de la cuisine",
    "TotRmsAbvGrd": "Nombre total de pièces au-dessus du niveau du sol (hors salles de bain)",
    "Functional": "Note de fonctionnalité de la maison",
    "Fireplaces": "Nombre de cheminées",
    "FireplaceQu": "Qualité des cheminées",
    "GarageType": "Localisation du garage",
    "GarageFinish": "Qualité intérieure du garage",
    "GarageCars": "Capacité du garage (en nombre de voitures)",
    "GarageQual": "Qualité générale du garage",
    "PavedDrive": "Type d'allée pavée",
    "PoolArea": "Surface de la piscine (en pieds carrés)",
    "MiscVal": "Valeur des caractéristiques diverses (en dollars)",
    "MoSold": "Mois de la vente",
    "SaleType": "Type de vente",
    "SaleCondition": "Condition de la vente",
    "SalePrice": "Prix de vente de la maison",
    "houseage": "Âge de la maison (en années)",
    "houseremodelage": "Âge depuis la rénovation (en années)",
    "totalsf": "Surface totale finie (en pieds carrés)",
    "totalarea": "Surface totale (en pieds carrés)",
    "totalbaths": "Nombre total de salles de bain",
    "totalporchsf": "Surface totale des porches (en pieds carrés)"
}

data=data.rename(columns=labels)
# Initialisation de l'état de la page (si ce n'est pas déjà fait)
if "page" not in st.session_state:
    st.session_state.page = "Accueil"

# Titre de l'application
st.title("Application de prédiction des prix immobiliers")

# Fonction pour changer la page active dans st.session_state
def set_page(page_name):
    st.session_state.page = page_name

# Barre de navigation horizontale avec des boutons
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🏠 Accueil"):
        set_page("Accueil")
with col2:
    if st.button("📊 Analyse"):
        set_page("Analyse")
with col3:
    if st.button("🔍 Prédiction"):
        set_page("Prédiction")
with col4:
    if st.button("📈 Performance"):
        set_page("Performance")

# Section Accueil
if st.session_state.page == "Accueil":
    st.subheader("🏠 Bienvenue dans l'application de prédiction des prix immobiliers")
    st.write("""
        Cette application vous permet de :
        - Prédire les prix des maisons
        - Analyser les données des prix immobiliers
        - Évaluer les performances des modèles de prédiction
    """)

# Section Analyse des données
elif st.session_state.page == "Analyse":
    st.subheader("📊 Analyse des Données")
    st.write("Exploration des données des prix immobiliers.")

    # Affichage des données brutes si l'option est activée
    if st.checkbox("Afficher les données brutes"):
        st.subheader("Données des prix immobiliers")
        st.dataframe(data)

    # Statistiques descriptives
    st.write("### Statistiques descriptives")
    st.write(data.describe())

    # Sélection des variables pour la visualisation
    st.write("### Visualisation de deux variables")
    variable_x = st.selectbox("Sélectionnez la première variable (axe X)", data.columns)
    variable_y = st.selectbox("Sélectionnez la deuxième variable (axe Y)", data.columns)

    # Visualisation des relations entre les variables
    fig, ax = plt.subplots(figsize=(10, 8))
    if data[variable_x].dtype in ['int64', 'float64'] and data[variable_y].dtype in ['int64', 'float64']:
        sns.scatterplot(data=data, x=variable_x, y=variable_y, ax=ax, color="teal", s=100, edgecolor='black')
        ax.set_title(f"Nuage de points entre {variable_x} et {variable_y}", fontsize=16, fontweight='bold')
        ax.set_xlabel(variable_x, fontsize=14)
        ax.set_ylabel(variable_y, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
    elif data[variable_x].dtype == 'object' and data[variable_y].dtype == 'object':
        grouped_data = data.groupby([variable_x, variable_y]).size().unstack()
        grouped_data.plot(kind='bar', stacked=True, ax=ax, cmap='coolwarm')
        ax.set_title(f"Graphique en barres empilées de {variable_x} par {variable_y}", fontsize=16, fontweight='bold')
        ax.set_xlabel(variable_x, fontsize=14)
        ax.set_ylabel("Effectifs", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(title=variable_y, fontsize=12)
    else:
        sns.boxplot(data=data, x=variable_x, y=variable_y, ax=ax, palette="Set2")
        ax.set_title(f"Graphique de boîte de {variable_y} par {variable_x}", fontsize=16, fontweight='bold')
        ax.set_xlabel(variable_x, fontsize=14)
        ax.set_ylabel(variable_y, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

    st.pyplot(fig)

    # Matrice de corrélation
    st.write("### Matrice de Corrélation")
    correlation_matrix = data.select_dtypes(include=['int64', 'float64']).corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    fig_corr, ax_corr = plt.subplots(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".1f", ax=ax_corr, cbar=True, annot_kws={'size': 10}, mask=mask)
    ax_corr.set_title("Matrice de Corrélation")
    st.pyplot(fig_corr)

# Section Prédiction des prix
elif st.session_state.page == "Prédiction":
    st.subheader("🔍 Prédiction des Prix")
    st.write("Utilisez ce formulaire pour entrer les valeurs des caractéristiques et prédire le prix d'une maison.")

    # Formulaire de saisie
    form_data = {}
    for col in data.columns.drop("Prix de vente de la maison"):
        if data[col].dtype == 'object':
            # Champ de saisie de texte pour les variables catégorielles
            form_data[col] = st.selectbox(f"{col}", data[col].unique())
        elif data[col].dtype in ['int64', 'float64']:
            # Champ de saisie numérique pour les variables numériques
            min_val = data[col].min()
            max_val = data[col].max()
            form_data[col] = st.number_input(f"{col}", min_value=float(min_val), max_value=float(max_val), value=float(min_val))

    # Bouton pour lancer la prédiction
    if st.button("Prédire le Prix"):
        st.write("Lancer la prédiction avec les valeurs suivantes :")
        input_data = pd.DataFrame([form_data])
        st.write("Vérification des données d'entrée avant prédiction :", input_data)

        # Prédiction
        try:
            input_data = pipeline.transform(input_data)
            predicted_price = np.expm1(ridge_model.predict(input_data))
            st.write(f"Le prix prédit par le modèle Ridge est : {predicted_price[0]:,.2f} unité monétaire")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# Section Performance
elif st.session_state.page == "Performance":
    st.subheader("📈 Évaluation des Performances du Modèle")
    st.write("Examinez les performances des modèles utilisés pour la prédiction des prix.")

    # Calcul de la performance sur un jeu de test
    data2=data
    data2 = pipeline.transform(data2)
    X_test = data2.drop(columns=["Prix de vente de la maison"])  # Remplacer "price" par la colonne cible
    y_test = data2["Prix de vente de la maison"]  # Assurez-vous que "price" est la colonne cible

    y_pred = ridge_model.predict(X_test)

    # Affichage des métriques de performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    st.write(f"Erreur Absolue Moyenne (MAE) : {mae:,.2f}")
    st.write(f"Erreur Quadratique Moyenne (MSE) : {mse:,.2f}")
    st.write(f"Racine de l'Erreur Quadratique Moyenne (RMSE) : {rmse:,.2f}")
