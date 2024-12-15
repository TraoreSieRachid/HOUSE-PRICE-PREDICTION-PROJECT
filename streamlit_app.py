import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns # type: ignore
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
ridge_cv_performance = joblib.load('code/ridge_cv_performance.pkl')
lr_performance = joblib.load('code/lr_performance.pkl')
cat_cv_performance = joblib.load('code/cat_cv_performance.pkl')
GBR_cv_performance = joblib.load('code/GBR_cv_performance.pkl')
lgbm_cv_performance = joblib.load('code/lgbm_cv_performance.pkl')
vr_performance = joblib.load('code/vr_performance.pkl')
stackreg_performance = joblib.load('code/stackreg_performance.pkl')
xgb_cv_performance = joblib.load('code/xgb_cv_performance.pkl')
rfr_cv_performance = joblib.load('code/rfr_cv_performance.pkl')
# Fonction pour charger les données (mise en cache)
@st.cache_data
def load_data1():
    return pd.read_csv("data/train_df.csv")  # Remplacez par le chemin réel de vos données

def load_data2():
    return pd.read_csv("data/test_df.csv")  # Remplacez par le chemin réel de vos données

def load_data3():
    return pd.read_csv("data/train_df_labelled.csv")  # Remplacez par le chemin réel de vos données

def load_data4():
    return pd.read_csv("data/test_df_labelled.csv")  # Remplacez par le chemin réel de vos données

train_df = load_data1()
test_df = load_data2()
train_df_labelled= load_data3()
test_df_labelled= load_data4()

# Initialisation de l'état de la page (si ce n'est pas déjà fait)
if "page" not in st.session_state:
    st.session_state.page = "Accueil"

# Titre de l'application
st.title("🏡 **Application de Prédiction des Prix Immobiliers**")

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
    st.write("---")  # Ligne de séparation

    # Sous-titre de bienvenue
    st.header("Bienvenue 👋")
    st.write("""
        Cette application vous offre des outils intuitifs pour :
        - 🏘️ Prédire les **prix des maisons** à partir de caractéristiques clés.
        - 📊 Analyser les **tendances des prix immobiliers** dans des zones spécifiques.
        - 🛠️ Évaluer les **performances des modèles de prédiction** et leurs ajustements.
    """)

    # Description des données
    st.header("📂 Description des Données Brutes")
    st.write("""
        Voici un aperçu des données utilisées pour nos analyses et prédictions. 
        Ces informations décrivent les différentes variables et leurs catégories associées.
    """)

    # Chemin vers le fichier texte
    file_path = "Ressources/data_description.txt"

    # Lire et afficher le fichier
    try:
        with open(file_path, "r") as file:
            description = file.read()
    except FileNotFoundError:
        st.error(f"🚨 Le fichier '{file_path}' est introuvable. Veuillez vérifier son emplacement.")
        st.stop()

    # Texte enrichi avec une barre défilante
    st.text_area(
        "🔍 **Aperçu du fichier de description :**",
        description,
        height=300,
        placeholder="Le contenu du fichier sera affiché ici..."
    )

    # Télécharger le fichier texte
    st.write("""
        📥 **Téléchargez la description complète des données en cliquant ci-dessous :**
    """)
    st.download_button(
        label="Télécharger le fichier de description 📄",
        data=description,
        file_name="description.txt",
        mime="text/plain",
        help="Téléchargez le fichier texte contenant les descriptions des données utilisées dans l'application."
    )

    # Complément sur l'utilisation
    st.write("---")
    st.info("""
        ℹ️ **Astuce :** Utilisez la barre de navigation pour explorer les fonctionnalités comme la prédiction des prix ou l'analyse des performances des modèles.
    """)

# Section Analyse des données
elif st.session_state.page == "Analyse":
    st.subheader("📊 Analyse des Données")
    st.write("Exploration des données des prix immobiliers.")
    st.write("---")

    # Affichage des données brutes si l'option est activée
    if st.checkbox("Afficher les données brutes"):
        st.subheader("Données des prix immobiliers")
        st.dataframe(train_df_labelled)
    st.write("---")

    # Statistiques descriptives
    st.write("### Statistiques descriptives")
    st.write(train_df_labelled.describe())
    st.write("---")

    # Sélection des variables pour la visualisation
    st.write("### Visualisation de deux variables")
    variable_x = st.selectbox("Sélectionnez la première variable (axe X)", train_df_labelled.columns)
    variable_y = st.selectbox("Sélectionnez la deuxième variable (axe Y)", train_df_labelled.columns)

    # Visualisation des relations entre les variables
    fig, ax = plt.subplots(figsize=(10, 8))
    if train_df_labelled[variable_x].dtype in ['int64', 'float64'] and train_df_labelled[variable_y].dtype in ['int64', 'float64']:
        sns.scatterplot(train_df_labelled=train_df_labelled, x=variable_x, y=variable_y, ax=ax, color="teal", s=100, edgecolor='black')
        ax.set_title(f"Nuage de points entre {variable_x} et {variable_y}", fontsize=16, fontweight='bold')
        ax.set_xlabel(variable_x, fontsize=14)
        ax.set_ylabel(variable_y, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
    elif train_df_labelled[variable_x].dtype == 'object' and train_df_labelled[variable_y].dtype == 'object':
        grouped_train_df_labelled = train_df_labelled.groupby([variable_x, variable_y]).size().unstack()
        grouped_train_df_labelled.plot(kind='bar', stacked=True, ax=ax, cmap='coolwarm')
        ax.set_title(f"Graphique en barres empilées de {variable_x} par {variable_y}", fontsize=16, fontweight='bold')
        ax.set_xlabel(variable_x, fontsize=14)
        ax.set_ylabel("Effectifs", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(title=variable_y, fontsize=12)
    else:
        sns.boxplot(train_df_labelled=train_df_labelled, x=variable_x, y=variable_y, ax=ax, palette="Set2")
        ax.set_title(f"Graphique de boîte de {variable_y} par {variable_x}", fontsize=16, fontweight='bold')
        ax.set_xlabel(variable_x, fontsize=14)
        ax.set_ylabel(variable_y, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

    st.pyplot(fig)
    st.write("---")

    # Matrice de corrélation
    st.write("### Matrice de Corrélation")
    correlation_matrix = train_df_labelled.select_dtypes(include=['int64', 'float64']).corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    fig_corr, ax_corr = plt.subplots(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".1f", ax=ax_corr, cbar=True, annot_kws={'size': 10}, mask=mask)
    ax_corr.set_title("Matrice de Corrélation")
    st.pyplot(fig_corr)
    st.write("---")
    
# Section Prédiction des prix
elif st.session_state.page == "Prédiction":
    st.subheader("🔍 Prédiction des Prix")
    st.write("Utilisez ce formulaire pour entrer les valeurs des caractéristiques et prédire le prix d'une maison.")   
    # Création du formulaire
    form_data = {}
    for col_label in train_df_labelled.columns:
        col_index = train_df_labelled.columns.get_loc(col_label)
        col_raw=train_df.columns[col_index]

        if train_df_labelled[col_label].dtype == 'object':
            # Champ de sélection pour les variables catégorielles
            form_data[col_raw] = st.selectbox(f"{col_label}", train_df_labelled[col_label].unique())

        elif train_df_labelled[col_label].dtype in ['int64', 'float64']:
            # Champ de saisie numérique pour les variables numériques
            min_val = train_df_labelled[col_label].min()
            max_val = train_df_labelled[col_label].max()

            form_data[col_raw] = st.number_input(
                f"{col_label}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(min_val)
            )

    # Bouton pour lancer la prédiction
    if st.button("Prédire le Prix"):
        st.write("Lancer la prédiction avec les valeurs suivantes :")
        input_data = pd.DataFrame([form_data])
        col=train_df_labelled.columns
        input_data_label=input_data.rename(columns=col)
        st.write("Vérification des données d'entrée avant prédiction :", input_data_label)

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
    
    if st.checkbox("Afficher les données brutes de test"):
        st.subheader("Données des prix immobiliers")
        st.dataframe(test_df_labelled)
    st.write("---")

    # Liste des DataFrames
    dataframes = [ridge_cv_performance, lr_performance, cat_cv_performance,
                GBR_cv_performance, lgbm_cv_performance, vr_performance,
                stackreg_performance, xgb_cv_performance, rfr_cv_performance]

    # Liste des noms des modèles correspondants
    model_names = ["Ridge", "Linear Regression", "CatBoost", 
                "Gradient Boosting Regressor", "LightGBM", "Voting Regressor",
                "Stacking Regressor", "XGBoost", "Random Forest Regressor"]

        # Titre de l'application
    st.subheader("Visualisation des performances des modèles")

    # Affichage des données et graphiques dans une grille
    cols = st.columns(3)

    for i, (df, model_name) in enumerate(zip(dataframes, model_names)):
        col = cols[i % 3]  # Sélectionner la colonne appropriée
        with col:
            st.subheader(f"{model_name}")
            
            # Afficher le DataFrame
            st.dataframe(df)

            # Créer un graphique pour les performances
            fig, ax = plt.subplots(figsize=(6, 4))
            x = range(len(df))
            bar_width = 0.4

            ax.bar(x, df["train"], width=bar_width, label="Train", color="skyblue", align="center")
            ax.bar([p + bar_width for p in x], df["test"], width=bar_width, label="Test", color="salmon", align="center")
            ax.set_xticks([p + bar_width / 2 for p in x])
            ax.set_xticklabels(df["metric"], rotation=45, ha="right")
            ax.set_title(f"Performances : {model_name}")
            ax.legend()

            # Afficher le graphique
            st.pyplot(fig)
            col.write("---")