import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Estimation Immo IA", page_icon="🏠", layout="centered")

# --- CHARGEMENT DES FICHIERS SAUVEGARDÉS ---
@st.cache_resource # Cette ligne permet de garder le modèle en mémoire pour aller plus vite
def load_data():
    model = joblib.load('mon_modele_immo.pkl')
    cols = joblib.load('colonnes_modele.pkl')
    return model, cols

try:
    model, cols_modele = load_data()
except FileNotFoundError:
    st.error("❌ Erreur : Les fichiers .pkl sont introuvables. As-tu bien exécuté l'étape 1 ?")
    st.stop()

# --- TITRE ET INTRODUCTION ---
st.title("🏠 Estimateur de Prix Immobilier")
st.markdown("""
Cette application utilise un modèle d'Intelligence Artificielle (**XGBoost**) pour estimer la valeur d'une maison 
basée sur ses caractéristiques principales.
""")

st.write("---")

# --- FORMULAIRE UTILISATEUR (SIDEBAR) ---
st.sidebar.header("🛠️ Caractéristiques du bien")

def user_input_features():
    # On définit des valeurs par défaut moyennes
    
    # 1. Surface Habitable
    gr_liv_area = st.sidebar.slider("Surface Habitable (pieds²)", 500, 5000, 1500)
    
    # 2. Année de construction
    year_built = st.sidebar.slider("Année de Construction", 1900, 2024, 2005)
    
    # 3. Qualité Globale (Le facteur le plus important !)
    overall_qual = st.sidebar.select_slider("Qualité Générale (1=Ruine, 10=Luxe)", options=[1,2,3,4,5,6,7,8,9,10], value=6)
    
    # 4. Garage
    garage_cars = st.sidebar.selectbox("Nombre de places de Garage", [0, 1, 2, 3, 4], index=2)
    garage_area = 0
    if garage_cars > 0:
        garage_area = st.sidebar.number_input("Surface Garage (pieds²)", value=500)

    # 5. Sous-sol
    total_bsmt_sf = st.sidebar.number_input("Surface Sous-sol (pieds²)", value=800)
    
    # 6. Salles de bain
    full_bath = st.sidebar.slider("Salles de bain complètes", 0, 4, 2)
    
    # Bouton de validation
    return {
        'GrLivArea': gr_liv_area,
        'YearBuilt': year_built,
        'OverallQual': overall_qual,
        'GarageCars': garage_cars,
        'GarageArea': garage_area,
        'TotalBsmtSF': total_bsmt_sf,
        'FullBath': full_bath,
        # On calcule la variable "TotalSF" qu'on avait créée en Feature Engineering
        'TotalSF': gr_liv_area + total_bsmt_sf
    }

input_dict = user_input_features()

# --- LOGIQUE DE PRÉDICTION ---
# Bouton principal
if st.button("🚀 Lancer l'Estimation"):
    
    # 1. Création d'un DataFrame vide avec TOUTES les colonnes du modèle (remplies de 0)
    # C'est l'étape cruciale pour éviter l'erreur "Mismatch columns"
    input_df = pd.DataFrame(0, index=[0], columns=cols_modele)
    
    # 2. Remplissage des valeurs connues
    # On parcourt notre dictionnaire et on remplit le DataFrame
    for col, val in input_dict.items():
        if col in input_df.columns:
            input_df[col] = val
    
    # (Optionnel) Gestion simplifiée d'un quartier par défaut
    # Si ton modèle a besoin d'un quartier pour ne pas planter, on peut mettre 'NAmes' par défaut
    if 'Neighborhood_NAmes' in input_df.columns:
        input_df['Neighborhood_NAmes'] = 1
        
    # 3. Prédiction
    try:
        # Le modèle prédit un Log(Prix)
        log_prediction = model.predict(input_df)[0]
        
        # On convertit en Vrais Dollars
        real_prediction = np.expm1(log_prediction)
        
        # 4. Affichage du Résultat
        st.success(f"💎 Prix Estimé : **{real_prediction:,.0f} $**")
        
        # Détails supplémentaires
        st.info(f"Prix au pied carré estimé : **{real_prediction / input_dict['GrLivArea']:,.2f} $/pi²**")
        
        # Jauge visuelle (Barre de progression)
        st.write("Positionnement gamme de prix :")
        st.progress(min(int(real_prediction / 600000 * 100), 100)) # Supposons max 600k pour la barre
        
    except Exception as e:
        st.error(f"Une erreur est survenue lors du calcul : {e}")
        st.write("Détails techniques : Vérifie que les noms de colonnes dans app.py correspondent exactement à ceux du notebook.")

else:
    st.info("👈 Modifiez les paramètres dans le menu de gauche et cliquez sur 'Lancer l'Estimation'")
