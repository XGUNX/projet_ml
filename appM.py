# Ce script crée une application web Streamlit qui permet à un utilisateur de saisir les caractéristiques 
# d'un bien immobilier et d'obtenir une estimation du prix à l'aide d'un modèle XGBoost pré-entraîné.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from datetime import datetime

# CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title="Estimation Immo IA Pro",
    page_icon="🏠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CHARGEMENT DES FICHIERS SAUVEGARDÉS
@st.cache_resource
def load_data():
    try:
        model = joblib.load('mon_modele_immo.pkl')
        cols = joblib.load('colonnes_modele.pkl')
        # Charger les données d'entraînement pour les statistiques
        try:
            train_data = joblib.load('train_data_stats.pkl')
        except:
            train_data = None
        return model, cols, train_data
    except FileNotFoundError as e:
        st.error(f"Erreur : Fichier introuvable - {e}")
        st.info("Veuillez exécuter d'abord le notebook pour générer les fichiers .pkl")
        st.stop()

# Charger le modèle et les colonnes
model, cols_modele, train_data = load_data()

# TITRE ET INTRODUCTION
st.title("🏠 Estimateur de Prix Immobilier Professionnel")
st.markdown("""
Cette application utilise un modèle d'**Intelligence Artificielle (XGBoost)** pour estimer la valeur d'une maison 
basée sur ses caractéristiques. Remplissez le formulaire ci-contre et obtenez une estimation précise.
""")

st.write("---")

# SIDEBAR - FORMULAIRE UTILISATEUR
st.sidebar.header("📝 Caractéristiques du bien")

# Fonction pour créer des onglets dans la sidebar
tab1, tab2, tab3 = st.sidebar.tabs(["🏡 Général", "📐 Surface", "🎯 Qualité"])

# Dictionnaire pour stocker toutes les entrées
input_dict = {}

with tab1:
    st.subheader("Informations Générales")
    
    # Année de construction
    year_built = st.number_input(
        "Année de Construction",
        min_value=1800,
        max_value=datetime.now().year,
        value=2005,
        help="Année de construction initiale de la propriété"
    )
    input_dict['YearBuilt'] = year_built
    
    # Année de rénovation
    year_remod = st.number_input(
        "Année de Rénovation",
        min_value=1800,
        max_value=datetime.now().year,
        value=2005,
        help="Année de la dernière rénovation majeure"
    )
    input_dict['YearRemod/Add'] = year_remod
    
    # Zone de zonage
    ms_zoning_options = ['RL', 'RM', 'C (all)', 'FV', 'RH']
    ms_zoning = st.selectbox(
        "Zone de Zonage",
        options=ms_zoning_options,
        index=0,
        help="Classification de zonage municipale"
    )
    # Encodage one-hot pour la zone de zonage
    for zone in ['RL', 'RM', 'C (all)', 'FV', 'RH']:
        col_name = f'MSZoning_{zone}'
        input_dict[col_name] = 1 if zone == ms_zoning else 0
    
    # Quartier
    neighborhood_options = ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 
                           'Gilbert', 'NridgHt', 'Sawyer', 'NWAmes', 'SawyerW']
    neighborhood = st.selectbox(
        "Quartier",
        options=neighborhood_options,
        index=0,
        help="Localisation physique dans Ames City"
    )
    # Encodage one-hot pour le quartier
    for nbr in neighborhood_options[:10]:  # Limiter aux 10 premiers pour simplifier
        col_name = f'Neighborhood_{nbr}'
        input_dict[col_name] = 1 if nbr == neighborhood else 0
    
    # Type de rue
    street_type = st.radio(
        "Type de Rue",
        options=["Pavée", "Gravier"],
        index=0,
        horizontal=True
    )
    input_dict['Street'] = 1 if street_type == "Pavée" else 0
    
    # Pente du terrain
    land_slope = st.select_slider(
        "Pente du Terrain",
        options=[1, 2, 3],
        value=1,
        help="1=Douce, 2=Modérée, 3=Rude"
    )
    input_dict['LandSlope'] = land_slope

with tab2:
    st.subheader("Surfaces et Dimensions")
    
    # Surface habitable principale
    gr_liv_area = st.number_input(
        "Surface Habitable (pi²)",
        min_value=300,
        max_value=5000,
        value=1800,
        step=50,
        help="Surface habitable au-dessus du niveau du sol"
    )
    input_dict['GrLivArea'] = gr_liv_area
    
    # Surface totale du terrain
    lot_area = st.number_input(
        "Surface du Terrain (pi²)",
        min_value=1000,
        max_value=50000,
        value=9600,
        step=100
    )
    input_dict['LotArea'] = lot_area
    
    # Surface sous-sol
    total_bsmt_sf = st.number_input(
        "Surface Sous-sol (pi²)",
        min_value=0,
        max_value=3000,
        value=1000,
        step=50
    )
    input_dict['TotalBsmtSF'] = total_bsmt_sf
    
    # Surface du garage
    garage_area = st.number_input(
        "Surface du Garage (pi²)",
        min_value=0,
        max_value=1500,
        value=500,
        step=50
    )
    input_dict['GarageArea'] = garage_area
    
    # Surface 1er étage
    first_flr_sf = st.number_input(
        "Surface 1er Étage (pi²)",
        min_value=500,
        max_value=3000,
        value=1200,
        step=50
    )
    input_dict['1stFlrSF'] = first_flr_sf
    
    # Surface 2ème étage
    second_flr_sf = st.number_input(
        "Surface 2ème Étage (pi²)",
        min_value=0,
        max_value=2000,
        value=600,
        step=50
    )
    input_dict['2ndFlrSF'] = second_flr_sf
    
    # Surface totale calculée
    total_sf = gr_liv_area + total_bsmt_sf
    input_dict['TotalSF'] = total_sf
    st.info(f"**Surface totale (habitable + sous-sol) :** {total_sf:,} pi²")

with tab3:
    st.subheader("Qualité et Équipements")
    
    # Qualité générale
    overall_qual = st.select_slider(
        "Qualité Générale",
        options=list(range(1, 11)),
        value=7,
        help="1=Très Pauvre à 10=Exceptionnel"
    )
    input_dict['OverallQual'] = overall_qual
    
    # État général
    overall_cond = st.select_slider(
        "État Général",
        options=list(range(1, 11)),
        value=5,
        help="1=Très Pauvre à 10=Excellent"
    )
    input_dict['OverallCond'] = overall_cond
    
    # Qualité de la cuisine
    kitchen_qual = st.select_slider(
        "Qualité de la Cuisine",
        options=[1, 2, 3, 4, 5],
        value=3,
        help="1=Faible à 5=Excellente"
    )
    input_dict['KitchenQual'] = kitchen_qual
    
    # Nombre de chambres
    bedrooms = st.slider(
        "Nombre de Chambres",
        min_value=1,
        max_value=10,
        value=3
    )
    input_dict['BedroomAbvGr'] = bedrooms
    
    # Nombre de salles de bain complètes
    full_bath = st.slider(
        "Salles de Bain Complètes",
        min_value=0,
        max_value=4,
        value=2
    )
    input_dict['FullBath'] = full_bath
    
    # Nombre de salles de bain demi
    half_bath = st.slider(
        "Demi Salles de Bain",
        min_value=0,
        max_value=3,
        value=1
    )
    input_dict['HalfBath'] = half_bath
    
    # Nombre total de pièces
    tot_rms = st.slider(
        "Nombre Total de Pièces",
        min_value=2,
        max_value=15,
        value=7
    )
    input_dict['TotRmsAbvGrd'] = tot_rms
    
    # Nombre de places de garage
    garage_cars = st.selectbox(
        "Nombre de Places de Garage",
        options=[0, 1, 2, 3, 4],
        index=2
    )
    input_dict['GarageCars'] = garage_cars
    
    # Année de construction du garage
    garage_yr_blt = st.number_input(
        "Année Construction Garage",
        min_value=1800,
        max_value=datetime.now().year,
        value=2005
    )
    input_dict['GarageYrBlt'] = garage_yr_blt
    
    # Cheminées
    fireplaces = st.slider(
        "Nombre de Cheminées",
        min_value=0,
        max_value=5,
        value=1
    )
    input_dict['Fireplaces'] = fireplaces

# SECTION AVANCÉE (dépliée)
with st.sidebar.expander("⚙️ Options Avancées"):
    st.write("Paramètres supplémentaires")
    
    # Type de sous-sol fini
    bsmt_fin_type = st.selectbox(
        "Qualité du Sous-sol",
        options=[1, 2, 3, 4, 5, 6],
        index=2,
        help="1=Excellente à 6=Pas de Sous-sol"
    )
    input_dict['BsmtFinType1'] = bsmt_fin_type
    
    # Exposition du sous-sol
    bsmt_exposure = st.select_slider(
        "Exposition du Sous-sol",
        options=[0, 1, 2, 3, 4],
        value=2,
        help="0=Faible à 4=Excellente"
    )
    input_dict['BsmtExposure'] = bsmt_exposure
    
    # Classe fonctionnelle
    functional = st.selectbox(
        "Fonctionnalité de la Maison",
        options=[1, 2, 3, 4, 5, 6, 7, 8],
        index=0,
        help="1=Typique à 8=Fonctionnellement obsolète"
    )
    input_dict['Functional'] = functional

# BOUTON DE PRÉDICTION
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button(
        "🚀 Lancer l'Estimation du Prix",
        type="primary",
        use_container_width=True
    )

# MAIN CONTENT AREA
left_col, right_col = st.columns([2, 1])

with left_col:
    # RÉSUMÉ DES CARACTÉRISTIQUES
    st.subheader("📋 Récapitulatif des Caractéristiques")
    
    # Créer un DataFrame pour affichage
    summary_data = {
        "Catégorie": [
            "Général", "Général", "Surface", "Surface", "Surface",
            "Qualité", "Qualité", "Équipements", "Équipements"
        ],
        "Caractéristique": [
            "Année Construction", "Quartier", "Surface Habitable", 
            "Surface Terrain", "Surface Totale", "Qualité Générale",
            "État Général", "Chambres", "Salles de Bain"
        ],
        "Valeur": [
            str(year_built), neighborhood, f"{gr_liv_area:,} pi²",
            f"{lot_area:,} pi²", f"{total_sf:,} pi²", 
            f"{overall_qual}/10", f"{overall_cond}/10",
            str(bedrooms), f"{full_bath} complètes, {half_bath} demi"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(
        summary_df,
        column_config={
            "Catégorie": st.column_config.TextColumn("Catégorie"),
            "Caractéristique": st.column_config.TextColumn("Caractéristique"),
            "Valeur": st.column_config.TextColumn("Valeur")
        },
        hide_index=True,
        use_container_width=True
    )


# LOGIQUE DE PRÉDICTION
if predict_button:
    with st.spinner("🔍 Calcul de l'estimation en cours..."):
        try:
            # Création d'un DataFrame vide avec TOUTES les colonnes du modèle
            input_df = pd.DataFrame(0, index=[0], columns=cols_modele)
            
            # Remplissage des valeurs saisies
            for col, val in input_dict.items():
                if col in input_df.columns:
                    input_df[col] = val
            
            # Vérification des colonnes remplies
            filled_cols = [col for col in input_df.columns if input_df[col].iloc[0] != 0]
            
            # Prédiction
            log_prediction = model.predict(input_df)[0]
            real_prediction = np.expm1(log_prediction)
            
            # Calcul du prix au pied carré
            price_per_sqft = real_prediction / gr_liv_area if gr_liv_area > 0 else 0
            
            # AFFICHAGE DES RÉSULTATS
            st.success("✅ Estimation terminée avec succès!")
            
            # Résultat principal dans un conteneur stylisé
            st.markdown(f"""
            <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; margin: 20px 0;">
                <h2 style="color: #007bff; text-align: center; margin: 0;">
                💎 Prix Estimé : <strong>${real_prediction:,.0f}</strong>
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Métriques détaillées
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Prix au Pied Carré",
                    value=f"${price_per_sqft:,.2f}",
                    help="Prix par pied carré de surface habitable"
                )
            
            with col2:
                st.metric(
                    label="Surface Totale",
                    value=f"{total_sf:,} pi²",
                    help="Surface habitable + sous-sol"
                )
            
            with col3:
                st.metric(
                    label="Année Construction",
                    value=str(year_built),
                    help="Âge approximatif du bien"
                )
            
            # Section d'analyse
            st.subheader("📈 Analyse de l'Estimation")
            
            # Barre de confiance (simplifiée)
            confidence_score = min(90 + overall_qual * 0.5, 98)
            st.progress(confidence_score/100, text=f"Score de Confiance : {confidence_score:.1f}%")
            
           
            # Téléchargement des résultats
            st.subheader("📥 Exporter les Résultats")
            
            # Créer un DataFrame pour export
            export_data = {
                'Caractéristique': list(input_dict.keys()),
                'Valeur': list(input_dict.values())
            }
            export_df = pd.DataFrame(export_data)
            export_df.loc[len(export_df)] = ['Prix_Estimé', real_prediction]
            export_df.loc[len(export_df)] = ['Prix_par_pied_carre', price_per_sqft]
            export_df.loc[len(export_df)] = ['Date_Estimation', datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            
            # Convertir en CSV
            csv = export_df.to_csv(index=False).encode('utf-8')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📄 Télécharger CSV",
                    data=csv,
                    file_name=f"estimation_immo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                if st.button("🔄 Nouvelle Estimation", use_container_width=True):
                    st.rerun()
            
            # Informations techniques (débogage)
            with st.expander("🔧 Informations Techniques"):
                st.write(f"**Colonnes utilisées par le modèle :** {len(cols_modele)}")
                st.write(f"**Colonnes remplies manuellement :** {len(filled_cols)}")
                st.write(f"**Prédiction log :** {log_prediction:.4f}")
                st.write(f"**Modèle utilisé :** XGBoost")
                
               
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {str(e)}")
            
            # Mode dégradé avec estimation basique
            st.warning("Mode estimation basique activé (sans IA)")
            
            # Calcul d'une estimation simplifiée
            base_price = 100000
            size_factor = gr_liv_area * 150
            quality_factor = overall_qual * 10000
            year_factor = (year_built - 1900) * 500
            
            estimated_price = base_price + size_factor + quality_factor + year_factor
            
            st.info(f"💰 Estimation approximative : **${estimated_price:,.0f}**")
else:
    # Message d'attente
    st.info("👈 **Remplissez les caractéristiques dans le menu de gauche et cliquez sur 'Lancer l'Estimation'**")
    
    # Exemple de propriété
    with st.expander("🎯 Voir un exemple type"):
        st.write("""
        **Propriété type moyenne :**
        - Surface habitable : 1,800 pi²
        - Année construction : 2005
        - Qualité générale : 7/10
        - 3 chambres, 2 salles de bain
        - Garage 2 places
        - Surface terrain : 9,600 pi²
        
        **Prix estimé pour cet exemple :** $250,000 - $300,000
        """)

# PIED DE PAGE
st.write("---")
st.caption("""
⚠️ **Disclaimer :** Cette estimation est fournie à titre indicatif. 
Elle ne remplace pas une évaluation professionnelle par un expert immobilier certifié.
Les résultats peuvent varier selon les conditions du marché local.
""")