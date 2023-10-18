import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sklearn
import lightgbm as lgb
from lightgbm import LGBMClassifier
import shap
from streamlit_shap import st_shap
import shap
import pickle
from PIL import Image

############################
# Configuration de la page #
############################
st.set_page_config(
        page_title='Pret prediction app',
        page_icon = 'home_credit_logo.png',
        layout="wide" )

# Définition de quelques styles css
# st.markdown(unsafe_allow_html=True)
new_title = '<p style="font-family:sans-serif; color:Green; font-size: 32px;">Zouheir HMIDI - Data Scientist - Projet 7 - OpenClassRooms</p>'
st.markdown(new_title, unsafe_allow_html=True)

# Centrage de l'image du logo dans la sidebar
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.sidebar.write("")
with col2:
    image = Image.open('logo.png')
    st.sidebar.image(image, use_column_width="always")
with col3:
    st.sidebar.write("")

    
@st.cache_data 
def lecture_X_test_original():
    X_test_original = pd.read_csv("test_orig.csv", low_memory=False)
    X_test_original = X_test_original.rename(columns=str.lower)
    return X_test_original

@st.cache_data 
def lecture_X_test_clean():
    X_test_clean = pd.read_csv("test_clean.csv", low_memory=False)
    
    return X_test_clean

@st.cache_data
def calcul_valeurs_shap():
    model_LGBM = pickle.load(open("trained_model.pkl", "rb"))
    explainer = shap.TreeExplainer(model_LGBM)
    shap_values = explainer.shap_values(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))
    return shap_values

if __name__ == "__main__":


    # Titre 1
    st.markdown("""
                <h1 style="color:#9c0418;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                Score Client ?</h1>
                """, 
                unsafe_allow_html=True)
    st.write("")

##########################################################
# Création et affichage du sélecteur du numéro de client #
##########################################################
    liste_clients = list(lecture_X_test_original()['sk_id_curr'])
    col1, col2 = st.columns(2) 
    with col1:
        ID_client = st.selectbox("ID Client ", 
                                (liste_clients))
        
    with col2:
        st.write("")

        
    
#################################################
# Lecture du modèle de prédiction et des scores #
#################################################
   
    model_LGBM = pickle.load(open("trained_model.pkl", "rb"))
    y_pred_lgbm = model_LGBM.predict(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))    
    y_pred_lgbm_proba = model_LGBM.predict_proba(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1)) 
    
    y_pred_lgbm_proba_df = pd.DataFrame(y_pred_lgbm_proba, columns=['proba_classe_0', 'proba_classe_1'])
    y_pred_lgbm_proba_df = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'],lecture_X_test_clean()['sk_id_curr']], axis=1)
    
   
    score = (y_pred_lgbm_proba_df[y_pred_lgbm_proba_df['sk_id_curr']==ID_client])
    
    score_value = round(score.proba_classe_1.iloc[0]*100, 2)
    

# Récupération de la décision
    y_pred_lgbm_df = pd.DataFrame(y_pred_lgbm, columns=['prediction'])
    y_pred_lgbm_df = pd.concat([y_pred_lgbm_df, lecture_X_test_clean()['sk_id_curr']], axis=1)
    y_pred_lgbm_df['client'] = np.where(y_pred_lgbm_df.prediction == 1, ':red[**non solvable**]', ':green[**solvable**]')
    y_pred_lgbm_df['decision'] = np.where(y_pred_lgbm_df.prediction == 1, ':red[**refuser**]', ':green[**accorder**]')
    solvabilite = y_pred_lgbm_df.loc[y_pred_lgbm_df['sk_id_curr']==ID_client, "client"].values
    decision = y_pred_lgbm_df.loc[y_pred_lgbm_df['sk_id_curr']==ID_client, "decision"].values
    
##############################################################
# Affichage du score et du graphique de gauge sur 2 colonnes #
##############################################################
    col1, col2 = st.columns(2)
    with col2:
        st.markdown(""" <br> <br> """, unsafe_allow_html=True)
        st.write(f"Le client dont l'identifiant est **{ID_client}** a obtenu le score de **{score_value:.1f}%**.")
        st.write(f"**Il y a donc un risque de {score_value:.1f}% que le client ait des difficultés de paiement.**")
        st.write(f"Le client est donc considéré comme **{solvabilite[0]}**.")
        st.write( f"Le crédit est **{decision[0]}**. ")
    # Impression du graphique jauge
    with col1:
        fig = go.Figure(go.Indicator(
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        value = float(score_value),
                        mode = "gauge+number+delta",
                        title = {'text': "Score du client", 'font': {'size': 24}},
                        # delta = {'reference': 35.2, 'increasing': {'color': "#9c0418"}},
                        gauge = {'axis': {'range': [None, 100],
                                'tickwidth': 3,
                                'tickcolor': 'darkblue'},
                                'bar': {'color': 'white', 'thickness' : 0.3},
                                'bgcolor': 'white',
                                'borderwidth': 1,
                                'bordercolor': 'gray',
                                'steps': [{'range': [0, 20], 'color': '#cbf078'},
                                        {'range': [20, 40], 'color': '#34c917'},
                                        {'range': [40, 60], 'color': '#a2c11c'},
                                        {'range': [60, 80], 'color': '#e8630a'},
                                        {'range': [80, 100], 'color': '#ff0000'}],
                                'threshold': {'line': {'color': 'white', 'width': 8},
                                            'thickness': 0.8,
                                            'value': 35.2 }}))

        fig.update_layout(paper_bgcolor='white',
                        height=400, width=500,
                        font={'color': '#9c0418', 'family': 'Roboto Condensed'},
                        margin=dict(l=30, r=30, b=5, t=5))
        st.plotly_chart(fig, use_container_width=True)

################################
# Explication de la prédiction #
################################
# Titre 2
    st.markdown("""
                <h1 style="color:#9c0418;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                Explication du calcul du score client ?</h1>
                """, 
                unsafe_allow_html=True)
    st.write("")

# Calcul des valeurs Shap
    explainer_shap = shap.TreeExplainer(model_LGBM)
    shap_values = explainer_shap.shap_values(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))


# # # récupération de l'index correspondant à l'identifiant du client
    idx = int(lecture_X_test_clean()[lecture_X_test_clean()['sk_id_curr']==ID_client].index[0])

    st_shap(shap.decision_plot(explainer_shap.expected_value[1],
                           shap_values[1][idx, :],
                           lecture_X_test_clean().drop(
                           labels="sk_id_curr", axis=1).iloc[idx, :],
                           feature_names=lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).columns.to_list(),
                           feature_order='importance',
                           # affichage des 10 variables les + importantes
                           feature_display_range=slice(None, -11, -1),
                           link='logit'))


#######################################################
# Récupération et affichage des informations du client #
########################################################

    data_client=lecture_X_test_original()[lecture_X_test_original().sk_id_curr == ID_client]
    
    col1, col2 = st.columns(2)
    with col1:
    # Titre H2
        st.markdown("""
                    <h2 style="color:#9c0418;text-align:left;font-size:1.8em;font-style:italic;font-weight:700;margin:0px;">
                    Profil socio-économique</h2>
                    """, 
                    unsafe_allow_html=True)
        st.write("")
        st.write(f"Genre : **{data_client['code_gender'].values[0]}**")
        st.write(f"Tranche d'âge : **{data_client['tranche_age'].values[0]}**")
        st.write(f"Situation familiale : **{data_client['name_family_status'].values[0]}**")
        st.write(f"Taille de la famille : **{data_client['cnt_fam_members'].values[0]}**")
        st.write(f"Nombre d'enfants : **{data_client['cnt_children'].values[0]}**")
    
        st.write(f"Revenu Total Annuel : **{data_client['amt_income_total'].values[0]} $**")
        st.write(f"Type d'emploi : **{data_client['name_income_type'].values[0]}**")
        st.write(f"Type d'organisation : **{data_client['organization_type'].values[0]}**")
        st.write(f"Ancienneté dans son entreprise actuelle : **{data_client['anciennete_entreprise'].values[0]}**")
        st.write(f"Type d'habitation : **{data_client['name_housing_type'].values[0]}**")
    
    
    with col2:
        # Titre H2
        st.markdown("""
                    <h2 style="color:#9c0418;text-align:left;font-size:1.8em;font-style:italic;font-weight:700;margin:0px;">
                    Profil emprunteur</h2>
                    """, 
                    unsafe_allow_html=True)
        st.write("")
        st.write(f"Type de Crédit demandé par le client : **{data_client['name_contract_type'].values[0]}**")
        st.write(f"Crédit total : **{data_client['amt_credit'].values[0]}**")
        st.write(f"Prix du bien immobilier : **{data_client['amt_goods_price'].values[0]}**")
        st.write(f"mensualité de remboursement : **{data_client['amt_annuity'].values[0]} $**")
    

    
    
###############################################################
# Comparaison du profil du client à son groupe d'appartenance #
###############################################################

# Titre 1
    st.markdown("""
                <br>
                <h1 style="color:#9c0418;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                Comparaison du profil du client à celui des clients dont la probabilité de défaut de paiement est proche</h1>
                """, 
                unsafe_allow_html=True)


# st.write("Pour la définition des groupes de clients, faites défiler la page vers le bas.")
    
# Calcul des valeurs Shap
    explainer_shap = shap.TreeExplainer(model_LGBM)
    shap_values = explainer_shap.shap_values(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))
    shap_values_df = pd.DataFrame(data=shap_values[1], columns=lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).columns)
    
    df_groupes = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'], shap_values_df], axis=1)
    df_groupes['typologie_clients'] = pd.qcut(df_groupes.proba_classe_1,
                                              q=5,
                                              precision=1,
                                              labels=['20%_et_moins',
                                                      '21%_30%',
                                                      '31%_40%',
                                                      '41%_60%',
                                                      '61%_et_plus'])

# Titre H2
    st.markdown("""
                <h2 style="color:#418b85;text-align:left;font-size:1.8em;font-style:italic;font-weight:700;margin:0px;">
                Comparaison de “la trajectoire” prise par la prédiction du client à celles des groupes de Clients</h2>
                """, 
            unsafe_allow_html=True)
    st.write("")

# Moyenne des variables par classe
    df_groupes_mean = df_groupes.groupby(['typologie_clients']).mean()
    df_groupes_mean = df_groupes_mean.rename_axis('typologie_clients').reset_index()
    df_groupes_mean["index"]=[1,2,3,4,5]
    df_groupes_mean.set_index('index', inplace = True)

# récupération de l'index correspondant à l'identifiant du client
    idx = int(lecture_X_test_clean()[lecture_X_test_clean()['sk_id_curr']==ID_client].index[0])

# dataframe avec shap values du client et des 5 groupes de clients
    comparaison_client_groupe = pd.concat([df_groupes[df_groupes.index == idx], 
                                            df_groupes_mean],
                                            axis = 0)
    comparaison_client_groupe['typologie_clients'] = np.where(comparaison_client_groupe.index == idx, 
                                                          lecture_X_test_clean().iloc[idx, 0],
                                                          comparaison_client_groupe['typologie_clients'])
# transformation en array
    nmp = comparaison_client_groupe.drop(labels=['typologie_clients', "proba_classe_1"], axis=1).to_numpy()

    fig = plt.figure(figsize=(8, 20))
    st_shap(shap.decision_plot(explainer_shap.expected_value[1], 
                            nmp, 
                            feature_names=comparaison_client_groupe.drop(
                                labels=['typologie_clients', "proba_classe_1"], axis=1).columns.to_list(),
                            feature_order='importance',
                            highlight=0,
                            legend_labels=['Client', '20%_et_moins', '21%_30%', '31%_40%', '41%_60%', '61%_et_plus'],
                            plot_color='inferno_r',
                            legend_location='center right',
                            feature_display_range=slice(None, -25, -1),
                            link='logit')) 

    
    liste_variables = lecture_X_test_clean().drop(
    labels="sk_id_curr", axis=1).columns.to_list()

# division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
    col1, col2, = st.columns(2)
    with col1:
        ID_var = st.selectbox("*Veuillez sélectionner une variable à l'aide du menu déroulant 👇*",
                          (liste_variables))
        st.write("Vous avez sélectionné la variable :", ID_var)

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121)
    shap.dependence_plot(ID_var,
                     calcul_valeurs_shap()[1],
                     lecture_X_test_clean().drop(labels="sk_id_curr", axis=1),
                     interaction_index=None,
                     alpha=0.5,
                     x_jitter=0.5,
                     title="Graphique de Dépendance",
                     ax=ax1,
                     show=False)
    ax2 = fig.add_subplot(122)
    shap.dependence_plot(ID_var,
                     calcul_valeurs_shap()[1],
                     lecture_X_test_clean().drop(labels="sk_id_curr", axis=1),
                     interaction_index='auto',
                     alpha=0.5,
                     x_jitter=0.5,
                     title="Graphique de Dépendance et Intéraction",
                     ax=ax2,
                     show=False)
    fig.tight_layout()
    st.pyplot(fig)
    