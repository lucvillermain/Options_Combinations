import streamlit as st  # library to display website
import pandas as pd  # I'll use it to recap the different options that are inputs 
import numpy as np  # vectorisation for quicker computation
import matplotlib.pyplot as plt   # plotting the P&L graphs
import math as m  # maths functions


def norm_cdf_math(x):
    return 0.5 * (1 + m.erf(x / m.sqrt(2)))

st.set_page_config(layout="wide")

st.title('Options Trading using Vanilla Options')

with st.expander('About this app'):
  st.write('This app enbables to plot the P&L at expiry of the combination of call and put options with different strikes, the (European) options premium are computed using the Black-Scholes pricing model. You can input the parameters in the left sidebar ')

#---------- Side bar -----------------------
st.sidebar.markdown(
    """
    <div style='text-align: center; font-size: 16px;'>
        Created by <span style='color: #39FF14;'> Luc Villermain</span>
    </div>
    """, unsafe_allow_html=True
)

st.sidebar.header('Input Parameters')



S = st.sidebar.selectbox('Spot price (S)', range(1, 1001))
r = st.sidebar.selectbox('Risk-free Interest Rate %(r)', [round(x * 0.25, 2) for x in range(0, 41)])
T = st.sidebar.selectbox('Time to expiry in days (T)', range(1, 366)) / 365  # Convertir en années
sigma = st.sidebar.selectbox('Volatility (%)', [round(x * 0.25, 2) for x in range(4, 121)])

sigma = sigma/100
r = r/100

liste_spot = np.arange(1, 1001, 1)  # 1000 éléments

# Fonction pour calculer le prix d'un Call
def price_call(S, K, sigma, T, r):
    d1 = (m.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * m.sqrt(T))
    d2 = d1 - sigma * m.sqrt(T)
    return S * norm_cdf_math(d1) - K * m.exp(-r * T) * norm_cdf_math(d2)

# Fonction pour calculer le prix d'un Put
def price_put(S, K, sigma, T, r):
    return price_call(S, K, sigma, T, r) - S + K * m.exp(-r * T)  # Parité put-call



# Fonction pour générer le P&L d'un Put (1000 éléments)
def put_function(K_put):
    liste_OTM = np.array([0 for i in np.arange(K_put,1001,1)])
    liste_ITM = np.array([K_put-i for i in np.arange(1,K_put,1)])  
    PandL = np.concatenate((liste_ITM, liste_OTM)) - price_put(S, K_put, sigma, T, r)
    return PandL

# Fonction pour générer le P&L d'un Call (1000 éléments)
def call_function(K_call):
    liste_OTM = np.array([0 for i in range(1,K_call+1)])
    liste_ITM = np.array([i-K_call for i in np.arange(K_call+1,1001,1)])
    PandL = np.concatenate((liste_OTM, liste_ITM)) - price_call(S, K_call, sigma, T, r)
    return PandL





# Initialisation de session_state
if 'liste_options' not in st.session_state:
    st.session_state.liste_options = np.zeros(1000)

if 'data' not in st.session_state:
    st.session_state.data = []

# Étape 1 : Sélectionner le strike
selected_strike = st.selectbox('Pick the strike AND THEN click on the position you want with the latter ', range(1, 1001))
st.write('(You can enter the strike using your keyboard) ')
# Étape 2 : Boutons pour ajouter des options après sélection du strike
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button('Add Long Call'):
        st.session_state.liste_options += call_function(selected_strike)
        st.session_state.data.append(["Long Call", selected_strike])

with col2:
    if st.button('Add Short Call'):
        st.session_state.liste_options -= call_function(selected_strike)
        st.session_state.data.append(["Short Call", selected_strike])

with col3:
    if st.button('Add Long Put'):
        st.session_state.liste_options += put_function(selected_strike)
        st.session_state.data.append(["Long Put", selected_strike])

with col4:
    if st.button('Add Short Put'):
        st.session_state.liste_options -= put_function(selected_strike)
        st.session_state.data.append(["Short Put", selected_strike])

# Bouton RESET pour réinitialiser toutes les options
if st.button('RESET'):
    st.session_state.liste_options = np.zeros(1000)  # Remettre à zéro
    st.session_state.data = []  # Vider la liste des options

# Affichage de la table récapitulative des options ajoutées
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data, columns=["Option", "Strike"])
    st.write("Table Recap of positions:")
    st.dataframe(df)


st.header("P&L Graph of Combined Options")

X = liste_spot
Y = st.session_state.liste_options

# Création du graphique
fig, ax = plt.subplots()
ax.plot(X, Y, color='black')

# Fill the area above and below zero with different colors
ax.fill_between(X, Y, where=(Y > 0), color='green', alpha=0.6, interpolate=True)
ax.fill_between(X, Y, where=(Y < 0), color='red', alpha=0.6, interpolate=True)

# Add horizontal line at y=0
ax.axhline(0, color='black', linewidth=1)

# Afficher le graphique dans Streamlit
st.pyplot(fig)













    

