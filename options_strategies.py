import streamlit as st  # library to display website
import pandas as pd  # I'll use it to recap the different options that are inputs 
import numpy as np  # vectorisation for quicker computation
import matplotlib.pyplot as plt   # plotting the P&L graphs
import math as m  # maths functions
from scipy.stats import norm    # N(d1) and N(d2) in BS princing

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

liste_spot = np.arange(1, 1001, 1)  # 1000 éléments


def price_call(S, K, sigma, T, r):
    d1 = (m.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * m.sqrt(T))
    d2 = d1 - sigma * m.sqrt(T)
    return S * norm.cdf(d1) - K * m.exp(-r * T) * norm.cdf(d2)


def price_put(S, K, sigma, T, r):
    return price_call(S, K, sigma, T, r) - S + K * m.exp(-r * T)  # Parité put-call


def put_function(K_put):
    liste_OTM = np.zeros(1001 - K_put)  # Hors de la monnaie
    liste_ITM = K_put - np.arange(0, K_put)  # Dans la monnaie
    PandL = np.concatenate((liste_ITM, liste_OTM))[:1000] - price_put(S, K_put, sigma, T, r)
    return PandL


def call_function(K_call):
    liste_OTM = np.zeros(K_call + 1)  
    liste_ITM = np.arange(K_call + 1, 1001) - K_call  
    PandL = np.concatenate((liste_OTM, liste_ITM))[:1000] - price_call(S, K_call, sigma, T, r)
    return PandL


if 'liste_options' not in st.session_state:
    st.session_state.liste_options = np.zeros(1000)

if 'data' not in st.session_state:
    st.session_state.data = []


selected_strike = st.selectbox('Pick the strike AND THEN click on the position you want with the latter ', range(1, 1001))
st.write('(You can enter the strike using your keyboard) ')

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


if st.button('RESET'):
    st.session_state.liste_options = np.zeros(1000)  # Remettre à zéro
    st.session_state.data = []  # Vider la liste des options


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










    

