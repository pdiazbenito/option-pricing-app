
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy.stats import norm
from enum import Enum
from abc import ABC, abstractmethod
import yfinance as yf

# ── CLASES BASE ──────────────────────────────────────────
class OPTION_TYPE(Enum):
    CALL_OPTION = "Call Option"
    PUT_OPTION  = "Put Option"

class OptionPricingModel(ABC):
    def calculate_option_price(self, option_type):
        if option_type == OPTION_TYPE.CALL_OPTION.value:
            return self._calculate_call_option_price()
        elif option_type == OPTION_TYPE.PUT_OPTION.value:
            return self._calculate_put_option_price()
        return -1
    @abstractmethod
    def _calculate_call_option_price(self): pass
    @abstractmethod
    def _calculate_put_option_price(self): pass

# ── BLACK-SCHOLES ─────────────────────────────────────────
class BlackScholesModel(OptionPricingModel):
    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma):
        self.S = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma
    def _calculate_call_option_price(self):
        d1 = (np.log(self.S/self.K) + (self.r + 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
        d2 = (np.log(self.S/self.K) + (self.r - 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
        return self.S*norm.cdf(d1) - self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
    def _calculate_put_option_price(self):
        d1 = (np.log(self.S/self.K) + (self.r + 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
        d2 = (np.log(self.S/self.K) + (self.r - 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
        return self.K*np.exp(-self.r*self.T)*norm.cdf(-d2) - self.S*norm.cdf(-d1)

# ── MONTE CARLO ───────────────────────────────────────────
class MonteCarloPricing(OptionPricingModel):
    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_simulations):
        self.S_0 = underlying_spot_price
        self.K   = strike_price
        self.T   = days_to_maturity / 365
        self.r   = risk_free_rate
        self.sigma = sigma
        self.N   = number_of_simulations
        self.num_of_steps = days_to_maturity
        self.dt  = self.T / self.num_of_steps
        self.simulation_results_S = None
    def simulate_prices(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        S = np.zeros((self.num_of_steps, self.N))
        S[0] = self.S_0
        for t in range(1, self.num_of_steps):
            Z = np.random.standard_normal(self.N)
            S[t] = S[t-1] * np.exp((self.r - 0.5*self.sigma**2)*self.dt + self.sigma*np.sqrt(self.dt)*Z)
        self.simulation_results_S = S
    def _calculate_call_option_price(self):
        if self.simulation_results_S is None: return -1
        return np.exp(-self.r*self.T) * (1/self.N) * np.sum(np.maximum(self.simulation_results_S[-1] - self.K, 0))
    def _calculate_put_option_price(self):
        if self.simulation_results_S is None: return -1
        return np.exp(-self.r*self.T) * (1/self.N) * np.sum(np.maximum(self.K - self.simulation_results_S[-1], 0))

# ── BINOMIAL TREE ─────────────────────────────────────────
class BinomialTreeModel(OptionPricingModel):
    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_time_steps):
        self.S = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma
        self.number_of_time_steps = number_of_time_steps
    def _calculate_call_option_price(self):
        dT = self.T / self.number_of_time_steps
        u  = np.exp(self.sigma * np.sqrt(dT))
        d  = 1.0 / u
        S_T = np.array([(self.S * u**j * d**(self.number_of_time_steps-j)) for j in range(self.number_of_time_steps+1)])
        a = np.exp(self.r * dT); p = (a-d)/(u-d); q = 1.0-p
        V = np.maximum(S_T - self.K, 0.0)
        for i in range(self.number_of_time_steps-1, -1, -1):
            V[:-1] = np.exp(-self.r*dT) * (p*V[1:] + q*V[:-1])
        return V[0]
    def _calculate_put_option_price(self):
        dT = self.T / self.number_of_time_steps
        u  = np.exp(self.sigma * np.sqrt(dT))
        d  = 1.0 / u
        S_T = np.array([(self.S * u**j * d**(self.number_of_time_steps-j)) for j in range(self.number_of_time_steps+1)])
        a = np.exp(self.r * dT); p = (a-d)/(u-d); q = 1.0-p
        V = np.maximum(self.K - S_T, 0.0)
        for i in range(self.number_of_time_steps-1, -1, -1):
            V[:-1] = np.exp(-self.r*dT) * (p*V[1:] + q*V[:-1])
        return V[0]

# ── TICKER ────────────────────────────────────────────────
class Ticker:
    @staticmethod
    def get_historical_data(ticker, start_date=None, end_date=None):
        if start_date is None:
            start_date = datetime.datetime.now() - datetime.timedelta(days=365)
        if end_date is None:
            end_date = datetime.datetime.now()
        stock = yf.Ticker(ticker)
        data  = stock.history(start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data for {ticker}")
        return data
    @staticmethod
    def get_last_price(data, column_name):
        return data[column_name].iloc[-1]

# ── INTERFAZ STREAMLIT ────────────────────────────────────
st.set_page_config(page_title="Option Pricing", page_icon="📈", layout="wide")
st.title("📈 Option Pricing Models")

st.sidebar.header("⚙️ Parámetros")
ticker_symbol   = st.sidebar.text_input("Ticker", value="AAPL")
strike_price    = st.sidebar.number_input("Strike Price (K)", value=200.0)
risk_free_rate  = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
sigma           = st.sidebar.slider("Volatilidad (%)", 1.0, 100.0, 25.0) / 100
exercise_date   = st.sidebar.date_input("Vencimiento", value=datetime.date.today() + datetime.timedelta(days=90))
model_type      = st.sidebar.selectbox("Modelo", ["Black-Scholes", "Monte Carlo", "Binomial Tree"])
num_simulations = st.sidebar.number_input("Simulaciones (MC)", value=10000, step=1000)
num_time_steps  = st.sidebar.number_input("Pasos (Binomial)", value=100, step=10)

days_to_maturity = (exercise_date - datetime.date.today()).days

if st.sidebar.button("🚀 Calcular"):
    if days_to_maturity <= 0:
        st.error("La fecha de vencimiento debe ser futura.")
    else:
        try:
            data        = Ticker.get_historical_data(ticker_symbol)
            spot_price  = Ticker.get_last_price(data, "Close")
            st.subheader(f"{ticker_symbol} — Precio actual: ${spot_price:.2f} | Días a vencimiento: {days_to_maturity}")

            fig, ax = plt.subplots(figsize=(10, 3))
            data["Close"].plot(ax=ax, color="steelblue")
            ax.set_title(f"Histórico {ticker_symbol}"); ax.set_ylabel("Precio ($)")
            st.pyplot(fig)

            if model_type == "Black-Scholes":
                model = BlackScholesModel(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma)
            elif model_type == "Monte Carlo":
                model = MonteCarloPricing(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, int(num_simulations))
                model.simulate_prices()
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.plot(model.simulation_results_S[:, :200], alpha=0.3, linewidth=0.5)
                ax2.axhline(strike_price, c="red", linewidth=2, label="Strike Price")
                ax2.set_title("Simulaciones Monte Carlo"); ax2.legend(); st.pyplot(fig2)
            elif model_type == "Binomial Tree":
                model = BinomialTreeModel(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, int(num_time_steps))

            call = model.calculate_option_price(OPTION_TYPE.CALL_OPTION.value)
            put  = model.calculate_option_price(OPTION_TYPE.PUT_OPTION.value)

            col1, col2 = st.columns(2)
            col1.metric("📗 Call Option", f"${call:.4f}")
            col2.metric("📕 Put Option",  f"${put:.4f}")
        except Exception as e:
            st.error(f"Error: {e}")
