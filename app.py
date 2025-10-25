# app.py
import os
import json
import random
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Flatten, Conv1D, Conv2D,
                                     MaxPooling1D, Activation, LSTM, Concatenate)
from tensorflow.keras.optimizers import Adam

# =========================
# CONFIG
# =========================
LOOKBACK = 50
MODEL_TYPE = "EIIE"
STOCKS = ['ETHUSDT', 'BTCUSDT', 'BNBUSDT', 'XRPUSDT']  # keep 4 to match trained weights
MODEL_DIR = r"C:\Users\joshu\dl\2025_09_13_20_29_Crypto_trader"
WEIGHT_PREFIX = "final_Crypto_trader"  # files like final_Crypto_trader_Actor.weights.h5

# =========================
# Reproducibility
# =========================
random.seed(2002)
np.random.seed(32)
tf.random.set_seed(100)

# EIIE code below expects channels_last (H, W, C)
K.set_image_data_format("channels_last")

# =========================
# Shared Model (EIIE ready)
# =========================
class Shared_Model:
    def __init__(self, input_shape, action_space, lr, optimizer, model="Dense"):
        X_input = Input(input_shape)  # EIIE expects (assets, lookback, features) with channels_last
        self.action_space = action_space
        self.model = model

        if model == "CNN":
            X = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)
            X = MaxPooling1D(pool_size=2)(X)
            X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X)
            X = MaxPooling1D(pool_size=2)(X)
            X = Flatten()(X)

        elif model == "EIIE":
            # Main branch on market tensor
            # Assumes X_input shape: (assets, lookback, features)
            # Use small kernels to avoid negative dims; we just feature-mix
            X = Conv2D(2, (1, 1), padding="valid", activation='relu')(X_input)
            X = Conv2D(20, (1, 1), padding="valid", activation='relu')(X)

            # Orders/history side branch (match what training expected: (1, 50, 10))
            inputB = Input(shape=(1, LOOKBACK, 10))
            modelB = Conv2D(filters=2, kernel_size=(1, 1), activation='relu')(inputB)
            modelB = Conv2D(filters=20, kernel_size=(1, 1), activation='relu')(modelB)

            # Flatten & merge
            X = Flatten()(X)
            modelB = Flatten()(modelB)
            merged = Concatenate(axis=1)([X, modelB])
            X = Dense(256, activation="relu")(merged)

        elif model == "LSTM":
            X = LSTM(512, return_sequences=True)(X_input)
            X = LSTM(256)(X)

        else:
            X = Flatten()(X_input)
            X = Dense(512, activation="relu")(X)

        # Critic
        V = Dense(512, activation="relu")(X)
        V = Dense(256, activation="relu")(V)
        V = Dense(64, activation="relu")(V)
        value = Dense(1, activation=None)(V)
        if model == "EIIE":
            self.Critic = Model(inputs=[X_input, inputB], outputs=value)
        else:
            self.Critic = Model(inputs=X_input, outputs=value)
        self.Critic.compile(loss=self.critic_PPO2_loss,
                            optimizer=optimizer(learning_rate=lr))

        # Actor
        A = Dense(512, activation="relu")(X)
        A = Dense(256, activation="relu")(A)
        A = Dense(64, activation="relu")(A)
        output = Dense(self.action_space, activation="softmax")(A)
        if model == "EIIE":
            self.Actor = Model(inputs=[X_input, inputB], outputs=output)
        else:
            self.Actor = Model(inputs=X_input, outputs=output)
        self.Actor.compile(loss=self.ppo_loss,
                           optimizer=optimizer(learning_rate=lr))

    # PPO Actor loss (clipped)
    def ppo_loss(self, y_true, y_pred):
        advantages, prediction_picks = y_true[:, :1], y_true[:, 1:1+self.action_space]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001

        prob = K.clip(y_pred, 1e-10, 1.0)
        old_prob = K.clip(prediction_picks, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))
        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        return actor_loss - entropy

    def critic_PPO2_loss(self, y_true, y_pred):
        return K.mean((y_true - y_pred) ** 2)

    # Predict helpers
    def actor_predict(self, state, order):
        if self.model == "EIIE":
            return self.Actor.predict([state, order], verbose=0)
        else:
            return self.Actor.predict(state, verbose=0)

    def critic_predict(self, state, order):
        if self.model == "EIIE":
            return self.Critic.predict([state, order], verbose=0)
        else:
            return self.Critic.predict(state, verbose=0)

# =========================
# Agent
# =========================
class CustomAgent:
    def __init__(self, lookback_window_size=50, lr=0.00005, epochs=1, stocks=None,
                 optimizer=Adam, batch_size=32, model="", shape=None, depth=0, comment=""):
        self.lookback_window_size = lookback_window_size
        self.model = model
        self.comment = comment
        self.depth = depth
        self.stocks = stocks or []
        self.shape = shape or ()
        self.action_space = np.arange(len(self.stocks))
        self.log_name = datetime.now().strftime("%Y_%m_%d_%H_%M") + "_Crypto_trader"

        if self.model == "EIIE":
            # input to model: (assets, lookback, features)
            self.state_size = (len(self.stocks), lookback_window_size, self.shape[1])
        else:
            self.state_size = (lookback_window_size, self.shape[1]*self.shape[2] + 2 + 2*self.shape[2])

        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.Actor = self.Critic = Shared_Model(
            input_shape=self.state_size,
            action_space=self.action_space.shape[0],
            lr=self.lr,
            optimizer=self.optimizer,
            model=self.model
        )

    def act(self, state, order):
        pred = self.Actor.actor_predict(
            np.expand_dims(state, axis=0),
            np.expand_dims(np.expand_dims(order, axis=0), axis=0)
        )[0]
        return pred

    def load(self, folder, name):
        actor_path = os.path.join(folder, f"{name}_Actor.weights.h5")
        critic_path = os.path.join(folder, f"{name}_Critic.weights.h5")
        # Safe load, tolerate minor mismatches if any
        self.Actor.Actor.load_weights(actor_path, by_name=True, skip_mismatch=True)
        self.Critic.Critic.load_weights(critic_path, by_name=True, skip_mismatch=True)
        print(f"Loaded Actor from {actor_path}")
        print(f"Loaded Critic from {critic_path}")

# =========================
# Environment
# =========================
class CustomEnv:
    """A custom multi-asset trading environment matching your training format."""
    def __init__(self, df, df_normalized, initial_balance=1000, stocks=None,
                 lookback_window_size=50, model=''):
        self.stocks = stocks or []
        self.xarray = df_normalized       # shape: (time, features, assets)
        self.df = df                      # shape: (time, features, assets)
        self.df_total_steps = self.xarray.shape[0]
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.normalize_value = 40000
        self.model = model

        self.n_assets = self.df.shape[2]
        self.weights = [1] + [0]*(self.n_assets - 1)
        self.quants = [0]*self.n_assets
        self.quants_ubah = [0]*self.n_assets
        self.cash = 0
        self.market_state = {str(i): deque(maxlen=self.lookback_window_size) for i in range(self.n_assets)}

        self.orders_history = deque(maxlen=self.lookback_window_size)
        self.market_history = deque(maxlen=self.lookback_window_size)

    def reset(self, env_steps_size=0):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.weights = [1] + [0]*(self.n_assets - 1)
        self.quants = [0]*self.n_assets
        self.quants_ubah = [0]*self.n_assets
        self.cash = self.initial_balance

        if env_steps_size > 0:
            self.start_step = random.randint(self.lookback_window_size,
                                             self.df_total_steps - env_steps_size - 1)
            self.end_step = self.start_step + env_steps_size
        else:
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps - 1

        self.current_step = self.start_step

        # Buy & hold quantities based on "Close" feature index 2
        prices_now = np.array([self.df[self.current_step, 2, a] for a in range(self.n_assets)])
        self.quants_ubah = [(self.initial_balance/self.n_assets) / prices_now]

        # Fill initial orders (net_worth, cash, quants..., weights...)
        for i in reversed(range(self.lookback_window_size)):
            self.orders_history.append(
                [self.net_worth/self.normalize_value,
                 self.cash/self.normalize_value] +
                [q for q in self.quants] +
                [w for w in self.weights]
            )

        # Fill market window
        for a in range(self.n_assets):
            self.market_state[str(a)].clear()
            for i in reversed(range(self.lookback_window_size)):
                self.market_state[str(a)].append(self.xarray[self.current_step - i, :, a])

        if self.model == "EIIE":
            # shape: (assets, lookback, features)
            state = np.stack([self.market_state[str(a)] for a in range(self.n_assets)], axis=0)
        else:
            state = np.concatenate([self.market_state[str(a)] for a in range(self.n_assets)], axis=1)
            state = np.concatenate((state, self.orders_history), axis=1)

        return state, self.orders_history

    def _next_observation(self):
        for a in range(self.n_assets):
            self.market_state[str(a)].append(self.xarray[self.current_step, :, a])

        if self.model == "EIIE":
            obs = np.stack([self.market_state[str(a)] for a in range(self.n_assets)], axis=0)
        else:
            obs = np.concatenate([self.market_state[str(a)] for a in range(self.n_assets)], axis=1)
            obs = np.concatenate((obs, self.orders_history), axis=1)
        return obs

    def step(self, prediction):
        # previous and new prices (use Close = index 2)
        prices_old = np.array([self.df[self.current_step, 2, a] for a in range(self.n_assets)])
        self.current_step += 1
        prices_new = np.array([self.df[self.current_step, 2, a] for a in range(self.n_assets)])

        # balance using old quantities at new prices
        self.balance = self.cash + np.dot(prices_new[1:], self.quants[1:])
        quants_old = np.array(self.quants, dtype=float)

        # rebalance by predicted weights
        self.quants = [self.balance * prediction[a] / prices_new[a] for a in range(self.n_assets)]

        # transaction fee ~0.1%
        tax = np.sum(np.abs(np.dot(self.quants, prices_new) - np.dot(quants_old, prices_old))) * 0.001

        self.cash = self.quants[0] * prices_new[0]
        self.prev_net_worth = self.net_worth
        self.net_worth = float(np.dot(self.quants, prices_new) - tax)

        self.orders_history.append(
            [self.net_worth/self.normalize_value,
             self.cash/self.normalize_value] +
            [q/self.normalize_value for q in self.quants] +
            prediction.tolist()
        )

        reward = float(np.log(self.net_worth / self.prev_net_worth)) if self.prev_net_worth > 0 else 0.0
        done = self.net_worth <= self.initial_balance/2 or self.current_step >= self.end_step
        obs = self._next_observation()
        return obs, self.orders_history, reward, done, prices_new

    def render(self):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}")

# =========================
# Helpers: build 3D arrays from uploads
# =========================
def ensure_3d_assets_from_csvs(files, expected_assets=4):
    """
    Accept 1..expected_assets CSVs and return (time, features, assets).
    If only 1 CSV is provided, we tile it across assets to match model outputs.
    """
    dfs = [pd.read_csv(f) for f in files]
    # Convert to numpy 2D: (time, features)
    mats = [d.values for d in dfs]
    # Align lengths by min time
    min_len = min(m.shape[0] for m in mats)
    mats = [m[:min_len] for m in mats]
    # Feature alignment: assume same columns/order as training (Open, Close, Low, High, Volume, ...)
    # Stack as assets on last axis
    stacked = np.stack(mats, axis=-1)  # (time, features, n_assets)
    # If fewer than expected_assets, tile last axis
    n_assets = stacked.shape[2]
    if n_assets == 1 and expected_assets > 1:
        stacked = np.repeat(stacked, expected_assets, axis=2)
    elif n_assets < expected_assets:
        # Repeat proportionally to reach expected_assets
        reps = int(np.ceil(expected_assets / n_assets))
        stacked = np.concatenate([stacked]*reps, axis=2)[:, :, :expected_assets]
    elif n_assets > expected_assets:
        stacked = stacked[:, :, :expected_assets]
    return stacked

# =========================
# STREAMLIT UI
# =========================
st.title("📈 Crypto Trading RL Agent Demo")
st.markdown("Upload **1–4 CSVs** (ETH/BTC/BNB/XRP). If you upload only one, we'll tile it to 4 assets to match the trained model. "
            "The app will load your pre-trained **EIIE Actor–Critic** and simulate a short run.")

uploaded_files = st.file_uploader(
    "Upload 1–4 normalized CSVs (same columns as training: Open, Close, Low, High, Volume, ...)",
    type="csv",
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 4:
        st.warning("Please upload at most 4 files. Using the first 4.")
        uploaded_files = uploaded_files[:4]

    # Build raw & normalized (we'll just use the same for both, like your training loader)
    x_all = ensure_3d_assets_from_csvs(uploaded_files, expected_assets=4)  # (T, F, 4)
    st.write("Shape (time, features, assets):", x_all.shape)

    # Quick preview
    uploaded_files[0].seek(0)  # reset buffer pointer
    preview_df = pd.read_csv(uploaded_files[0])
    st.write("### First file preview")
    st.dataframe(preview_df.head())

    # Create Env + Agent
    test_env = CustomEnv(
        df=x_all, df_normalized=x_all,
        stocks=STOCKS,
        lookback_window_size=LOOKBACK,
        model=MODEL_TYPE
    )

    agent = CustomAgent(
        lookback_window_size=LOOKBACK,
        lr=0.00001,
        epochs=1,
        stocks=STOCKS,
        optimizer=Adam,
        batch_size=32,
        model=MODEL_TYPE,
        shape=x_all.shape  # (T, F, A) -> features = shape[1]
    )

    # Load weights
    agent.load(folder=MODEL_DIR, name=WEIGHT_PREFIX)

    # Simulate one episode
    steps = min(200, x_all.shape[0] - LOOKBACK - 2)
    if steps <= 0:
        st.error("Not enough rows for the chosen LOOKBACK window. Please upload a longer CSV.")
        st.stop()

    state, order = test_env.reset(env_steps_size=steps)
    net_worths, ubah_values = [], []

    for _ in range(steps):
        action = agent.act(state, np.array(order))
        state, order, reward, done, prices = test_env.step(action)
        net_worths.append(test_env.net_worth)
        ubah_values.append(float(np.dot(test_env.quants_ubah, prices)))
        if done:
            break

    # Results
    st.subheader("Results")
    st.write(f"**Final RL Net Worth:** {net_worths[-1]:.2f}")
    st.write(f"**Final Buy & Hold:** {ubah_values[-1]:.2f}")

    fig, ax = plt.subplots()
    ax.plot(net_worths, label="RL Agent")
    ax.plot(ubah_values, label="Buy & Hold")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Upload 1–4 CSV files to begin.")
