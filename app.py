import os
import json
import random
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Flatten, Conv1D, Conv2D,
                                     MaxPooling1D, Activation, LSTM, Concatenate)
from tensorflow.keras.optimizers import Adam

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Crypto Trading RL Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# CONFIG
# =========================
LOOKBACK = 50
MODEL_TYPE = "EIIE"
STOCKS = ['ETHUSDT', 'BTCUSDT', 'BNBUSDT', 'XRPUSDT']
MODEL_DIR = r"C:\Users\joshu\dl\2025_09_13_20_29_Crypto_trader"
WEIGHT_PREFIX = "final_Crypto_trader"

# =========================
# Reproducibility
# =========================
random.seed(2002)
np.random.seed(32)
tf.random.set_seed(100)
K.set_image_data_format("channels_last")

# =========================
# MODEL CLASSES (unchanged)
# =========================
class Shared_Model:
    def __init__(self, input_shape, action_space, lr, optimizer, model="Dense"):
        X_input = Input(input_shape)
        self.action_space = action_space
        self.model = model

        if model == "CNN":
            X = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)
            X = MaxPooling1D(pool_size=2)(X)
            X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X)
            X = MaxPooling1D(pool_size=2)(X)
            X = Flatten()(X)
        elif model == "EIIE":
            # Original training used (1, 2) or (2, 1) kernel - adjust based on error
            X = Conv2D(2, (1, 2), padding="same", activation='relu')(X_input)
            X = Conv2D(20, (1, 2), padding="same", activation='relu')(X)

            inputB = Input(shape=(1, LOOKBACK, 10))
            modelB = Conv2D(filters=2, kernel_size=(1, 2), padding='same', activation='relu')(inputB)
            modelB = Conv2D(filters=20, kernel_size=(1, 2), padding='same', activation='relu')(modelB)

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

        V = Dense(512, activation="relu")(X)
        V = Dense(256, activation="relu")(V)
        V = Dense(64, activation="relu")(V)
        value = Dense(1, activation=None)(V)
        if model == "EIIE":
            self.Critic = Model(inputs=[X_input, inputB], outputs=value)
        else:
            self.Critic = Model(inputs=X_input, outputs=value)
        self.Critic.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(learning_rate=lr))

        A = Dense(512, activation="relu")(X)
        A = Dense(256, activation="relu")(A)
        A = Dense(64, activation="relu")(A)
        output = Dense(self.action_space, activation="softmax")(A)
        if model == "EIIE":
            self.Actor = Model(inputs=[X_input, inputB], outputs=output)
        else:
            self.Actor = Model(inputs=X_input, outputs=output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=lr))

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

    def actor_predict(self, state, order):
        if self.model == "EIIE":
            return self.Actor.predict([state, order], verbose=0)
        else:
            return self.Actor.predict(state, verbose=0)

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
        
        if not os.path.exists(actor_path):
            st.error(f"❌ Actor weights not found at: {actor_path}")
            return False
        if not os.path.exists(critic_path):
            st.error(f"❌ Critic weights not found at: {critic_path}")
            return False
            
        try:
            # Try loading with skip_mismatch to handle shape differences
            self.Actor.Actor.load_weights(actor_path, skip_mismatch=True)
            self.Critic.Critic.load_weights(critic_path, skip_mismatch=True)
            st.success("✅ Weights loaded successfully (some layers may have been skipped due to shape mismatch)")
            return True
        except Exception as e:
            st.error(f"⚠️ Error loading weights: {e}")
            st.warning("""
            **Troubleshooting Tips:**
            - Ensure the model was trained with the same TensorFlow version
            - Check that LOOKBACK and STOCKS match the training configuration
            - Verify the kernel sizes in Conv2D layers match training
            - Try retraining the model or adjusting the architecture
            """)
            return False

class CustomEnv:
    def __init__(self, df, df_normalized, initial_balance=1000, stocks=None,
                 lookback_window_size=50, model=''):
        self.stocks = stocks or []
        self.xarray = df_normalized
        self.df = df
        self.df_total_steps = self.xarray.shape[0]
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.normalize_value = 40000
        self.model = model
        self.n_assets = self.df.shape[2]
        self.weights = [1] + [0]*(self.n_assets - 1)
        self.quants = [0]*self.n_assets
        self.cash = 0
        self.market_state = {str(i): deque(maxlen=self.lookback_window_size) for i in range(self.n_assets)}
        self.orders_history = deque(maxlen=self.lookback_window_size)

    def reset(self, env_steps_size=0):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.weights = [1] + [0]*(self.n_assets - 1)
        self.quants = [0]*self.n_assets
        self.cash = self.initial_balance

        if env_steps_size > 0:
            self.start_step = random.randint(self.lookback_window_size,
                                             self.df_total_steps - env_steps_size - 1)
            self.end_step = self.start_step + env_steps_size
        else:
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps - 1

        self.current_step = self.start_step
        prices_now = np.array([self.df[self.current_step, 2, a] for a in range(self.n_assets)])
        self.quants_ubah = [(self.initial_balance/self.n_assets) / prices_now]

        for i in reversed(range(self.lookback_window_size)):
            self.orders_history.append(
                [self.net_worth/self.normalize_value, self.cash/self.normalize_value] +
                [q for q in self.quants] + [w for w in self.weights]
            )

        for a in range(self.n_assets):
            self.market_state[str(a)].clear()
            for i in reversed(range(self.lookback_window_size)):
                self.market_state[str(a)].append(self.xarray[self.current_step - i, :, a])

        state = np.stack([self.market_state[str(a)] for a in range(self.n_assets)], axis=0)
        return state, self.orders_history

    def _next_observation(self):
        for a in range(self.n_assets):
            self.market_state[str(a)].append(self.xarray[self.current_step, :, a])
        return np.stack([self.market_state[str(a)] for a in range(self.n_assets)], axis=0)

    def step(self, prediction):
        prices_old = np.array([self.df[self.current_step, 2, a] for a in range(self.n_assets)])
        self.current_step += 1
        prices_new = np.array([self.df[self.current_step, 2, a] for a in range(self.n_assets)])
        self.balance = self.cash + np.dot(prices_new[1:], self.quants[1:])
        quants_old = np.array(self.quants, dtype=float)
        self.quants = [self.balance * prediction[a] / prices_new[a] for a in range(self.n_assets)]
        tax = np.sum(np.abs(np.dot(self.quants, prices_new) - np.dot(quants_old, prices_old))) * 0.001
        self.cash = self.quants[0] * prices_new[0]
        self.prev_net_worth = self.net_worth
        self.net_worth = float(np.dot(self.quants, prices_new) - tax)
        self.orders_history.append(
            [self.net_worth/self.normalize_value, self.cash/self.normalize_value] +
            [q/self.normalize_value for q in self.quants] + prediction.tolist()
        )
        reward = float(np.log(self.net_worth / self.prev_net_worth)) if self.prev_net_worth > 0 else 0.0
        done = self.net_worth <= self.initial_balance/2 or self.current_step >= self.end_step
        obs = self._next_observation()
        return obs, self.orders_history, reward, done, prices_new

def ensure_3d_assets_from_csvs(files, expected_assets=4):
    dfs = [pd.read_csv(f) for f in files]
    mats = [d.values for d in dfs]
    min_len = min(m.shape[0] for m in mats)
    mats = [m[:min_len] for m in mats]
    stacked = np.stack(mats, axis=-1)
    n_assets = stacked.shape[2]
    if n_assets < expected_assets:
        reps = int(np.ceil(expected_assets / n_assets))
        stacked = np.concatenate([stacked]*reps, axis=2)[:, :, :expected_assets]
    elif n_assets > expected_assets:
        stacked = stacked[:, :, :expected_assets]
    return stacked

# =========================
# STREAMLIT UI
# =========================

# Header
st.markdown('<div class="main-header">🚀 Crypto Trading RL Agent</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/bitcoin.png", width=80)
    st.markdown("### ⚙️ Configuration")
    
    initial_balance = st.number_input("💰 Initial Balance ($)", 
                                      min_value=100, 
                                      max_value=100000, 
                                      value=1000, 
                                      step=100)
    
    num_steps = st.slider("📊 Number of Trading Steps", 
                          min_value=50, 
                          max_value=500, 
                          value=200, 
                          step=50)
    
    st.markdown("---")
    st.markdown("### 🔧 Advanced Settings")
    
    with st.expander("Model Architecture"):
        conv_kernel = st.selectbox(
            "Conv2D Kernel Size",
            options=["(1, 1)", "(1, 2)", "(1, 3)", "(2, 2)", "(2, 3)"],
            index=2,
            help="Adjust to match your trained model"
        )
        
        use_skip_mismatch = st.checkbox(
            "Skip mismatched layers",
            value=True,
            help="Load weights even if some layers don't match"
        )
    
    st.markdown("---")
    st.markdown("### 📋 Model Info")
    st.info(f"""
    **Model Type:** {MODEL_TYPE}  
    **Assets:** {', '.join(STOCKS)}  
    **Lookback:** {LOOKBACK} periods
    """)
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This app demonstrates a **Reinforcement Learning agent** 
    trained using PPO (Proximal Policy Optimization) for 
    cryptocurrency portfolio management.
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📈 Trading Simulation", "📊 Data Preview", "ℹ️ How It Works"])

with tab1:
    uploaded_files = st.file_uploader(
        "Upload 1-4 Normalized CSV Files (Open, Close, Low, High, Volume, ...)",
        type="csv",
        accept_multiple_files=True,
        help="Upload market data files for each cryptocurrency"
    )

    if uploaded_files:
        with st.spinner("🔄 Processing data..."):
            x_all = ensure_3d_assets_from_csvs(uploaded_files, expected_assets=4)
        
        st.success(f"✅ Loaded data with shape: {x_all.shape} (time, features, assets)")
        
        if st.button("🚀 Run Trading Simulation", type="primary", use_container_width=True):
            
            # Show model info
            with st.expander("🔍 Model Diagnostics", expanded=False):
                st.write(f"**Expected input shape:** {x_all.shape}")
                st.write(f"**Lookback window:** {LOOKBACK}")
                st.write(f"**Number of assets:** {len(STOCKS)}")
                st.write(f"**Model directory:** {MODEL_DIR}")
                st.write(f"**Weight prefix:** {WEIGHT_PREFIX}")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize environment and agent
            test_env = CustomEnv(
                df=x_all, 
                df_normalized=x_all,
                initial_balance=initial_balance,
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
                shape=x_all.shape
            )

            status_text.text("📦 Loading model weights...")
            weights_loaded = agent.load(folder=MODEL_DIR, name=WEIGHT_PREFIX)
            
            if not weights_loaded:
                demo_mode = st.warning("⚠️ Could not load weights. Would you like to run in DEMO MODE with random actions?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Yes, Demo Mode"):
                        weights_loaded = "demo"
                with col2:
                    if st.button("❌ Cancel"):
                        st.stop()
                
                if weights_loaded != "demo":
                    st.stop()
            
            if weights_loaded:
                steps = min(num_steps, x_all.shape[0] - LOOKBACK - 2)
                state, order = test_env.reset(env_steps_size=steps)
                
                net_worths, ubah_values = [], []
                allocations_history = []
                
                status_text.text("🤖 Running agent...")
                
                for step in range(steps):
                    action = agent.act(state, np.array(order))
                    state, order, reward, done, prices = test_env.step(action)
                    net_worths.append(test_env.net_worth)
                    ubah_values.append(float(np.dot(test_env.quants_ubah, prices)))
                    allocations_history.append(action.tolist())
                    
                    progress_bar.progress((step + 1) / steps)
                    
                    if done:
                        break
                
                status_text.empty()
                progress_bar.empty()
                
                # Results
                st.markdown("## 📊 Simulation Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "🤖 Final RL Portfolio",
                        f"${net_worths[-1]:,.2f}",
                        f"{((net_worths[-1]/initial_balance - 1) * 100):+.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "🏪 Final Buy & Hold",
                        f"${ubah_values[-1]:,.2f}",
                        f"{((ubah_values[-1]/initial_balance - 1) * 100):+.2f}%"
                    )
                
                with col3:
                    outperformance = net_worths[-1] - ubah_values[-1]
                    st.metric(
                        "📈 Outperformance",
                        f"${outperformance:,.2f}",
                        f"{(outperformance/ubah_values[-1] * 100):+.2f}%"
                    )
                
                with col4:
                    sharpe = np.mean(np.diff(net_worths)) / (np.std(np.diff(net_worths)) + 1e-6) * np.sqrt(252)
                    st.metric(
                        "📉 Sharpe Ratio",
                        f"{sharpe:.2f}"
                    )
                
                # Portfolio value chart
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Portfolio Value Over Time', 'Asset Allocation Over Time'),
                    vertical_spacing=0.12,
                    row_heights=[0.6, 0.4]
                )
                
                fig.add_trace(
                    go.Scatter(x=list(range(len(net_worths))), y=net_worths,
                              name="RL Agent", line=dict(color='#667eea', width=3)),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=list(range(len(ubah_values))), y=ubah_values,
                              name="Buy & Hold", line=dict(color='#f093fb', width=3, dash='dash')),
                    row=1, col=1
                )
                
                # Allocation chart
                allocations_array = np.array(allocations_history)
                for i, stock in enumerate(STOCKS):
                    fig.add_trace(
                        go.Scatter(x=list(range(len(allocations_history))),
                                  y=allocations_array[:, i],
                                  name=stock,
                                  stackgroup='one',
                                  mode='lines'),
                        row=2, col=1
                    )
                
                fig.update_xaxes(title_text="Trading Steps", row=1, col=1)
                fig.update_xaxes(title_text="Trading Steps", row=2, col=1)
                fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
                fig.update_yaxes(title_text="Allocation", row=2, col=1)
                
                fig.update_layout(
                    height=800,
                    hovermode='x unified',
                    showlegend=True,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.markdown("### 📈 Performance Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    returns_rl = np.diff(net_worths) / net_worths[:-1]
                    returns_bh = np.diff(ubah_values) / ubah_values[:-1]
                    
                    stats_df = pd.DataFrame({
                        'Metric': ['Total Return', 'Max Drawdown', 'Volatility', 'Win Rate'],
                        'RL Agent': [
                            f"{((net_worths[-1]/initial_balance - 1) * 100):.2f}%",
                            f"{(np.min(np.minimum.accumulate(net_worths) - net_worths) / np.maximum.accumulate(net_worths).max() * 100):.2f}%",
                            f"{(np.std(returns_rl) * np.sqrt(252) * 100):.2f}%",
                            f"{(np.sum(returns_rl > 0) / len(returns_rl) * 100):.1f}%"
                        ],
                        'Buy & Hold': [
                            f"{((ubah_values[-1]/initial_balance - 1) * 100):.2f}%",
                            f"{(np.min(np.minimum.accumulate(ubah_values) - ubah_values) / np.maximum.accumulate(ubah_values).max() * 100):.2f}%",
                            f"{(np.std(returns_bh) * np.sqrt(252) * 100):.2f}%",
                            f"{(np.sum(returns_bh > 0) / len(returns_bh) * 100):.1f}%"
                        ]
                    })
                    
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                with col2:
                    # Final allocation pie chart
                    final_allocation = allocations_history[-1]
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=STOCKS,
                        values=final_allocation,
                        hole=.3,
                        marker=dict(colors=['#667eea', '#764ba2', '#f093fb', '#4facfe'])
                    )])
                    fig_pie.update_layout(title="Final Portfolio Allocation", height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            else:
                st.error("❌ Failed to load model weights. Please check the model directory.")
    
    else:
        st.info("👆 Please upload CSV files to begin the simulation")

with tab2:
    if uploaded_files:
        st.markdown("### 📊 Data Preview")
        
        for i, file in enumerate(uploaded_files[:4]):
            file.seek(0)
            df_preview = pd.read_csv(file)
            
            with st.expander(f"📄 {file.name} ({STOCKS[i] if i < len(STOCKS) else 'Asset'})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.dataframe(df_preview.head(10), use_container_width=True)
                
                with col2:
                    st.write("**Statistics:**")
                    st.write(df_preview.describe())
    else:
        st.info("Upload files in the Trading Simulation tab to preview data")

with tab3:
    st.markdown("""
    ## 🧠 How It Works
    
    ### Architecture
    This trading agent uses **Proximal Policy Optimization (PPO)**, a state-of-the-art reinforcement learning algorithm.
    
    ### Model Components
    - **Actor Network**: Decides portfolio allocation across assets
    - **Critic Network**: Evaluates the quality of the actor's decisions
    - **EIIE Architecture**: Ensemble of Identical Independent Evaluators for multi-asset trading
    
    ### Training Process
    1. **State**: Market data (OHLCV) for multiple cryptocurrencies
    2. **Action**: Portfolio weights across assets
    3. **Reward**: Log return of portfolio value
    4. **Optimization**: PPO with clipped objective for stable training
    
    ### Key Features
    - 🎯 Multi-asset portfolio management
    - 📊 Considers transaction costs (0.1%)
    - 🔄 Rebalances portfolio at each time step
    - 📈 Compares against buy-and-hold strategy
    
    ### Advantages
    - Adapts to market conditions
    - Risk-aware allocation
    - No manual feature engineering needed
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Built with Streamlit • TensorFlow • Reinforcement Learning</p>
        <p>⚠️ For educational purposes only. Not financial advice.</p>
    </div>
""", unsafe_allow_html=True)