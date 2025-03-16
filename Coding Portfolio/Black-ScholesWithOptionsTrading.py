import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import qfin as qf

def black_scholes_call(S, K, sigma, r, t):
    d1 = (np.log(S/K) + (r + (sigma**2)/2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return S * norm.cdf(d1) - K * np.exp(-r*t) * norm.cdf(d2)

# Initial parameters
S0 = 100  # Initial stock price
K = 100   # Strike price
sigma = 0.3  # Volatility
r = 0.05  # Risk-free rate
T = 1  # Time to expiration in years

def simulate_trades(num_trades=100000):
    premium_list = []
    pls = []
    
    for _ in range(num_trades):
        path = qf.simulations.GeometricBrownianMotion(S0, r, sigma, 1/252, 1)
        final_price = path.simulated_path[-1]
        
        # Compute dynamic premium
        option_price = black_scholes_call(S0, K, sigma, r, T)
        bid_price, ask_price = option_price - 0.3, option_price + 0.3  # Simulated spread
        trade_price = np.random.uniform(bid_price, ask_price)  # Assume execution within the spread
        
        premium_paid = trade_price * 100
        payoff = max(final_price - K, 0) * 100
        
        pls.append(payoff - premium_paid)
        premium_list.append(premium_paid)
    
    return np.array(pls), np.array(premium_list)

# Run simulation
pls, premiums = simulate_trades()

# Plot results
plt.figure(figsize=(10, 5))
plt.title("Trading Edge Over Time")
plt.plot(np.cumsum(pls), label="Account Equity")
plt.axhline(0, color='red', linestyle='dashed', label="Break-even")
plt.xlabel("Number of Trades")
plt.ylabel("Portfolio Value")
plt.legend()
plt.style.use("dark_background")
plt.show()

# Print final results
print(f"Average P/L per trade: {np.mean(pls):.2f}")
print(f"Total profit/loss after {len(pls)} trades: {np.sum(pls):.2f}")
