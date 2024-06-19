import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# Helper functions
def get_data(tickers, start_date, end_date):
    return yf.download(tickers, start=start_date, end=end_date)['Adj Close']

def calculate_returns(prices):
    return prices.pct_change().dropna()

def portfolio_performance(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_std

def negative_sharpe_ratio(weights, returns, risk_free_rate=0.01):
    p_return, p_std = portfolio_performance(weights, returns)
    return -(p_return - risk_free_rate) / p_std

def max_sharpe_ratio(returns):
    num_assets = returns.shape[1]
    args = (returns,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def calculate_var(returns, confidence_level=0.05):
    return returns.quantile(confidence_level, axis=0)  # Returns a series of quantiles for each asset

# Historical price data collection
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
data = get_data(tickers, '2020-01-01', '2023-01-01')

# Calculate daily returns
returns = calculate_returns(data)

# Performance analysis
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
risk_free_rate = 0.01

# Portfolio optimization
optimal_portfolio = max_sharpe_ratio(returns)
optimal_weights = optimal_portfolio.x

# Calculate performance of optimized portfolio
opt_return, opt_std = portfolio_performance(optimal_weights, returns)
opt_sharpe = (opt_return - risk_free_rate) / opt_std

# Calculate VaR
VaR_95 = calculate_var(returns, 0.05)

# Streamlit application setup
st.title('Banking Investments - Statistical Dashboard')

# List of Evaluated Assets
st.subheader('Evaluated Assets')
st.write(", ".join(tickers))

# Objective
with st.expander("Explanation of the Banking Investments Statistical Dashboard"):
    st.write("""
This dashboard was developed to assist in the analysis and optimization of investments in a set of popular financial assets such as Apple (AAPL), Microsoft (MSFT), Google (GOOGL), Amazon (AMZN), and Tesla (TSLA).

**Objectives:**
- **Performance Analysis:** Evaluate how these assets have performed over time in terms of return and risk.
- **Portfolio Optimization:** Determine the optimal allocation of each asset to maximize risk-adjusted return.
- **Sensitivity Analysis:** Understand how different scenarios, such as changes in the risk-free rate, affect portfolio performance.

**Indicators Used:**
1. **Expected Annual Return:** Weighted average of expected annual returns for the selected assets.
2. **Annual Volatility:** Indicates the degree of fluctuation in returns over time, reflecting risk.
3. **Sharpe Ratio:** Measures risk-adjusted return, considering the risk-free rate as a benchmark.
4. **VaR 95% (Value at Risk):** Estimates the worst expected loss with 95% confidence, providing a measure of risk.

**Questions Answered:**
- **What was the performance of the portfolio over time?** The cumulative performance chart shows how the investment evolved.
- **How is risk and return distributed among different assets?** The distribution of returns and risk provides insights into optimal asset allocation.
- **How does portfolio performance vary with different levels of risk-free rate?** Sensitivity analysis helps understand the impact of changes in the risk-free rate on the portfolio.

**Conclusions:**
- **Positive Results:** Indicate good portfolio performance with returns higher than expected and a positive Sharpe Ratio. 📈
- **Negative Results:** Suggest the portfolio may not be optimized for the assumed risk, with returns below expectations or a lower Sharpe Ratio. 📉
""")
    
# Cumulative Performance Chart
st.subheader('Cumulative Performance Chart')
portfolio_returns = returns.dot(optimal_weights)
cumulative_returns = (1 + portfolio_returns).cumprod()
st.line_chart(cumulative_returns)

# Analysis of Cumulative Performance Chart
with st.expander("Analysis of Cumulative Performance Chart"):
    st.write("""
    The cumulative performance chart shows how the investment has evolved over time. We observe an overall growth trend, reflecting the cumulative return of the optimized portfolio. It's important to note that periods of high volatility may be associated with greater variability in daily returns, which directly impacts long-term growth trajectory.
    """)
    if portfolio_returns.iloc[-1] > 1:
        st.write(":chart_with_upwards_trend: **Evaluation:** The result is positive, showing consistent investment growth over the analyzed period, despite short-term fluctuations.")
    else:
        st.write(":chart_with_downwards_trend: **Evaluation:** The result is negative, indicating a possible decline in investment over the analyzed period.")

# Performance Metrics in Cards
st.subheader('Summary of Performance Metrics')
col1, col2, col3, col4 = st.columns(4)
col1.metric(label='Expected Annual Return', value=f'{opt_return:.2%}')
col2.metric(label='Annual Volatility', value=f'{opt_std:.2%}')
col3.metric(label='Sharpe Ratio', value=f'{opt_sharpe:.2f}')
col4.metric(label='VaR 95%', value=f'{VaR_95.mean():.2%}')  # Adjusted to display a single VaR value

# Analysis of Performance Metrics
with st.expander("Analysis of Performance Metrics"):
    st.write("""
    The presented metrics are essential for evaluating portfolio performance and risk. **Expected Annual Return** reflects the weighted average of expected returns for the selected assets. **Annual Volatility** indicates the degree of fluctuation in returns, serving as a risk indicator. **Sharpe Ratio** adjusts portfolio return for its risk, considering the risk-free rate. **VaR 95%** provides a measure of loss risk, indicating the worst expected outcome with 95% confidence.
    """)
    if opt_return > 0 and opt_sharpe > 0:
        st.write(":white_check_mark: **Evaluation:** The results are positive. Expected Annual Return and Sharpe Ratio are above expectations, indicating a well-optimized portfolio relative to assumed risk.")
    elif opt_return < 0 or opt_sharpe < 0:
        st.write(":x: **Evaluation:** The results are negative. Expected Annual Return and/or Sharpe Ratio are below expectations, suggesting the portfolio may not be optimized relative to assumed risk.")

# Distribution of Returns and Risk
st.subheader('Distribution of Returns and Risk')
st.bar_chart(optimal_weights)

# Analysis of Distribution of Returns and Risk
with st.expander("Analysis of Distribution of Returns and Risk"):
    st.write("""
    The weight distribution shows how asset allocation is distributed within the optimized portfolio. We observe that some assets may have a more significant share, reflecting their expected contributions to portfolio return.
    """)
    if np.any(optimal_weights < 0):
        st.write(":warning: **Evaluation:** The distribution may not be ideal due to negative allocation in some assets.")
    else:
        st.write(":white_check_mark: **Evaluation:** The distribution appears balanced, with adequate diversification among selected assets.")

# Sensitivity Analysis
st.subheader('Sensitivity Analysis')
risk_free_rate_slider = st.slider('Risk-Free Rate (%)', min_value=0.0, max_value=5.0, value=1.0, step=0.1)
st.write(f"Selected Risk-Free Rate: {risk_free_rate_slider}%")

# Recalculation of optimized portfolio performance with new risk-free rate
opt_sharpe_adjusted = (opt_return - risk_free_rate_slider / 100) / opt_std

# Analysis of Sensitivity to Risk-Free Rate
with st.expander("Analysis of Sensitivity to Risk-Free Rate"):
    st.write("""
    The risk-free rate is a critical factor in Sharpe Ratio calculation, affecting the relationship between portfolio return and risk.
    """)
    if opt_sharpe_adjusted > 0:
        st.write(":white_check_mark: **Evaluation:** Adjusting the risk-free rate allows for a more accurate assessment of risk-adjusted return, crucial in different economic scenarios.")
    else:
        st.write(":x: **Evaluation:** Adjusting the risk-free rate may not be contributing significantly to improving the portfolio's Sharpe Ratio.")

# Explanatory and Methodological Notes
st.subheader('Explanatory and Methodological Notes')
st.write("""
In this dashboard, we used a portfolio optimization-based methodology to calculate performance and risk metrics. Historical asset prices were obtained from Yahoo Finance, and daily returns were calculated based on these prices. Portfolio optimization was performed using the minimization method to maximize the Sharpe Ratio, considering the trade-off between risk and return.

Assumptions include the normal distribution or close to it of asset returns and market efficiency in reflecting past and present information. It is important to note that past results do not guarantee future returns, and analysis should be continuously reviewed and adjusted as new information becomes available.

**Overall Assessment:** The use of these techniques provided positive results, reflected in performance metrics and consistent growth of the optimized portfolio over the analyzed period.
""")

