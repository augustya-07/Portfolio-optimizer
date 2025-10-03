import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Page setup
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("📈 Goldman Sachs-Style Portfolio Optimizer")
st.markdown("""
**Powered by Markowitz Mean-Variance Optimization**  
*Maximize returns for your chosen risk level*
""")

# Sidebar for user input
st.sidebar.header("Investment Parameters")
tickers_input = st.sidebar.text_input("Enter stock tickers (comma-separated)", "GS, JPM, MS, BLK, SPY")
risk_level = st.sidebar.select_slider("Risk Level", options=["Low", "Medium", "High"], value="Medium")
max_weight = st.sidebar.slider("Max % per Stock", 10, 100, 40)

# Process the tickers
tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]

# Button to run optimization
if st.sidebar.button("🚀 Optimize My Portfolio", type="primary"):
    with st.spinner("Analyzing market data and optimizing..."):
        try:
            # Download data
            data = yf.download(tickers, period="5y", auto_adjust=True)
            stock_data = data['Close']
            stock_data.ffill(inplace=True)
            stock_data.bfill(inplace=True)
            
            # Calculate expected returns and covariance
            mu = expected_returns.mean_historical_return(stock_data)
            S = risk_models.sample_cov(stock_data)
            
            # Optimize portfolio based on risk level
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w >= 0)
            ef.add_constraint(lambda w: w <= max_weight/100)
            
            if risk_level == "Low":
                ef.min_volatility()  # Minimum risk
            elif risk_level == "High":
                ef.max_sharpe()  # Maximum return (higher risk)
            else:
                ef.max_quadratic_utility()  # Balanced approach
                
            weights = ef.clean_weights()
            
            # Display results in two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Optimal Portfolio Allocation")
                # Create a nice dataframe
                weights_df = pd.DataFrame({
                    'Stock': list(weights.keys()),
                    'Weight': [f"{w:.2%}" for w in weights.values()]
                })
                st.dataframe(weights_df, hide_index=True, use_container_width=True)
                
                # Pie chart
                fig = px.pie(
                    values=list(weights.values()),
                    names=list(weights.keys()),
                    title="Portfolio Allocation"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Performance metrics
                expected_return, expected_volatility, sharpe_ratio = ef.portfolio_performance()
                st.subheader("Expected Performance (Annualized)")
                
                metrics_data = {
                    'Metric': ['Return', 'Volatility (Risk)', 'Sharpe Ratio'],
                    'Value': [
                        f"{expected_return:.2%}", 
                        f"{expected_volatility:.2%}", 
                        f"{sharpe_ratio:.3f}"
                    ]
                }
                st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)
                
                # Normalized performance chart
                st.subheader("Historical Performance")
                normalized_prices = (stock_data / stock_data.iloc[0] * 100)
                fig = px.line(normalized_prices, title="Growth of ₹100 Investment")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**How it works:**  
This tool uses Modern Portfolio Theory (Nobel Prize, 1990) to optimize your investments.  
It finds the perfect balance between stocks to maximize returns while minimizing risk.
""")
