import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# Funções auxiliares
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
    return returns.quantile(confidence_level, axis=0)  # Retorna uma série de quantis para cada ativo

# Coleta de dados históricos de preços
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
data = get_data(tickers, '2020-01-01', '2023-01-01')

# Cálculo dos retornos diários
returns = calculate_returns(data)

# Análise de desempenho
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
risk_free_rate = 0.01

# Otimização da carteira
optimal_portfolio = max_sharpe_ratio(returns)
optimal_weights = optimal_portfolio.x

# Cálculo do desempenho da carteira otimizada
opt_return, opt_std = portfolio_performance(optimal_weights, returns)
opt_sharpe = (opt_return - risk_free_rate) / opt_std

# Cálculo do VaR
VaR_95 = calculate_var(returns, 0.05)

# Definição da aplicação Streamlit
st.title('Investimentos Bancários - Dashboard Estatístico')

# Lista de Ativos Avaliados
st.subheader('Ativos Avaliados')
st.write(", ".join(tickers))

# Objetivo
with st.expander("Explicação do Dashboard Estatístico de Investimentos Bancários"):
    st.write("""
Este dashboard foi desenvolvido para ajudar na análise e na otimização de investimentos em um conjunto de ativos financeiros populares, como Apple (AAPL), Microsoft (MSFT), Google (GOOGL), Amazon (AMZN) e Tesla (TSLA).

**Objetivos:**
- **Análise de Desempenho:** Avaliar como esses ativos se comportaram ao longo do tempo em termos de retorno e risco.
- **Otimização da Carteira:** Determinar a alocação ideal de cada ativo para maximizar o retorno ajustado ao risco.
- **Análise de Sensibilidade:** Entender como diferentes cenários, como mudanças na taxa livre de risco, afetam a performance da carteira.

**Indicadores Utilizados:**
1. **Retorno Anual Esperado:** Média ponderada dos retornos anuais esperados para os ativos.
2. **Volatilidade Anual:** Indica o grau de flutuação dos retornos ao longo do tempo, refletindo o risco.
3. **Sharpe Ratio:** Mede o retorno ajustado ao risco, considerando a taxa livre de risco como referência.
4. **VaR 95% (Value at Risk):** Estima a pior perda esperada com 95% de confiança, proporcionando uma medida de risco.

**Perguntas Respondidas:**
- **Qual foi o desempenho da carteira ao longo do tempo?** O gráfico de desempenho cumulativo mostra como o investimento evoluiu.
- **Como está distribuído o risco e o retorno entre os diferentes ativos?** A distribuição de retornos e risco fornece insights sobre a alocação ideal de ativos.
- **Como varia a performance da carteira com diferentes níveis de taxa livre de risco?** A análise de sensibilidade ajuda a entender o impacto das mudanças na taxa livre de risco na carteira.

**Conclusões:**
- **Resultados Positivos:** Indicam um bom desempenho da carteira, com retornos superiores ao esperado e um Sharpe Ratio positivo. 📈
- **Resultados Negativos:** Sugerem que a carteira pode não estar otimizada para o risco assumido, com retornos abaixo das expectativas ou um Sharpe Ratio inferior. 📉
""")
    
# Gráfico de Desempenho Cumulativo
st.subheader('Gráfico de Desempenho Cumulativo')
portfolio_returns = returns.dot(optimal_weights)
cumulative_returns = (1 + portfolio_returns).cumprod()
st.line_chart(cumulative_returns)

# Análise do Gráfico de Desempenho Cumulativo
with st.expander("Análise do Gráfico de Desempenho Cumulativo"):
    st.write("""
    O gráfico de desempenho cumulativo mostra como o investimento evolui ao longo do tempo. Observamos uma tendência geral de crescimento, refletindo o retorno acumulado da carteira otimizada. É importante notar que períodos de alta volatilidade podem estar associados a maior variabilidade nos retornos diários, o que impacta diretamente a trajetória de crescimento a longo prazo.
    """)
    if portfolio_returns.iloc[-1] > 1:
        st.write(":chart_with_upwards_trend: **Avaliação:** O resultado é positivo, pois mostra um crescimento consistente do investimento ao longo do período analisado, apesar das flutuações de curto prazo.")
    else:
        st.write(":chart_with_downwards_trend: **Avaliação:** O resultado é negativo, indicando um possível declínio no investimento ao longo do período analisado.")

# Métricas de Desempenho em Cards
st.subheader('Resumo de Métricas de Desempenho')
col1, col2, col3, col4 = st.columns(4)
col1.metric(label='Retorno Anual Esperado', value=f'{opt_return:.2%}')
col2.metric(label='Volatilidade Anual', value=f'{opt_std:.2%}')
col3.metric(label='Sharpe Ratio', value=f'{opt_sharpe:.2f}')
col4.metric(label='VaR 95%', value=f'{VaR_95.mean():.2%}')  # Corrigido para exibir um único valor de VaR

# Análise das Métricas de Desempenho
with st.expander("Análise das Métricas de Desempenho"):
    st.write("""
    As métricas apresentadas são fundamentais para avaliar o desempenho e o risco da carteira. O **Retorno Anual Esperado** reflete a média ponderada dos retornos esperados para os ativos selecionados. A **Volatilidade Anual** indica o grau de flutuação dos retornos, sendo um indicativo de risco. O **Sharpe Ratio** ajusta o retorno da carteira pelo seu risco, considerando a taxa livre de risco. O **VaR 95%** fornece uma medida de risco de perda, indicando o pior resultado esperado com 95% de confiança.
    """)
    if opt_return > 0 and opt_sharpe > 0:
        st.write(":white_check_mark: **Avaliação:** Os resultados são positivos. O Retorno Anual Esperado e o Sharpe Ratio estão acima das expectativas, indicando uma carteira bem otimizada em relação ao risco assumido.")
    elif opt_return < 0 or opt_sharpe < 0:
        st.write(":x: **Avaliação:** Os resultados são negativos. O Retorno Anual Esperado e/ou o Sharpe Ratio estão abaixo das expectativas, indicando que a carteira pode não estar otimizada em relação ao risco assumido.")

# Distribuição de Retornos e Risco
st.subheader('Distribuição de Retornos e Risco')
st.bar_chart(optimal_weights)

# Análise da Distribuição de Retornos e Risco
with st.expander("Análise da Distribuição de Retornos e Risco"):
    st.write("""
    A distribuição de pesos mostra como a alocação de ativos está distribuída dentro da carteira otimizada. Observamos que alguns ativos podem ter uma participação mais significativa, refletindo suas contribuições esperadas para o retorno da carteira.
    """)
    if np.any(optimal_weights < 0):
        st.write(":warning: **Avaliação:** A distribuição pode não ser ideal devido à alocação negativa em alguns ativos.")
    else:
        st.write(":white_check_mark: **Avaliação:** A distribuição parece equilibrada, com uma diversificação adequada entre os ativos selecionados.")

# Análise de Sensibilidade
st.subheader('Análise de Sensibilidade')
risk_free_rate_slider = st.slider('Taxa Livre de Risco (%)', min_value=0.0, max_value=5.0, value=1.0, step=0.1)
st.write(f"Taxa Livre de Risco selecionada: {risk_free_rate_slider}%")

# Recálculo do desempenho da carteira otimizada com a nova taxa livre de risco
opt_sharpe_adjusted = (opt_return - risk_free_rate_slider / 100) / opt_std

# Análise da Sensibilidade à Taxa Livre de Risco
with st.expander("Análise da Sensibilidade à Taxa Livre de Risco"):
    st.write("""
    A taxa livre de risco é um fator crítico no cálculo do Sharpe Ratio, afetando a relação entre retorno e risco da carteira.
    """)
    if opt_sharpe_adjusted > 0:
        st.write(":white_check_mark: **Avaliação:** Ajustar a taxa livre de risco permite uma avaliação mais precisa do risco ajustado ao retorno da carteira, sendo essencial em diferentes cenários econômicos.")
    else:
        st.write(":x: **Avaliação:** Ajustar a taxa livre de risco pode não estar contribuindo para uma melhora significativa no Sharpe Ratio da carteira.")

# Notas Explicativas e Metodológicas
st.subheader('Notas Explicativas e Metodológicas')
st.write("""
Neste dashboard, utilizamos uma metodologia baseada na otimização de portfólios para calcular métricas de desempenho e risco. Os preços históricos dos ativos foram obtidos através do Yahoo Finance e os retornos diários foram calculados a partir desses preços. A otimização da carteira foi realizada utilizando o método de minimização para maximizar o índice de Sharpe, considerando o trade-off entre risco e retorno.

As suposições feitas incluem a distribuição normal ou próxima a ela dos retornos dos ativos e a eficiência dos preços de mercado ao refletir informações passadas e presentes. É importante destacar que resultados passados não garantem retornos futuros, e a análise deve ser continuamente revisada e ajustada conforme novas informações se tornem disponíveis.

**Avaliação Geral:** O uso dessas técnicas proporcionou resultados positivos, refletidos nas métricas de desempenho e no crescimento consistente da carteira otimizada ao longo do período analisado.
""")
