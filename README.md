# 📊 Analyzing Historical Stock Revenue Data and Building a Dashboard

This project demonstrates the process of **analyzing historical stock and revenue data** and **visualizing insights through an interactive Streamlit dashboard**.  
It combines Python-based data analysis with a modern, responsive web interface for real-time financial exploration.

---

## 🔗 Link
https://stock-revenue.streamlit.app/

---

## 🚀 Overview

The repository contains:
- **Data analysis notebook** (`Extracting and Visualizing Stock Data.ipynb`) – performs data collection, cleaning, and visualization of stock and revenue trends.
- **Streamlit dashboard** (`app.py`) – transforms the insights into an interactive web app for exploring stock performance, trends, and financial indicators.
- **Configuration and dependencies** for easy local setup.

---

## 🧠 Key Features

- 📈 **Multi-Ticker Stock Analysis**: Compare multiple stocks over a custom time range.  
- ⚙️ **Dynamic Indicators**: Enable or disable Moving Averages (20/50/200), RSI, and MACD.  
- 📊 **KPI Overview**: View Total Return, Max Drawdown, and Sharpe Ratio for each stock.  
- 🧮 **Normalized Performance View**: Benchmark all tickers starting at 100 for fair comparison.  
- 📂 **Custom CSV Upload**: Analyze your own stock or asset data directly in the dashboard.  
- 💾 **One-Click Export**: Download processed data as CSV.  

---

## 🧰 Tech Stack

| Category | Tools |
|-----------|-------|
| **Data Retrieval** | [yfinance](https://pypi.org/project/yfinance/) |
| **Data Analysis** | pandas, numpy |
| **Visualization** | matplotlib |
| **Frontend / Dashboard** | Streamlit |
| **Environment** | Python ≥ 3.9 |


---

## 🖥️ Installation & Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/Analyzing-Historical-Stock-Revenue-Data-and-Building-a-Dashboard.git
cd Analyzing-Historical-Stock-Revenue-Data-and-Building-a-Dashboard
python -m pip install -r requirements.txt
```

Install deps and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---
## 📁 Project Structure
```bash
📦 Analyzing-Historical-Stock-Revenue-Data-and-Building-a-Dashboard
├── Extracting and Visualizing Stock Data.ipynb   # Data analysis notebook
├── app.py                                        # Streamlit dashboard
├── requirements.txt                              # Dependencies
├── config.toml                                   # Streamlit theme configuration
└── README.md                                     # Project documentation

```

## 📈 Example Use Case

- Enter multiple tickers, such as AAPL, MSFT, GOOGL.
- Select your preferred date range.
- Enable Moving Averages and RSI from the sidebar.
- Instantly visualize insights, returns, and performance metrics.
- Download the processed data for further analysis.
