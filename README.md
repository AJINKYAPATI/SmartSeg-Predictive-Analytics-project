<div align="center">

# ◈ SmartSeg
### *Customer Intelligence & Segmentation Platform*

**Automatically discover who your customers really are — using Machine Learning.**

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://smartseg-predictive-analytics-project-qq3ptgcc8n6pjmvdrusshf.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-KMeans-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-Academic-1A56A0?style=for-the-badge)](/)

</div>

---

## 🎯 What Is This Project?

SmartSeg helps businesses answer one critical question:

> **"Who are my customers, and how should I treat each group differently?"**

It reads your customer transaction data, does the math, and automatically groups your customers based on **3 simple but powerful rules** — when they last bought, how often they buy, and how much they spend. Then it shows you exactly who falls into each group — with beautiful charts — so you can market smarter.

No coding needed. Just upload your data and let the machine do the work.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📂 **Data Upload** | Upload your own `.txt` transaction file or use the built-in sample data |
| 🧹 **Data Cleaning** | Automatically checks for missing values, duplicates, and data quality issues |
| 📊 **Visual Exploration** | Interactive histograms, box plots, and time-series trend charts |
| 🤖 **KMeans Clustering** | Groups customers into segments using unsupervised machine learning |
| 📐 **Elbow Method** | Finds the mathematically optimal number of customer groups |
| 🗺️ **Rich Visualisations** | Scatter plots, treemaps, and 3D cluster charts — all interactive |
| 🔮 **Live Predictions** | Enter a new customer's details and instantly predict their segment |
| ⬇️ **CSV Export** | Download prediction results for use in your CRM or email tool |

---

## 🧠 How It Works — The Simple Version

```
Your Transaction Data
        ↓
   Calculate 3 scores per customer:
   R → How RECENTLY did they buy?
   F → How FREQUENTLY do they buy?
   M → How MUCH money do they spend?
        ↓
   KMeans groups similar customers together
        ↓
   You get clear segments like:
   👑 Champions  →  Reward them
   💤 At-Risk    →  Win them back
   🌱 Potential  →  Nurture them
        ↓
   Predict where NEW customers belong
```

---

## 🚀 Try It Live

**No installation required.** Click the button below to open the live app:

<div align="center">

### 👉 [Open SmartSeg App](https://smartseg-predictive-analytics-project-qq3ptgcc8n6pjmvdrusshf.streamlit.app/)

</div>

---

## 🗂️ App Pages Walkthrough

```
📋 Page 1 — Business Understanding
         Why this project exists and what problem it solves

📊 Page 2 — Data Understanding
         Load your data and see a quick overview

🔧 Page 3 — Data Preparation
         Clean the data and explore distributions

🤖 Page 4 — Modeling & Evaluation
         Run KMeans, pick clusters, view results

🔮 Page 5 — Predict
         Classify new customers into segments instantly
```

---

## 🛠️ Tech Stack

```python
Language    →  Python 3.9+
Web App     →  Streamlit
ML Model    →  Scikit-learn  (KMeans Clustering)
Data        →  Pandas, NumPy
Charts      →  Plotly Express, Matplotlib, Squarify
Deployment  →  Streamlit Cloud
```

---

## 📁 Project Structure

```
SmartSeg/
│
├── main.py                  # Main Streamlit application
├── data/
│   └── CDNOW_master.txt     # Sample transaction dataset
├── kmeans_model.pkl         # Exported trained model (auto-generated)
├── feedback.csv             # User feedback log (auto-generated)
└── README.md                # You are here
```

---

## ⚡ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/smartseg.git
cd smartseg

# 2. Install dependencies
pip install streamlit pandas scikit-learn plotly matplotlib squarify

# 3. Launch the app
streamlit run main.py
```

Then open your browser at `http://localhost:8501`

---

## 📊 Sample Dataset

The app uses the **CDNOW transactional dataset** — a classic benchmark in customer behaviour research.

| Column | Description | Example |
|--------|-------------|---------|
| `Customer_id` | Unique customer identifier | `1234` |
| `day` | Purchase date (YYYYMMDD) | `19970115` |
| `Quantity` | Items purchased | `2` |
| `Sales` | Transaction value (USD) | `$29.99` |

> **Time period:** January 1997 – June 1998 · ~23,570 unique customers · ~69,659 transactions

---

## 📈 What You Get — Example Segments

After running the model, every customer gets a label:

| Segment | Who Are They? | What To Do? |
|---------|--------------|-------------|
| 👑 **Champions** | Buy often, spend big, bought recently | Reward with VIP perks |
| 💤 **At-Risk / Lost** | Haven't bought in a long time | Send win-back offers |
| 🌱 **Potential Loyalists** | Moderate buyers with growth potential | Nurture with incentives |

---

## 🤝 Feedback

The app includes a built-in feedback form on each page. Ratings and suggestions are saved to `feedback.csv` and the 5 most recent responses are displayed in-app.

---

<div align="center">

**Built with ❤️ using Python & Streamlit**

*Academic Project — Data Science & Machine Learning*

[![Live App](https://img.shields.io/badge/Try%20SmartSeg%20Live-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://smartseg-predictive-analytics-project-qq3ptgcc8n6pjmvdrusshf.streamlit.app/)

</div>
