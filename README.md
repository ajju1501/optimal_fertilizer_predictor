# 🌾 Fertilizer Recommendation System

A machine learning-based system that recommends the **top 3 fertilizers** based on soil, crop, and environmental parameters. Built with **Python, Streamlit, and XGBoost**, this app helps farmers optimize fertilizer usage for better crop yield.

---

## 🛠 Features

- Predicts **top 3 fertilizers** with confidence scores.
- Displays **N-P-K composition** of recommended fertilizers.
- Takes **environmental inputs** like temperature, humidity, moisture.
- Supports **various crop types** and **soil types**.
- Built as an **interactive Streamlit web app**.

---

## ⚡ Technologies Used

- **Python** – Core programming language  
- **Streamlit** – Web app framework for interactive UI  
- **XGBoost** – Machine learning model for fertilizer recommendation  
- **Pandas & NumPy** – Data handling  
- **Joblib** – Model serialization

---

## 🧪 Inputs

| Input | Type | Description |
|-------|------|-------------|
| Temperature | Number | Current temperature (°C) |
| Humidity | Number | Soil humidity (%) |
| Moisture | Number | Soil moisture (%) |
| Nitrogen (N) | Number | Nitrogen content in soil |
| Phosphorus (P) | Number | Phosphorus content in soil |
| Potassium (K) | Number | Potassium content in soil |
| Soil Type | Select | Options: Black, Clayey, Loamy, Red, Sandy |
| Crop Type | Select | Options: Barley, Cotton, Ground Nuts, Maize, Millets, Oil seeds, Paddy, Pulses, Sugarcane, Tobacco, Wheat |

---

## 🌱 Fertilizer Recommendations

The system predicts fertilizers like:

- `14-35-14`  
- `10-26-26`  
- `17-17-17`  
- `28-28`  
- `20-20`  
- `DAP`  
- `Urea`

Each recommendation includes its **N-P-K composition** and **confidence score**.

---

## 🚀 How to Run Locally

1. **Clone the repo:**
```bash
git clone https://github.com/ajju1501/optimal_fertilizer_predictor.git
cd optimal_fertilizer_predictor

Create a virtual environment:

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

Requirements:
scikit-learn
streamlit
joblib
numpy
XGBoost


Run the Streamlit app:

streamlit run app.py
