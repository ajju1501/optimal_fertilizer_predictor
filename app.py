import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
clf = joblib.load("xgboost_model.pkl")   # GridSearchCV object

st.set_page_config(page_title="Fertilizer Recommendation", page_icon="🌱", layout="centered")
st.title("🌾 Fertilizer Recommendation System")
st.write("Enter details to get the **top 3 fertilizer suggestions**.")

# ---------------------------
# Inputs
# ---------------------------
temperature = st.number_input("🌡️ Temperature (°C)", -10, 60, 25, key="temp")
humidity = st.number_input("💧 Humidity (%)", 0, 100, 60, key="humidity")
moisture = st.number_input("🌊 Moisture (%)", 0, 100, 30, key="moisture")

nitrogen = st.number_input("🧪 Nitrogen (N)", 0, 200, 40, key="n")
potassium = st.number_input("🧪 Potassium (K)", 0, 200, 50, key="k")
phosphorous = st.number_input("🧪 Phosphorous (P)", 0, 200, 30, key="p")

soil_type = st.selectbox("🪨 Soil Type", ["Black", "Clayey", "Loamy", "Red", "Sandy"], key="soil")
crop_type = st.selectbox("🌱 Crop Type", ["Barley", "Cotton", "Ground Nuts", "Maize", "Millets",
                                          "Oil seeds", "Paddy", "Pulses", "Sugarcane", "Tobacco", "Wheat"], key="crop")

# ---------------------------
# One-hot encoding (manual)
# ---------------------------
soil_types = ["Black", "Clayey", "Loamy", "Red", "Sandy"]
crop_types = ["Barley", "Cotton", "Ground Nuts", "Maize", "Millets",
              "Oil seeds", "Paddy", "Pulses", "Sugarcane", "Tobacco", "Wheat"]

soil_ohe = [1 if soil_type == s else 0 for s in soil_types]
crop_ohe = [1 if crop_type == c else 0 for c in crop_types]

# Final feature vector (6 numeric + 11 crop + 5 soil = 22 features)
features = [temperature, humidity, moisture, nitrogen, potassium, phosphorous] + crop_ohe + soil_ohe

# Create DataFrame with correct feature names
feature_names = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"] + \
                [f"Crop Type_{c}" for c in crop_types] + \
                [f"Soil Type_{s}" for s in soil_types]

X_input = pd.DataFrame([features], columns=feature_names)

# ---------------------------
# Fertilizer mapping
# ---------------------------
fertilizer_map = { 0: "14-35-14", 1: "10-26-26", 2: "17-17-17", 3: "28-28", 4: "20-20", 5: "DAP", 6: "Urea" }

fertilizer_details = {
    "14-35-14": {"N": 14, "P2O5": 35, "K2O": 14},
    "10-26-26": {"N": 10, "P2O5": 26, "K2O": 26},
    "17-17-17": {"N": 17, "P2O5": 17, "K2O": 17},
    "28-28": {"N": 28, "P2O5": 28, "K2O": 0},
    "20-20": {"N": 20, "P2O5": 20, "K2O": 0},
    "DAP": {"N": 18, "P2O5": 46, "K2O": 0},
    "Urea": {"N": 46, "P2O5": 0, "K2O": 0}
}


# ---------------------------
# Prediction
# ---------------------------
if st.button("🔍 Recommend Fertilizers"):
    try:
        preds = clf.best_estimator_.predict_proba(X_input)

        top3_idx = np.argsort(preds[0])[-3:][::-1]  # top 3

        st.subheader("🌱 Top Fertilizer Recommendations:")
        for i, idx in enumerate(top3_idx, start=1):
            fert_name = fertilizer_map.get(idx, f"Class {idx}")
            details = fertilizer_details.get(fert_name, {})
            st.write(f"{i}. **{fert_name}** — confidence: `{preds[0][idx]:.2f}` | "
                    f"N:{details.get('N')} P:{details.get('P2O5')} K:{details.get('K2O')}")


    except Exception as e:
        st.error(f"⚠️ Error: {e}")
