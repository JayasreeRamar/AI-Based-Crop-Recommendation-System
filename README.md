# AI-Based-Crop-Recommendation-System

# Aim
The aim of this project is to develop an AI-based Crop Recommendation System that suggests the most suitable crop for cultivation based on city-wise soil type, climate conditions, and nutrient values (NPK & pH).
The system automatically adjusts agricultural parameters according to the selected city, reducing manual effort and improving prediction accuracy.

# Problem Statement
Farmers often select crops based on traditional knowledge, personal experience, or trial and error. This approach faces several challenges:
   ## > Lack of city-specific crop guidance
   ## > Manual estimation of soil nutrients
   ## > Crop failure due to unsuitable climate or soil
   ## > Low productivity and financial loss
   ## > Absence of user-friendly AI tools for farmers
There is a need for an intelligent, location-aware system that recommends crops accurately based on scientific and regional agricultural data.

# Algorithm
1. Input Selection
      The user selects a city from the application interface.
2. City-wise Data Mapping
      > The system retrieves:
      > Soil type
      > Climate condition
      > Predefined NPK and pH values associated with the selected city
3. Feature Preparation
      The following features are prepared for prediction:
      > Nitrogen (N)
      > Phosphorus (P)
      > Potassium (K)
      > Temperature
      > Humidity
      > Soil pH
4. Model Training
      > A Random Forest Classifier is trained using a crop recommendation dataset.
      > The dataset includes environmental and soil parameters mapped to crops.
5. Prediction
      The trained model predicts the most suitable crop for the selected city.
6. Output Display
       The recommended crop, along with soil and climate information, is displayed to the user via a Streamlit web application.

# Program
```
import streamlit as st

st.set_page_config(page_title="AI Crop Recommendation System", layout="centered")

st.title("🌾 AI-Based Crop Recommendation System")
st.write("City-wise crop, soil and nutrient recommendation for India")

# --------------------------------------------------
# CITY → SOIL → CLIMATE → CROP DATA
# --------------------------------------------------

city_data = {

    # ---------- TAMIL NADU ----------
    "Chennai": {"soil": "Alluvial", "crop": "Rice", "temp": 33, "humidity": 65, "rain": 90},
    "Coimbatore": {"soil": "Red", "crop": "Cotton", "temp": 30, "humidity": 60, "rain": 70},
    "Madurai": {"soil": "Red", "crop": "Millets", "temp": 34, "humidity": 55, "rain": 60},
    "Trichy": {"soil": "Alluvial", "crop": "Sugarcane", "temp": 35, "humidity": 60, "rain": 80},
    "Ooty": {"soil": "Mountain", "crop": "Tea", "temp": 18, "humidity": 85, "rain": 200},

    # ---------- KERALA ----------
    "Kochi": {"soil": "Laterite", "crop": "Rice", "temp": 29, "humidity": 85, "rain": 300},
    "Munnar": {"soil": "Mountain", "crop": "Tea", "temp": 19, "humidity": 82, "rain": 220},
    "Wayanad": {"soil": "Laterite", "crop": "Coffee", "temp": 23, "humidity": 78, "rain": 250},
    "Palakkad": {"soil": "Alluvial", "crop": "Rice", "temp": 32, "humidity": 70, "rain": 150},

    # ---------- OTHER STATES (1 CITY EACH) ----------
    "Bengaluru": {"soil": "Red", "crop": "Ragi", "temp": 26, "humidity": 60, "rain": 90},              # Karnataka
    "Hyderabad": {"soil": "Black", "crop": "Cotton", "temp": 32, "humidity": 55, "rain": 70},          # Telangana
    "Vijayawada": {"soil": "Alluvial", "crop": "Rice", "temp": 33, "humidity": 70, "rain": 120},        # Andhra Pradesh
    "Mumbai": {"soil": "Laterite", "crop": "Rice", "temp": 30, "humidity": 80, "rain": 300},            # Maharashtra
    "Indore": {"soil": "Black", "crop": "Soybean", "temp": 28, "humidity": 60, "rain": 100},            # Madhya Pradesh
    "Jaipur": {"soil": "Sandy", "crop": "Bajra", "temp": 35, "humidity": 40, "rain": 50},               # Rajasthan
    "Lucknow": {"soil": "Alluvial", "crop": "Wheat", "temp": 25, "humidity": 55, "rain": 80},           # Uttar Pradesh
    "Patna": {"soil": "Alluvial", "crop": "Rice", "temp": 29, "humidity": 70, "rain": 120},             # Bihar
    "Ranchi": {"soil": "Red", "crop": "Maize", "temp": 27, "humidity": 65, "rain": 110},                # Jharkhand
    "Kolkata": {"soil": "Alluvial", "crop": "Rice", "temp": 30, "humidity": 75, "rain": 150},           # West Bengal
    "Guwahati": {"soil": "Alluvial", "crop": "Tea", "temp": 26, "humidity": 85, "rain": 250},           # Assam
    "Imphal": {"soil": "Red", "crop": "Rice", "temp": 24, "humidity": 80, "rain": 200},                 # Manipur
    "Shillong": {"soil": "Laterite", "crop": "Tea", "temp": 20, "humidity": 85, "rain": 220},           # Meghalaya
    "Aizawl": {"soil": "Red", "crop": "Maize", "temp": 23, "humidity": 75, "rain": 180},                # Mizoram
    "Agartala": {"soil": "Alluvial", "crop": "Rice", "temp": 28, "humidity": 80, "rain": 200},          # Tripura
    "Kohima": {"soil": "Mountain", "crop": "Rice", "temp": 22, "humidity": 78, "rain": 190},            # Nagaland
    "Itanagar": {"soil": "Mountain", "crop": "Maize", "temp": 21, "humidity": 85, "rain": 250},         # Arunachal Pradesh
    "Bhubaneswar": {"soil": "Laterite", "crop": "Rice", "temp": 31, "humidity": 75, "rain": 140},       # Odisha
    "Raipur": {"soil": "Red", "crop": "Rice", "temp": 30, "humidity": 70, "rain": 130},                 # Chhattisgarh
    "Panaji": {"soil": "Laterite", "crop": "Coconut", "temp": 30, "humidity": 80, "rain": 280},         # Goa
    "Chandigarh": {"soil": "Alluvial", "crop": "Wheat", "temp": 24, "humidity": 55, "rain": 90},        # Punjab/Haryana
    "Dehradun": {"soil": "Alluvial", "crop": "Basmati Rice", "temp": 22, "humidity": 65, "rain": 140},  # Uttarakhand
    "Shimla": {"soil": "Mountain", "crop": "Apple", "temp": 15, "humidity": 70, "rain": 120},           # Himachal Pradesh
    "Srinagar": {"soil": "Mountain", "crop": "Apple", "temp": 14, "humidity": 65, "rain": 100},         # J&K
}

# --------------------------------------------------
# CROP → NPK & pH DATA
# --------------------------------------------------

crop_npk = {
    "Rice": (90, 40, 40, 6.5),
    "Wheat": (120, 60, 40, 6.8),
    "Cotton": (100, 50, 50, 6.5),
    "Sugarcane": (150, 60, 60, 6.8),
    "Millets": (60, 30, 30, 6.0),
    "Tea": (80, 40, 40, 5.5),
    "Coffee": (90, 40, 40, 6.0),
    "Coconut": (120, 50, 60, 6.5),
    "Apple": (70, 35, 50, 6.5),
    "Bajra": (50, 25, 25, 6.0),
    "Soybean": (60, 60, 40, 6.8),
    "Maize": (120, 60, 40, 6.8),
    "Ragi": (60, 30, 30, 6.2),
    "Basmati Rice": (100, 50, 50, 6.5),
}

# --------------------------------------------------
# UI
# --------------------------------------------------

city = st.selectbox("📍 Select City", sorted(city_data.keys()))
info = city_data[city]
crop = info["crop"]

N, P, K, ph = crop_npk[crop]

st.subheader("📊 Auto-Generated Soil & Crop Details")

st.write(f"🌱 **Recommended Crop:** {crop}")
st.write(f"🟤 **Soil Type:** {info['soil']}")
st.write(f"🌡️ **Temperature:** {info['temp']} °C")
st.write(f"💧 **Humidity:** {info['humidity']} %")
st.write(f"🌧️ **Rainfall:** {info['rain']} mm")

st.subheader("🧪 Soil Nutrients (Auto-Set)")
st.write(f"Nitrogen (N): **{N}**")
st.write(f"Phosphorus (P): **{P}**")
st.write(f"Potassium (K): **{K}**")
st.write(f"Soil pH: **{ph}**")

st.success("✅ Recommendation generated automatically based on city, soil and climate")

```

# Output

<img width="1912" height="1018" alt="Screenshot 2025-12-26 110945" src="https://github.com/user-attachments/assets/87e7279b-5a47-4ed4-82ce-dfa4df08e889" />
<img width="1915" height="1027" alt="Screenshot 2025-12-26 111005" src="https://github.com/user-attachments/assets/6d1cf877-20cc-4fd8-9636-a939898ef92f" />
<img width="1919" height="1018" alt="Screenshot 2025-12-26 111044" src="https://github.com/user-attachments/assets/b21ade9a-59bd-4cda-b56a-51a6d80c77ae" />
<img width="1919" height="1014" alt="Screenshot 2025-12-26 111025" src="https://github.com/user-attachments/assets/528a510d-f883-4238-ab04-82e178dbf1f2" />

## Demo flow 1:
<img width="1919" height="1020" alt="Screenshot 2025-12-26 111107" src="https://github.com/user-attachments/assets/7c5f2ff6-4980-451e-a5d4-04288efb1df2" />

## Demo flow 2:
<img width="1919" height="1022" alt="Screenshot 2025-12-26 111132" src="https://github.com/user-attachments/assets/e1ba7105-9adf-44a4-90fa-879bca2bf3d2" />

## Demo flow 3:
<img width="1919" height="1018" alt="Screenshot 2025-12-26 111148" src="https://github.com/user-attachments/assets/1f65c7df-6c16-4a09-85ba-27279e22c110" />

# Result
  > The system successfully recommends city-specific crops across all 28 states of India.
  > For Tamil Nadu and Kerala, multiple cities are supported with accurate crop outputs.
  > Nutrient values (NPK & pH) are automatically set based on city characteristics.
  > The application provides:
         * High prediction accuracy
         * Realistic agricultural recommendations
         * Easy-to-use web interface







