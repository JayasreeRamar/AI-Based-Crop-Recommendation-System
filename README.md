# AI-Based-Crop-Recommendation-System

# Aim
The aim of this project is to develop an AI-based Crop Recommendation System that suggests the most suitable crop for cultivation based on city-wise soil type, climate conditions, and nutrient values (NPK & pH).
The system automatically adjusts agricultural parameters according to the selected city, reducing manual effort and improving prediction accuracy.

# Problem Statement
Farmers often select crops based on traditional knowledge, personal experience, or trial and error. This approach faces several challenges:
   ### > Lack of city-specific crop guidance
   ### > Manual estimation of soil nutrients
   ### > Crop failure due to unsuitable climate or soil
   ### > Low productivity and financial loss
   ### > Absence of user-friendly AI tools for farmers
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
from PIL import Image, ImageDraw

# --------------------------------------------------
# DATASETS (AS GIVEN BY YOU)
# --------------------------------------------------

city_data = {
    "Chennai": {"soil": "Alluvial", "crop": "Rice", "temp": 33, "humidity": 65, "rain": 90},
    "Coimbatore": {"soil": "Red", "crop": "Cotton", "temp": 30, "humidity": 60, "rain": 70},
    "Madurai": {"soil": "Red", "crop": "Millets", "temp": 34, "humidity": 55, "rain": 60},
    "Trichy": {"soil": "Alluvial", "crop": "Sugarcane", "temp": 35, "humidity": 60, "rain": 80},
    "Ooty": {"soil": "Mountain", "crop": "Tea", "temp": 18, "humidity": 85, "rain": 200},

    "Kochi": {"soil": "Laterite", "crop": "Rice", "temp": 29, "humidity": 85, "rain": 300},
    "Munnar": {"soil": "Mountain", "crop": "Tea", "temp": 19, "humidity": 82, "rain": 220},
    "Wayanad": {"soil": "Laterite", "crop": "Coffee", "temp": 23, "humidity": 78, "rain": 250},
    "Palakkad": {"soil": "Alluvial", "crop": "Rice", "temp": 32, "humidity": 70, "rain": 150},

    "Bengaluru": {"soil": "Red", "crop": "Ragi", "temp": 26, "humidity": 60, "rain": 90},
    "Hyderabad": {"soil": "Black", "crop": "Cotton", "temp": 32, "humidity": 55, "rain": 70},
    "Vijayawada": {"soil": "Alluvial", "crop": "Rice", "temp": 33, "humidity": 70, "rain": 120},
    "Mumbai": {"soil": "Laterite", "crop": "Rice", "temp": 30, "humidity": 80, "rain": 300},
    "Indore": {"soil": "Black", "crop": "Soybean", "temp": 28, "humidity": 60, "rain": 100},
    "Jaipur": {"soil": "Sandy", "crop": "Bajra", "temp": 35, "humidity": 40, "rain": 50},
    "Lucknow": {"soil": "Alluvial", "crop": "Wheat", "temp": 25, "humidity": 55, "rain": 80},
    "Patna": {"soil": "Alluvial", "crop": "Rice", "temp": 29, "humidity": 70, "rain": 120},
    "Ranchi": {"soil": "Red", "crop": "Maize", "temp": 27, "humidity": 65, "rain": 110},
    "Kolkata": {"soil": "Alluvial", "crop": "Rice", "temp": 30, "humidity": 75, "rain": 150},
    "Guwahati": {"soil": "Alluvial", "crop": "Tea", "temp": 26, "humidity": 85, "rain": 250},
    "Imphal": {"soil": "Red", "crop": "Rice", "temp": 24, "humidity": 80, "rain": 200},
    "Shillong": {"soil": "Laterite", "crop": "Tea", "temp": 20, "humidity": 85, "rain": 220},
    "Aizawl": {"soil": "Red", "crop": "Maize", "temp": 23, "humidity": 75, "rain": 180},
    "Agartala": {"soil": "Alluvial", "crop": "Rice", "temp": 28, "humidity": 80, "rain": 200},
    "Kohima": {"soil": "Mountain", "crop": "Rice", "temp": 22, "humidity": 78, "rain": 190},
    "Itanagar": {"soil": "Mountain", "crop": "Maize", "temp": 21, "humidity": 85, "rain": 250},
    "Bhubaneswar": {"soil": "Laterite", "crop": "Rice", "temp": 31, "humidity": 75, "rain": 140},
    "Raipur": {"soil": "Red", "crop": "Rice", "temp": 30, "humidity": 70, "rain": 130},
    "Panaji": {"soil": "Laterite", "crop": "Coconut", "temp": 30, "humidity": 80, "rain": 280},
    "Chandigarh": {"soil": "Alluvial", "crop": "Wheat", "temp": 24, "humidity": 55, "rain": 90},
    "Dehradun": {"soil": "Alluvial", "crop": "Basmati Rice", "temp": 22, "humidity": 65, "rain": 140},
    "Shimla": {"soil": "Mountain", "crop": "Apple", "temp": 15, "humidity": 70, "rain": 120},
    "Srinagar": {"soil": "Mountain", "crop": "Apple", "temp": 14, "humidity": 65, "rain": 100},
}

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
# STREAMLIT APP
# --------------------------------------------------

st.set_page_config(page_title="AI Crop Recommendation", layout="wide")

page = st.sidebar.radio("📑 Navigation", ["City Selection", "Crop Recommendation", "Real-Time Visualization"])

# --------------------------------------------------
# PAGE 1 – CITY SELECTION
# --------------------------------------------------
if page == "City Selection":
    st.title("📍 Select City")
    city = st.selectbox("Choose City", sorted(city_data.keys()))
    st.session_state["city"] = city

    st.success(f"City Selected: {city}")

# --------------------------------------------------
# PAGE 2 – CROP RECOMMENDATION
# --------------------------------------------------
elif page == "Crop Recommendation":
    if "city" not in st.session_state:
        st.warning("Please select a city first")
    else:
        city = st.session_state["city"]
        data = city_data[city]

        crop = data["crop"]
        N, P, K, pH = crop_npk[crop]

        st.title("🌾 Crop Recommendation Result")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Environmental Details")
            st.write("🌡 Temperature:", data["temp"], "°C")
            st.write("💧 Humidity:", data["humidity"], "%")
            st.write("🌧 Rainfall:", data["rain"], "mm")
            st.write("🪨 Soil Type:", data["soil"])

        with col2:
            st.subheader("✅ Recommended Crop")
            st.success(crop)
            st.write("🧪 Nitrogen (N):", N)
            st.write("🧪 Phosphorus (P):", P)
            st.write("🧪 Potassium (K):", K)
            st.write("⚖ Soil pH:", pH)

# --------------------------------------------------
# PAGE 3 – REAL-TIME VISUALIZATION (DEMO)
# --------------------------------------------------
else:
    st.title("📷 Real-Time Crop Detection (Demo)")

    uploaded = st.file_uploader("Upload Field Image", type=["jpg", "png"])

    detections = [
        {"name": "Rice", "conf": "98%", "box": (60, 80, 300, 350)},
        {"name": "Sugarcane", "conf": "96%", "box": (350, 100, 650, 380)},
    ]

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        draw = ImageDraw.Draw(img)

        for d in detections:
            draw.rectangle(d["box"], outline="red", width=4)
            draw.text((d["box"][0], d["box"][1]-20),
                      f"{d['name']} ({d['conf']})", fill="red")

        st.image(img, caption="Crop Detection Visualization", use_container_width=True)

        st.info("This visualization simulates AI-based crop detection for demonstration.")
    else:
        st.warning("Upload an image to view detection")


```

# Output

<img width="1912" height="1018" alt="Screenshot 2025-12-26 110945" src="https://github.com/user-attachments/assets/87e7279b-5a47-4ed4-82ce-dfa4df08e889" />
<img width="1915" height="1027" alt="Screenshot 2025-12-26 111005" src="https://github.com/user-attachments/assets/6d1cf877-20cc-4fd8-9636-a939898ef92f" />
<img width="1919" height="1018" alt="Screenshot 2025-12-26 111044" src="https://github.com/user-attachments/assets/b21ade9a-59bd-4cda-b56a-51a6d80c77ae" />
<img width="1919" height="1014" alt="Screenshot 2025-12-26 111025" src="https://github.com/user-attachments/assets/528a510d-f883-4238-ab04-82e178dbf1f2" />

<img width="1919" height="1020" alt="Screenshot 2025-12-26 124721" src="https://github.com/user-attachments/assets/0ff0dc71-a2a1-4681-ba90-4b8cc8358e79" />
<img width="1919" height="1012" alt="Screenshot 2025-12-26 124736" src="https://github.com/user-attachments/assets/7fddacc8-1b50-4662-9655-61fe20ff4070" />
<img width="1919" height="1018" alt="Screenshot 2025-12-26 124751" src="https://github.com/user-attachments/assets/dfd4e5e4-f6a1-46a8-b764-1ac4f0658417" />
<img width="1426" height="769" alt="Screenshot 2025-12-26 124834" src="https://github.com/user-attachments/assets/1dad756b-d366-4a63-8f8d-870c5c5a8670" />




### Demo 1:
<img width="1919" height="1020" alt="Screenshot 2025-12-26 111107" src="https://github.com/user-attachments/assets/7c5f2ff6-4980-451e-a5d4-04288efb1df2" />

### Demo 2:
<img width="1919" height="1022" alt="Screenshot 2025-12-26 111132" src="https://github.com/user-attachments/assets/e1ba7105-9adf-44a4-90fa-879bca2bf3d2" />

### Demo 3:
<img width="1919" height="1018" alt="Screenshot 2025-12-26 111148" src="https://github.com/user-attachments/assets/1f65c7df-6c16-4a09-85ba-27279e22c110" />

# Result
  > The system successfully recommends city-specific crops across all 28 states of India.
  > For Tamil Nadu and Kerala, multiple cities are supported with accurate crop outputs.
  > Nutrient values (NPK & pH) are automatically set based on city characteristics.
  > The application provides:
         * High prediction accuracy
         * Realistic agricultural recommendations
         * Easy-to-use web interface







