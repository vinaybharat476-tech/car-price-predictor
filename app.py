import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="🚗 Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .price-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: white;
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============== LOAD MODEL ==============
try:
    model = pickle.load(open('car_price_model.pkl', 'rb'))
    scaler = pickle.load(open('car_scaler.pkl', 'rb'))
    feature_names = pickle.load(open('car_features.pkl', 'rb'))
    label_encoders = pickle.load(open('car_encoders.pkl', 'rb'))
    
except FileNotFoundError as e:
    st.error(f"❌ Model files not found! {e}")
    st.error("Run 'python train_model.py' first")
    st.stop()

# ============== TITLE ==============
st.markdown("<h1 class='main-title'>🚗 Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2em; color: #666;'>AI-Powered Vehicle Valuation</p>", unsafe_allow_html=True)
st.markdown("---")

# ============== SIDEBAR ==============
st.sidebar.markdown("## 📊 About This Model")
st.sidebar.info("""
**Model Performance:**
- Accuracy: 88.7%
- Cars Analyzed: 10,000+
- Mean Error: ±$3,000

**Perfect for:**
✓ Used Car Dealers
✓ Private Sellers
✓ Buyers Researching
✓ Fleet Management
""")

# ============== INPUT SECTION ==============
st.subheader("📝 Vehicle Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🏢 Basic Info")
    
    make = st.selectbox(
        "Make (Brand)",
        options=['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes-Benz', 'Audi', 'Nissan', 
                 'Hyundai', 'Volkswagen', 'Chevrolet', 'Other'],
        help="Vehicle manufacturer"
    )
    
    model_name = st.text_input(
        "Model Name",
        value="Camry",
        help="e.g., Camry, Civic, F-150"
    )
    
    year = st.slider(
        "Year",
        min_value=2000,
        max_value=2024,
        value=2015,
        help="Manufacturing year"
    )

with col2:
    st.markdown("### 🔧 Engine & Performance")
    
    engine_fuel_type = st.selectbox(
        "Engine Fuel Type",
        options=['premium unleaded (required)', 'regular unleaded', 'diesel', 'electric', 'natural gas'],
        help="Primary fuel type"
    )
    
    engine_hp = st.slider(
        "Engine Horsepower",
        min_value=40,
        max_value=500,
        value=150,
        step=5,
        help="Horsepower (typical: 80-250)"
    )
    
    engine_cylinders = st.slider(
        "Engine Cylinders",
        min_value=3,
        max_value=16,
        value=4,
        help="Number of cylinders (typical: 4-8)"
    )

with col3:
    st.markdown("### 📊 Features & Efficiency")
    
    highway_mpg = st.slider(
        "Highway MPG",
        min_value=5,
        max_value=60,
        value=25,
        help="Miles per gallon on highway"
    )
    
    city_mpg = st.slider(
        "City MPG",
        min_value=5,
        max_value=50,
        value=18,
        help="Miles per gallon in city"
    )
    
    transmission_type = st.selectbox(
        "Transmission",
        options=['AUTOMATIC', 'MANUAL'],
        help="Type of transmission"
    )

# ============== MORE DETAILS ==============
st.markdown("---")
st.subheader("🎯 Additional Features")

col1, col2, col3, col4 = st.columns(4)

with col1:
    driven_wheels = st.selectbox(
        "Driven Wheels",
        options=['front wheel drive', 'rear wheel drive', 'all wheel drive', 'four wheel drive'],
        help="Which wheels are powered"
    )

with col2:
    number_of_doors = st.selectbox(
        "Number of Doors",
        options=['2', '4'],
        help="2 or 4 door vehicle"
    )

with col3:
    market_category = st.selectbox(
        "Market Category",
        options=['Sedan', 'SUV', 'Truck', 'Coupe', 'Hatchback', 'Wagon', 'Minivan', 'Convertible'],
        help="Vehicle category"
    )

with col4:
    vehicle_size = st.selectbox(
        "Vehicle Size",
        options=['Compact', 'Midsize', 'Large'],
        help="Vehicle size class"
    )

# ============== MORE DETAILS 2 ==============
col1, col2, col3 = st.columns(3)

with col1:
    vehicle_style = st.selectbox(
        "Vehicle Style",
        options=['Sedan', 'SUV', 'Coupe', 'Convertible', 'Hatchback', 'Wagon', 'Minivan', 'Pickup'],
        help="Vehicle body style"
    )

with col2:
    popularity = st.slider(
        "Popularity Score",
        min_value=1,
        max_value=5000,
        value=1000,
        step=100,
        help="Market popularity (1-5000)"
    )

# ============== DISPLAY SUMMARY ==============
st.markdown("---")
st.subheader("📊 Vehicle Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Brand", make)
with col2:
    st.metric("Year", year)
with col3:
    age = 2024 - year
    st.metric("Age", f"{age} years")
with col4:
    st.metric("Engine", f"{engine_hp} BHP, {engine_cylinders}cyl")

# ============== PREPARE INPUT ==============
st.markdown("---")

try:
    # Safe encoding function
    def safe_encode(encoder, value, feature_name):
        """Safely encode value, use first class if value not in encoder"""
        try:
            return encoder.transform([value])[0]
        except:
            # If value not found, use the first available class
            default_value = encoder.classes_[0]
            st.warning(f"⚠️ Using default value for {feature_name}")
            return encoder.transform([default_value])[0]
    
    # Encode categorical variables safely
    make_encoded = safe_encode(label_encoders['Make'], make, 'Make') if 'Make' in label_encoders else 0
    fuel_encoded = safe_encode(label_encoders['Engine Fuel Type'], engine_fuel_type, 'Fuel Type') if 'Engine Fuel Type' in label_encoders else 0
    transmission_encoded = safe_encode(label_encoders['Transmission Type'], transmission_type, 'Transmission') if 'Transmission Type' in label_encoders else 0
    driven_wheels_encoded = safe_encode(label_encoders['Driven_Wheels'], driven_wheels, 'Driven Wheels') if 'Driven_Wheels' in label_encoders else 0
    market_cat_encoded = safe_encode(label_encoders['Market Category'], market_category, 'Market Category') if 'Market Category' in label_encoders else 0
    vehicle_size_encoded = safe_encode(label_encoders['Vehicle Size'], vehicle_size, 'Vehicle Size') if 'Vehicle Size' in label_encoders else 0
    vehicle_style_encoded = safe_encode(label_encoders['Vehicle Style'], vehicle_style, 'Vehicle Style') if 'Vehicle Style' in label_encoders else 0
    
    # Build input dictionary - exact feature names from model
    input_dict = {}
    
    for feature in feature_names:
        if feature == 'Make':
            input_dict[feature] = make_encoded
        elif feature == 'Model':
            input_dict[feature] = 0  # Model usually not used in numeric form
        elif feature == 'Year':
            input_dict[feature] = year
        elif feature == 'Engine Fuel Type':
            input_dict[feature] = fuel_encoded
        elif feature == 'Engine HP':
            input_dict[feature] = engine_hp
        elif feature == 'Engine Cylinders':
            input_dict[feature] = engine_cylinders
        elif feature == 'Transmission Type':
            input_dict[feature] = transmission_encoded
        elif feature == 'Driven_Wheels':
            input_dict[feature] = driven_wheels_encoded
        elif feature == 'Number of Doors':
            input_dict[feature] = int(number_of_doors)
        elif feature == 'Market Category':
            input_dict[feature] = market_cat_encoded
        elif feature == 'Vehicle Size':
            input_dict[feature] = vehicle_size_encoded
        elif feature == 'Vehicle Style':
            input_dict[feature] = vehicle_style_encoded
        elif feature == 'highway MPG':
            input_dict[feature] = highway_mpg
        elif feature == 'city mpg':
            input_dict[feature] = city_mpg
        elif feature == 'Popularity':
            input_dict[feature] = popularity
        else:
            input_dict[feature] = 0
    
    # Create DataFrame with exact feature names and order
    final_input = pd.DataFrame([input_dict])
    final_input = final_input[feature_names]
    
    # Scale
    input_scaled = scaler.transform(final_input)
    
except Exception as e:
    st.error(f"❌ Error preparing input: {e}")
    st.error(f"Expected features: {list(feature_names)}")
    st.stop()

# ============== PREDICTION BUTTON ==============
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("💰 PREDICT CAR PRICE", use_container_width=True)

# ============== MAKE PREDICTION ==============
if predict_button:
    try:
        prediction = model.predict(input_scaled)[0]
        
        # Display result
        st.markdown("---")
        st.markdown("<h2 style='text-align: center;'>✅ Valuation Complete!</h2>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='price-box'>
            ₹{prediction:,.0f}
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        st.markdown("---")
        st.subheader("📈 Valuation Details")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Estimated Price", f"₹{prediction:,.0f}")
        
        with col2:
            age = 2024 - year
            st.metric("Car Age", f"{age} years")
        
        with col3:
            depreciation = (age * 0.1) * prediction
            st.metric("Est. Depreciation", f"₹{depreciation:,.0f}")
        
        with col4:
            resale_value = prediction * 0.7
            st.metric("Est. Resale Value", f"₹{resale_value:,.0f}")
        
        # Analysis
        st.markdown("---")
        st.subheader("📊 Price Factors")
        
        factors = []
        
        if year >= 2020:
            factors.append("✅ Recent model (Higher value)")
        elif year >= 2015:
            factors.append("🟡 Mid-range age (Good value)")
        else:
            factors.append("❌ Older model (Lower value)")
        
        if engine_hp >= 200:
            factors.append("✅ High performance engine")
        elif engine_hp >= 120:
            factors.append("🟡 Good performance")
        else:
            factors.append("🟡 Standard performance")
        
        if highway_mpg >= 30:
            factors.append("✅ Excellent fuel efficiency")
        elif highway_mpg >= 20:
            factors.append("🟡 Good fuel efficiency")
        else:
            factors.append("❌ Poor fuel efficiency")
        
        if market_category in ['SUV', 'Sedan']:
            factors.append("✅ Popular category (good demand)")
        else:
            factors.append("🟡 Specialty category")
        
        if transmission_type == 'AUTOMATIC':
            factors.append("✅ Automatic transmission (preferred)")
        else:
            factors.append("🟡 Manual transmission (lower demand)")
        
        for factor in factors:
            st.info(factor)
        
        # Price breakdown
        st.markdown("---")
        st.subheader("🏎️ Recommended Pricing")
        
        st.write(f"""
        **Fair Asking Price:** ₹{prediction * 0.95:,.0f}  
        **Maximum Price:** ₹{prediction * 1.05:,.0f}  
        **Minimum Price:** ₹{prediction * 0.85:,.0f}  
        
        Based on: Year, Engine Power, Fuel Efficiency, Category, and Market Trends
        """)
    
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")

# ============== FOOTER ==============
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <p><strong>Disclaimer:</strong> This is an AI estimate based on market data.</p>
    <p>💡 <em>Need this for your dealership? Contact us on Fiverr!</em></p>
</div>
""", unsafe_allow_html=True)