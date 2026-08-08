import streamlit as st

st.title("🚗 Car Price Predictor")

# ============== INPUT SECTIONS ==============
st.subheader("📝 Enter Car Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🏢 Basic Info")
    make = st.selectbox(
        "Make (Brand)",
        options=['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes-Benz', 'Audi', 'Nissan', 'Hyundai', 'Volkswagen', 'Other']
    )
    
    model_name = st.text_input(
        "Model Name",
        value="Camry",
        help="e.g., Camry, Civic, F-150"
    )
    
    year = st.slider(
        "Year",
        min_value=1990,
        max_value=2024,
        value=2015
    )

with col2:
    st.markdown("### 🔧 Engine & Performance")
    engine_fuel_type = st.selectbox(
        "Engine Fuel Type",
        options=['premium unleaded (required)', 'regular unleaded', 'diesel', 'electric', 'natural gas']
    )
    
    engine_hp = st.slider(
        "Engine Horsepower",
        min_value=50,
        max_value=800,
        value=200
    )
    
    engine_cylinders = st.slider(
        "Engine Cylinders",
        min_value=3,
        max_value=16,
        value=4
    )

with col3:
    st.markdown("### 📏 Features & Efficiency")
    highway_mpg = st.slider(
        "Highway MPG",
        min_value=5,
        max_value=60,
        value=25
    )
    
    city_mpg = st.slider(
        "City MPG",
        min_value=5,
        max_value=50,
        value=18
    )
    
    transmission_type = st.selectbox(
        "Transmission Type",
        options=['AUTOMATIC', 'MANUAL']
    )

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    driven_wheels = st.selectbox(
        "Driven Wheels",
        options=['front wheel drive', 'rear wheel drive', 'all wheel drive', 'four wheel drive']
    )
    
    number_of_doors = st.selectbox(
        "Number of Doors",
        options=['2', '4']
    )

with col2:
    market_category = st.selectbox(
        "Market Category",
        options=['Sedan', 'SUV', 'Truck', 'Coupe', 'Hatchback', 'Wagon', 'Minivan']
    )
    
    vehicle_size = st.selectbox(
        "Vehicle Size",
        options=['Compact', 'Midsize', 'Large']
    )

with col3:
    vehicle_style = st.selectbox(
        "Vehicle Style",
        options=['Sedan', 'SUV', 'Coupe', 'Convertible', 'Hatchback', 'Wagon', 'Minivan', 'Pickup']
    )
    
    popularity = st.slider(
        "Popularity Score",
        min_value=1,
        max_value=5000,
        value=1000
    )

# ============== PREPARE INPUT ==============
st.markdown("---")

input_dict = {
    'Make': make,
    'Model': model_name,
    'Year': year,
    'Engine Fuel Type': engine_fuel_type,
    'Engine HP': engine_hp,
    'Engine Cylinders': engine_cylinders,
    'Transmission Type': transmission_type,
    'Driven Wheels': driven_wheels,
    'Number of Doors': number_of_doors,
    'Market Category': market_category,
    'Vehicle Size': vehicle_size,
    'Vehicle Style': vehicle_style,
    'highway MPG': highway_mpg,
    'city mpg': city_mpg,
    'Popularity': popularity
}