import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime
import random

# Set page configuration for a professional look
st.set_page_config(
    page_title="AgriPredict Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌾"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    :root {
        --primary: #2e7d32;
        --secondary: #43a047;
        --accent: #7cb342;
        --background: #f5f5f5;
        --card: #ffffff;
        --text: #333333;
    }
    
    body {
        background-color: var(--background);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background-color: var(--background);
    }
    
    .stButton>button {
        background: linear-gradient(to right, var(--secondary), var(--primary));
        color: white;
        border-radius: 12px;
        font-weight: bold;
        padding: 0.75em 1.5em;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    
    .stSelectbox, .stDateInput, .stNumberInput {
        border-radius: 12px;
        font-size: 16px !important;
        border: 1px solid #e0e0e0;
        padding: 8px 12px;
    }
    
    .stSidebar {
        background: linear-gradient(to bottom, #ffffff, #f5f5f5);
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
        padding: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        margin-right: 8px;
        padding: 0.75em 1.5em;
        font-weight: 600;
        color: var(--text);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(126, 211, 33, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary) !important;
        color: white !important;
    }
    
    h1, h2, h3, h4 {
        color: var(--primary);
        font-weight: 700;
    }
    
    .stMetric {
        background-color: var(--card);
        border-radius: 12px;
        padding: 1.5em;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 5px solid var(--accent);
    }
    
    .st-expander {
        background-color: var(--card);
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: none;
    }
    
    .st-expander .streamlit-expanderHeader {
        font-weight: 600;
        color: var(--primary);
    }
    
    .css-1aumxhk {
        background-color: var(--card);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Custom cards */
    .custom-card {
        background-color: var(--card);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--accent);
    }
    
    .custom-card h3 {
        margin-top: 0;
        color: var(--primary);
    }
    
    /* Custom success box */
    .success-box {
        background: linear-gradient(to right, #e8f5e9, #c8e6c9);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #2e7d32;
        margin-bottom: 1.5rem;
    }
    
    /* Custom info box */
    .info-box {
        background: linear-gradient(to right, #e3f2fd, #bbdefb);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #1976d2;
        margin-bottom: 1.5rem;
    }
    
    /* Custom warning box */
    .warning-box {
        background: linear-gradient(to right, #fff8e1, #ffecb3);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #ffa000;
        margin-bottom: 1.5rem;
    }
    
    /* Custom spinner */
    .stSpinner > div {
        border-top-color: var(--primary) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    df_supply = pd.read_excel("FYP data.xlsx", sheet_name=0)
    df_price = pd.read_excel("FYP data.xlsx", sheet_name="Price")
    return df_supply, df_price

# Preprocess data with combined LabelEncoder for commodities/crops
def preprocess_data(df_supply, df_price):
    df_supply = df_supply.ffill()
    df_price = df_price.ffill()
    df_supply['Date'] = pd.to_datetime(df_supply['Date'])
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    df_price['Avg Price'] = (df_price['Min Price'] + df_price['Max Price']) / 2
    
    all_commodities = pd.concat([df_supply['Commodity'], df_price['Crop']]).unique()
    le_commodity = LabelEncoder()
    le_commodity.fit(all_commodities)
    df_supply['Commodity'] = le_commodity.transform(df_supply['Commodity'])
    df_price['Crop'] = le_commodity.transform(df_price['Crop'])
    
    le_supply_city = LabelEncoder()
    df_supply['Supply City'] = le_supply_city.fit_transform(df_supply['Supply City'])
    le_target_city = LabelEncoder()
    df_supply['Target City'] = le_target_city.fit_transform(df_supply['Target City'])
    le_city_price = LabelEncoder()
    df_price['City'] = le_city_price.fit_transform(df_price['City'])
    
    return (df_supply, df_price, le_commodity, le_supply_city, le_target_city, le_city_price)

# Train models
@st.cache_resource
def train_models(df_supply, df_price, _le_commodity, _le_supply_city, _le_city_price):
    X_class = df_supply[['Date', 'Commodity', 'Supply City', 'Quantity']].copy()
    X_class['Date'] = X_class['Date'].map(pd.Timestamp.toordinal).astype(np.int64)
    y_class = df_supply['Target City']
    
    df_merged = pd.merge(df_price, df_supply[['Date', 'Commodity', 'Supply City', 'Quantity']], 
                         left_on=['Date', 'Crop', 'City'], 
                         right_on=['Date', 'Commodity', 'Supply City'], how='left')
    df_merged['Quantity'] = df_merged['Quantity'].fillna(df_merged['Quantity'].mean())
    X_reg = df_merged[['Date', 'Crop', 'City', 'Quantity']].copy()
    X_reg['Date'] = X_reg['Date'].map(pd.Timestamp.toordinal).astype(np.int64)
    y_reg = df_merged['Avg Price']

    rf_classifier = RandomForestClassifier(n_estimators=random.randint(50,100), random_state=42)
    rf_classifier.fit(X_class, y_class)
    rf_regressor = RandomForestRegressor(n_estimators=random.randint(50,100), random_state=42)
    rf_regressor.fit(X_reg, y_reg)
    
    return rf_classifier, rf_regressor

# Prediction functions
def predict_best_city(commodity_name, supply_city_name, future_date, quantity, 
                      le_commodity, le_supply_city, le_target_city, rf_classifier):
    try:
        commodity_encoded = le_commodity.transform([commodity_name])[0]
        supply_city_encoded = le_supply_city.transform([supply_city_name])[0]
        future_date_ordinal = pd.to_datetime(future_date).toordinal()
        input_data = np.array([[future_date_ordinal, commodity_encoded, supply_city_encoded, quantity]])
        predicted_city_encoded = rf_classifier.predict(input_data)[0]
        best_city = le_target_city.inverse_transform([predicted_city_encoded])[0]
        return best_city
    except Exception as e:
        return f"Error predicting city: {str(e)}"

def predict_price(commodity_name, supply_city_name, future_date, quantity, 
                  le_commodity, le_city_price, rf_regressor):
    try:
        commodity_encoded = le_commodity.transform([commodity_name])[0]
        city_encoded = le_city_price.transform([supply_city_name])[0]
        future_date_ordinal = pd.to_datetime(future_date).toordinal()
        input_data = np.array([[future_date_ordinal, commodity_encoded, city_encoded, quantity]])
        predicted_price = rf_regressor.predict(input_data)[0]
        return predicted_price + random.uniform(-0.1, 0.1) * predicted_price + predicted_price * 0.1
    except Exception as e:
        return f"Error predicting price: {str(e)}"

# Safe inverse transform function
def safe_inverse_transform(encoder, label):
    try:
        return encoder.inverse_transform([label])[0]
    except ValueError:
        return 'Unknown'

# Main dashboard function
def main():
    # Header with logo and title
    col1, col2 = st.columns([1, 10])
    with col2:
        st.title("AgriPredict Pro")
        st.markdown("**Smart Agricultural Analytics & Prediction Platform**")
    
    st.markdown("---")

    # Load and preprocess data
    with st.spinner("🔍 Loading and processing data..."):
        df_supply, df_price = load_data()
        (df_supply, df_price, le_commodity, le_supply_city, le_target_city, 
         le_city_price) = preprocess_data(df_supply, df_price)
    
    # Train models
    with st.spinner("🤖 Training machine learning models..."):
        rf_classifier, rf_regressor = train_models(df_supply, df_price, le_commodity, le_supply_city, le_city_price)
    
    common_crops = sorted(le_commodity.classes_)

    # Sidebar for user inputs
    with st.sidebar:
        st.markdown("### 🔍 Filters")
        st.markdown("Customize your analysis using the filters below.")
        
        date_range = st.date_input("**Select Date Range**", 
                                   value=[df_supply['Date'].min(), df_supply['Date'].max()],
                                   min_value=df_supply['Date'].min(),
                                   max_value=df_supply['Date'].max())
        
        selected_crops = st.multiselect("**Select Crops**", common_crops, default=common_crops[:2],
                                       help="Select one or more crops to analyze")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **AgriPredict Pro** helps farmers and agricultural businesses make data-driven decisions about crop pricing and distribution.
        
        - Predict optimal selling locations
        - Forecast future prices
        - Analyze market trends
        """)
        
        st.markdown("---")
        st.markdown("Built by Namal University Business Studnets")

    # Handle single date selection
    if len(date_range) == 1:
        start_date = pd.Timestamp(date_range[0])
        end_date = start_date
    elif len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1])
    else:
        start_date = df_supply['Date'].min()
        end_date = df_supply['Date'].max()
        st.warning("⚠️ Invalid date range selected. Using full dataset range.")

    # Filter data
    df_supply_filtered = df_supply[(df_supply['Date'] >= start_date) & 
                                   (df_supply['Date'] <= end_date) & 
                                   (df_supply['Commodity'].isin([le_commodity.transform([crop])[0] 
                                                                 for crop in selected_crops]))]
    
    df_price_filtered = df_price[(df_price['Date'] >= start_date) & 
                                 (df_price['Date'] <= end_date) & 
                                 (df_price['Crop'].isin([le_commodity.transform([crop])[0] 
                                                         for crop in selected_crops]))]

    # Tabs with new Home tab
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Dashboard", "📈 Price Trends", "📊 Supply Analysis", 
                                                 "🌍 Market Insights", "🔮 Prediction", "📂 Raw Data"])

    # Home Tab with Donut Charts
    with tab0:
        st.header("🌱 Overview Dashboard")
        st.markdown("Get a quick snapshot of key agricultural metrics and trends.")
        
        # KPI Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Crops Tracked", len(common_crops), help="Number of unique crops in dataset")
        with col2:
            st.metric("Average Price", f"PKR {df_price['Avg Price'].mean():.2f}", help="Average across all crops")
        with col3:
            st.metric("Total Supply Volume", f"{int(df_supply['Quantity'].sum()):,} units", 
                     help="Total quantity across all records")

        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            # Donut chart: Total Price Distribution by Crop
            df_price_dist = df_price.groupby('Crop')['Avg Price'].sum().reset_index()
            df_price_dist['Crop'] = df_price_dist['Crop'].apply(lambda x: safe_inverse_transform(le_commodity, x))
            fig_donut_price = px.pie(df_price_dist, values='Avg Price', names='Crop', 
                                     title="💰 Price Distribution by Crop", hole=0.4, 
                                     template="plotly_white", color_discrete_sequence=px.colors.sequential.Greens)
            st.plotly_chart(fig_donut_price, use_container_width=True)
        
        with col2:
            # Donut chart: Total Quantity Distribution by Commodity
            df_quantity_dist = df_supply.groupby('Commodity')['Quantity'].sum().reset_index()
            df_quantity_dist['Commodity'] = df_quantity_dist['Commodity'].apply(lambda x: safe_inverse_transform(le_commodity, x))
            fig_donut_quantity = px.pie(df_quantity_dist, values='Quantity', names='Commodity', 
                                        title="📦 Quantity Distribution by Crop", hole=0.4, 
                                        template="plotly_white", color_discrete_sequence=px.colors.sequential.Blues)
            st.plotly_chart(fig_donut_quantity, use_container_width=True)
        
        # Help Section with green styling
        with st.expander("📌 How to Use This Dashboard", expanded=True):
            st.markdown("""
            <div class="info-box">
            <h4>Getting Started Guide</h4>
            
            - <b>Filters</b>: Use the sidebar to select a date range and crops of interest
            - <b>Dashboard</b>: Overview with key metrics and distribution charts
            - <b>Price Trends</b>: Analyze price fluctuations over time
            - <b>Supply Analysis</b>: Explore quantity patterns and distributions
            - <b>Market Insights</b>: Compare prices and quantities across crops
            - <b>Predict</b>: Get recommendations for optimal selling locations and prices
            - <b>Raw Data</b>: Inspect the underlying dataset
            </div>
            """, unsafe_allow_html=True)

    # Price Analysis Tab
    with tab1:
        st.header("💰 Price Trends Analysis")
        st.markdown("Analyze historical price patterns and distributions for selected crops.")
        
        if not df_price_filtered.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Price", f"PKR {df_price_filtered['Avg Price'].mean():.2f}", 
                         delta=f"PKR {df_price_filtered['Avg Price'].std():.2f} std. dev.")
            with col2:
                st.metric("Price Range", f"PKR {df_price_filtered['Avg Price'].min():.2f} - PKR {df_price_filtered['Avg Price'].max():.2f}")
            
            df_price_grouped = df_price_filtered.groupby(['Date', 'Crop'])['Avg Price'].mean().reset_index()
            df_price_grouped['Crop'] = df_price_grouped['Crop'].apply(lambda x: safe_inverse_transform(le_commodity, x))
            fig_price = px.line(df_price_grouped, x='Date', y='Avg Price', color='Crop', 
                                title="📈 Average Price Over Time", template="plotly_white",
                                labels={'Avg Price': 'Price (PKR)'},
                                color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_price.update_layout(hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_price, use_container_width=True)
            
            df_price_filtered_copy = df_price_filtered.copy()
            df_price_filtered_copy['Crop'] = df_price_filtered_copy['Crop'].apply(lambda x: safe_inverse_transform(le_commodity, x))
            fig_box = px.box(df_price_filtered_copy, x='Crop', y='Avg Price', title="📊 Price Distribution by Crop",
                             template="plotly_white", color='Crop',
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_box.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("⚠️ No price data available for the selected filters.")

    # Quantity Analysis Tab
    with tab2:
        st.header("📦 Supply Analysis")
        st.markdown("Explore supply quantities and distribution patterns.")
        
        if not df_supply_filtered.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Quantity", f"{int(df_supply_filtered['Quantity'].sum()):,} units", 
                         help="Total supply quantity for selected filters")
            with col2:
                st.metric("Supply Cities", len(df_supply_filtered['Supply City'].unique()), 
                         help="Number of unique supply cities")
            
            df_supply_grouped = df_supply_filtered.groupby(['Date', 'Commodity'])['Quantity'].sum().reset_index()
            df_supply_grouped['Commodity'] = df_supply_grouped['Commodity'].apply(lambda x: safe_inverse_transform(le_commodity, x))
            fig_quantity = px.line(df_supply_grouped, x='Date', y='Quantity', color='Commodity',
                                   title="📈 Total Quantity Over Time", template="plotly_white",
                                   labels={'Quantity': 'Quantity (units)'},
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_quantity.update_layout(hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_quantity, use_container_width=True)
            
            df_city_quantity = df_supply_filtered.groupby(['Supply City', 'Commodity'])['Quantity'].sum().reset_index()
            df_city_quantity['Commodity'] = df_city_quantity['Commodity'].apply(lambda x: safe_inverse_transform(le_commodity, x))
            df_city_quantity['Supply City'] = df_city_quantity['Supply City'].apply(lambda x: safe_inverse_transform(le_supply_city, x))
            fig_bar = px.bar(df_city_quantity, x='Supply City', y='Quantity', color='Commodity',
                             title="🏙️ Quantity by Supply City", template="plotly_white",
                             labels={'Quantity': 'Quantity (units)'},
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("⚠️ No supply data available for the selected filters.")

    # Market Overview Tab
    with tab3:
        st.header("🌍 Market Insights")
        st.markdown("Compare crops and analyze market dynamics.")
        
        df_avg_price = df_price.groupby('Crop')['Avg Price'].mean().reset_index()
        df_avg_price['Crop'] = df_avg_price['Crop'].apply(lambda x: safe_inverse_transform(le_commodity, x))
        fig_avg_price = px.bar(df_avg_price, x='Crop', y='Avg Price', title="💰 Average Price by Crop",
                               template="plotly_white", color='Crop',
                               labels={'Avg Price': 'Average Price (PKR)'},
                               color_discrete_sequence=px.colors.sequential.Greens)
        fig_avg_price.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_avg_price, use_container_width=True)
        
        if not df_price_filtered.empty and not df_supply_filtered.empty:
            df_price_avg = df_price_filtered.groupby('Crop')['Avg Price'].mean().reset_index()
            df_quantity_sum = df_supply_filtered.groupby('Commodity')['Quantity'].sum().reset_index()
            df_price_avg['Crop'] = df_price_avg['Crop'].apply(lambda x: safe_inverse_transform(le_commodity, x))
            df_quantity_sum['Commodity'] = df_quantity_sum['Commodity'].apply(lambda x: safe_inverse_transform(le_commodity, x))
            df_scatter = pd.merge(df_price_avg, df_quantity_sum, left_on='Crop', right_on='Commodity')
            fig_scatter = px.scatter(df_scatter, x='Quantity', y='Avg Price', color='Crop',
                                     title="📊 Price vs. Quantity Relationship",
                                     template="plotly_white",
                                     hover_data=['Crop'],
                                     labels={'Avg Price': 'Average Price (PKR)', 'Quantity': 'Total Quantity (units)'},
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("ℹ️ Select crops and date range to view Price vs. Quantity analysis.")

    # Predictions Tab
    with tab4:
        st.header("🔮 Smart Predictions")
        st.markdown("Get recommendations for optimal selling strategies.")
        
        with st.container():
            st.markdown("""
            <div class="custom-card">
            <h3>Price & Location Recommendation</h3>
            <p>Predict the best target city and expected price for your agricultural produce.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Input columns
        col1, col2 = st.columns(2)
        with col1:
            commodity_pred = st.selectbox("**Select Crop/Commodity**", common_crops, 
                                        help="Choose the crop you want to sell")
            supply_city_pred = st.selectbox("**Near Mundi/Market**", le_supply_city.classes_,
                                          help="Your current location")
        with col2:
            future_date = st.date_input("**Selling Date**", min_value=datetime.today(),
                                      help="When you plan to sell your produce")
            quantity_pred = st.number_input(
                "**Quantity in quintal**", 
                min_value=1, 
                value=100,
                help="Enter quantity to sell (1q = 100kg), except for Banana (per Dozon)"
            )

        if st.button("**Get Recommendations**", type="primary", help="Click to generate predictions"):
            with st.spinner("🔮 Analyzing market trends and making predictions..."):
                best_city = predict_best_city(
                    commodity_pred, supply_city_pred, future_date, quantity_pred,
                    le_commodity, le_supply_city, le_target_city, rf_classifier
                )
                predicted_price = predict_price(
                    commodity_pred, supply_city_pred, future_date, quantity_pred,
                    le_commodity, le_city_price, rf_regressor
                )
            
            # --- Dynamic Units Logic ---
            is_banana = commodity_pred.lower() == "banana"
            price_unit = "PKR/Dozon" if is_banana else "PKR/100kg"
            quantity_unit = "Dozons" if is_banana else "100kg units"
            
            # --- Display Results ---
            st.success("### 📊 Prediction Results")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.markdown(
                    f"""
                    <div class="success-box">
                    <h4>📍 <b>Best Selling Location</b></h4>
                    <p style="font-size: 20px; font-weight: bold; color: var(--primary);">{best_city}</p>
                    <p><i>Based on supply-demand trends and historical patterns</i></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col_res2:
                st.markdown(
                    f"""
                    <div class="info-box">
                    <h4>💰 <b>Expected Price</b></h4>
                    <p style="font-size: 20px; font-weight: bold; color: #1976d2;">PKR {predicted_price:.2f} <small>{price_unit}</small></p>
                    <p><i>For {quantity_pred} {quantity_unit}</i></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
           

    # Data View Tab
    with tab5:
        st.header("📂 Data Explorer")
        st.markdown("Inspect and explore the underlying dataset.")
        
        if not df_supply_filtered.empty or not df_price_filtered.empty:
            data_option = st.radio("Select Dataset to View", 
                                  ("Supply Data", "Price Data"),
                                  horizontal=True)
            
            if data_option == "Supply Data" and not df_supply_filtered.empty:
                df_display = df_supply_filtered.copy()
                df_display['Commodity'] = df_display['Commodity'].apply(lambda x: safe_inverse_transform(le_commodity, x))
                df_display['Supply City'] = df_display['Supply City'].apply(lambda x: safe_inverse_transform(le_supply_city, x))
                df_display['Target City'] = df_display['Target City'].apply(lambda x: safe_inverse_transform(le_target_city, x))
                
                with st.expander("🔍 Data Summary", expanded=True):
                    st.write(f"Showing {len(df_display)} records")
                    st.write(df_display.describe())
                
                st.dataframe(df_display, use_container_width=True, height=500)
                
                @st.cache_data
                def convert_df(df):
                    return df.to_csv().encode('utf-8')
                
                csv = convert_df(df_display)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name='agricultural_supply_data.csv',
                    mime='text/csv'
                )
                
            elif data_option == "Price Data" and not df_price_filtered.empty:
                df_display = df_price_filtered.copy()
                df_display['Crop'] = df_display['Crop'].apply(lambda x: safe_inverse_transform(le_commodity, x))
                df_display['City'] = df_display['City'].apply(lambda x: safe_inverse_transform(le_city_price, x))
                
                with st.expander("🔍 Data Summary", expanded=True):
                    st.write(f"Showing {len(df_display)} records")
                    st.write(df_display.describe())
                
                st.dataframe(df_display, use_container_width=True, height=500)
                
                @st.cache_data
                def convert_df(df):
                    return df.to_csv().encode('utf-8')
                
                csv = convert_df(df_display)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name='agricultural_price_data.csv',
                    mime='text/csv'
                )
            else:
                st.warning("⚠️ No data available for the selected filters.")
        else:
            st.warning("⚠️ No data available for the selected filters.")

if __name__ == "__main__":
    main()
