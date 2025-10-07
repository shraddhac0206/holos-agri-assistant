"""
Agricultural dashboard for managing projects and running simulations.
User-friendly interface for CSM parameter input and results visualization.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import requests
import json

API_BASE = "http://127.0.0.1:8000"


def main():
    """Main dashboard interface."""
    st.set_page_config(
        page_title="Holos CSM Dashboard",
        page_icon="🌾",
        layout="wide"
    )
    
    st.title("🌾 Holos Crop Science Model Dashboard")
    st.markdown("Manage agricultural projects and run simulations")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["Project Management", "Yield Prediction", "LAI Analysis", "Variety Comparison", "Efficiency Analysis"]
    )
    
    if page == "Project Management":
        project_management_page()
    elif page == "Yield Prediction":
        yield_prediction_page()
    elif page == "LAI Analysis":
        lai_analysis_page()
    elif page == "Variety Comparison":
        variety_comparison_page()
    elif page == "Efficiency Analysis":
        efficiency_analysis_page()


def project_management_page():
    """Project management interface."""
    st.header("📋 Project Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Create New Project")
        
        with st.form("new_project"):
            project_name = st.text_input("Project Name", "Rice Field 2024")
            location = st.text_input("Location", "Punjab")
            crop_type = st.selectbox("Crop Type", ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"])
            
            col_a, col_b = st.columns(2)
            with col_a:
                planting_date = st.date_input("Planting Date")
                soil_type = st.selectbox("Soil Type", ["Loam", "Sandy", "Clay", "Sandy Loam"])
            
            with col_b:
                field_size = st.number_input("Field Size (hectares)", min_value=0.1, value=1.0)
                water_source = st.selectbox("Water Source", ["Canal", "Tube Well", "Rainfed", "Mixed"])
            
            submitted = st.form_submit_button("Create Project")
            
            if submitted:
                project_data = {
                    "name": project_name,
                    "location": location,
                    "crop_type": crop_type,
                    "planting_date": str(planting_date),
                    "soil_type": soil_type,
                    "field_size": field_size,
                    "water_source": water_source
                }
                
                st.success(f"Project '{project_name}' created successfully!")
                st.json(project_data)
    
    with col2:
        st.subheader("Active Projects")
        
        # Mock project data
        projects = [
            {"name": "Rice Field A", "crop": "Rice", "location": "Punjab", "status": "Active"},
            {"name": "Wheat Field B", "crop": "Wheat", "location": "Haryana", "status": "Planning"},
            {"name": "Maize Field C", "crop": "Maize", "location": "Maharashtra", "status": "Harvested"}
        ]
        
        for project in projects:
            with st.container():
                st.write(f"**{project['name']}**")
                st.write(f"Crop: {project['crop']}")
                st.write(f"Location: {project['location']}")
                st.write(f"Status: {project['status']}")
                st.divider()


def yield_prediction_page():
    """Yield prediction interface."""
    st.header("🌱 Yield Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Parameters")
        
        with st.form("yield_prediction"):
            crop_type = st.selectbox("Crop Type", ["rice", "wheat", "maize", "cotton"])
            
            col_a, col_b = st.columns(2)
            with col_a:
                planting_date = st.selectbox("Planting Date", ["early", "optimal", "late"])
                water_input = st.number_input("Water Input (mm)", min_value=0, value=800)
                nitrogen_input = st.number_input("Nitrogen Input (kg/ha)", min_value=0, value=100)
            
            with col_b:
                soil_quality = st.slider("Soil Quality Index", 0.0, 1.0, 0.7)
                avg_temperature = st.number_input("Average Temperature (°C)", value=25)
                pest_pressure = st.slider("Pest Pressure Index", 0.0, 1.0, 0.3)
            
            predict_button = st.form_submit_button("Predict Yield")
            
            if predict_button:
                parameters = {
                    "planting_date": planting_date,
                    "water_input_mm": water_input,
                    "nitrogen_input_kg_ha": nitrogen_input,
                    "soil_quality_index": soil_quality,
                    "avg_temperature_c": avg_temperature,
                    "pest_pressure_index": pest_pressure
                }
                
                # Call API
                response = requests.post(f"{API_BASE}/predict-yield", json={
                    "crop_type": crop_type,
                    "parameters": parameters
                })
                
                if response.ok:
                    result = response.json()
                    display_yield_results(result)
                else:
                    st.error(f"Error: {response.status_code}")


def display_yield_results(result: Dict[str, Any]):
    """Display yield prediction results."""
    st.subheader("📊 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Predicted Yield",
            f"{result['predicted_yield_kg_ha']} kg/ha",
            delta=f"±{result['uncertainty_percent']}%"
        )
    
    with col2:
        ci = result['confidence_interval']
        st.metric(
            "Confidence Interval",
            f"{ci['lower']}-{ci['upper']} kg/ha"
        )
    
    with col3:
        if 'water_efficiency' in result:
            wue = result['water_efficiency']
            st.metric(
                "Water Efficiency",
                f"{wue['efficiency_percent']}%",
                delta=wue['efficiency_rating']
            )
    
    # Display recommendations
    st.subheader("💡 Recommendations")
    for rec in result.get('recommendations', []):
        st.write(f"• {rec}")
    
    # Display efficiency analysis
    if 'water_efficiency' in result:
        display_efficiency_analysis(result['water_efficiency'], "Water")
    
    if 'nitrogen_efficiency' in result:
        display_efficiency_analysis(result['nitrogen_efficiency'], "Nitrogen")


def display_efficiency_analysis(efficiency_data: Dict[str, Any], resource_type: str):
    """Display efficiency analysis."""
    st.subheader(f"📈 {resource_type} Efficiency Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            f"{resource_type} Use Efficiency",
            f"{efficiency_data[f'{resource_type.lower()}_use_efficiency_kg_per_mm'] if resource_type == 'Water' else efficiency_data[f'{resource_type.lower()}_use_efficiency_kg_per_kg_n']}",
            delta=f"{efficiency_data['efficiency_rating']}"
        )
    
    with col2:
        st.metric(
            "Benchmark Comparison",
            f"{efficiency_data['efficiency_percent']}%",
            delta=f"vs {efficiency_data[f'benchmark_{resource_type.lower()[0]}ue']} benchmark"
        )
    
    # Efficiency recommendations
    if efficiency_data.get('recommendations'):
        st.write("**Recommendations:**")
        for rec in efficiency_data['recommendations']:
            st.write(f"• {rec}")


def lai_analysis_page():
    """LAI analysis interface."""
    st.header("🛰️ Leaf Area Index Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Analysis Parameters")
        
        region = st.selectbox("Region", ["Punjab", "Maharashtra", "Haryana", "Karnataka"])
        days_back = st.slider("Analysis Period (days)", 30, 365, 90)
        
        if st.button("Analyze LAI"):
            response = requests.post(f"{API_BASE}/analyze-lai", json={
                "region": region,
                "days_back": days_back
            })
            
            if response.ok:
                result = response.json()
                display_lai_results(result)
            else:
                st.error(f"Error: {response.status_code}")
    
    with col2:
        st.subheader("LAI Trends")
        # Mock LAI visualization
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
        lai_values = [2.1, 2.3, 2.8, 3.2, 4.1, 5.2, 5.8, 5.5, 4.8, 3.9, 3.2, 2.5, 2.1]
        
        fig = px.line(
            x=dates[:len(lai_values)],
            y=lai_values,
            title="LAI Trends Over Time",
            labels={'x': 'Date', 'y': 'LAI Value'}
        )
        st.plotly_chart(fig, use_container_width=True)


def display_lai_results(result: Dict[str, Any]):
    """Display LAI analysis results."""
    if 'error' in result:
        st.error(result['error'])
        return
    
    st.subheader("📊 LAI Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean LAI", f"{result['mean_lai']:.2f}")
    
    with col2:
        st.metric("Health Score", f"{result['health_score']}%")
    
    with col3:
        st.metric("Trend", result['trend_direction'].title())
    
    # Seasonal patterns
    if result.get('seasonal_patterns'):
        st.subheader("🌍 Seasonal Patterns")
        seasonal_df = pd.DataFrame(result['seasonal_patterns']).T
        st.dataframe(seasonal_df)
    
    # Recommendations
    if result.get('recommendations'):
        st.subheader("💡 Recommendations")
        for rec in result['recommendations']:
            st.write(f"• {rec}")


def variety_comparison_page():
    """Variety comparison interface."""
    st.header("🔬 Variety Comparison")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Comparison Parameters")
        
        with st.form("variety_comparison"):
            crop_type = st.selectbox("Crop Type", ["rice", "wheat", "maize"])
            
            # Default varieties based on crop
            default_varieties = {
                "rice": ["IR64", "Pusa Basmati", "Samba"],
                "wheat": ["HD2967", "PBW343", "DBW17"],
                "maize": ["Hybrid Maize", "Sweet Corn", "Popcorn"]
            }
            
            varieties = st.multiselect(
                "Select Varieties",
                default_varieties.get(crop_type, ["Variety 1", "Variety 2"]),
                default=default_varieties.get(crop_type, ["Variety 1", "Variety 2"])[:2]
            )
            
            # Common parameters
            water_input = st.number_input("Water Input (mm)", min_value=0, value=800)
            nitrogen_input = st.number_input("Nitrogen Input (kg/ha)", min_value=0, value=100)
            soil_quality = st.slider("Soil Quality Index", 0.0, 1.0, 0.7)
            
            compare_button = st.form_submit_button("Compare Varieties")
            
            if compare_button and len(varieties) >= 2:
                parameters = {
                    "water_input_mm": water_input,
                    "nitrogen_input_kg_ha": nitrogen_input,
                    "soil_quality_index": soil_quality
                }
                
                response = requests.post(f"{API_BASE}/compare-varieties", json={
                    "crop_type": crop_type,
                    "varieties": varieties,
                    "parameters": parameters
                })
                
                if response.ok:
                    result = response.json()
                    display_variety_comparison(result)
                else:
                    st.error(f"Error: {response.status_code}")
    
    with col2:
        st.subheader("Comparison Results")
        # This will be populated by display_variety_comparison


def display_variety_comparison(result: Dict[str, Any]):
    """Display variety comparison results."""
    st.subheader("📊 Variety Comparison Results")
    
    # Create comparison table
    comparison_data = []
    for variety, data in result['comparison_results'].items():
        comparison_data.append({
            'Variety': variety,
            'Predicted Yield (kg/ha)': data['predicted_yield_kg_ha'],
            'Uncertainty (%)': data['uncertainty_percent']
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Predicted Yield (kg/ha)', ascending=False)
    
    st.dataframe(df, use_container_width=True)
    
    # Rankings
    st.subheader("🏆 Rankings")
    st.write("**By Yield:**")
    for i, variety in enumerate(result['rankings']['by_yield'], 1):
        st.write(f"{i}. {variety}")
    
    # Visualization
    fig = px.bar(
        df,
        x='Variety',
        y='Predicted Yield (kg/ha)',
        title="Yield Comparison by Variety",
        color='Predicted Yield (kg/ha)',
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig, use_container_width=True)


def efficiency_analysis_page():
    """Efficiency analysis interface."""
    st.header("⚡ Resource Efficiency Analysis")
    
    st.subheader("Water and Nitrogen Efficiency Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Input Parameters**")
        crop_type = st.selectbox("Crop Type", ["rice", "wheat", "maize", "cotton"])
        yield_achieved = st.number_input("Yield Achieved (kg/ha)", min_value=0, value=3000)
        water_used = st.number_input("Water Used (mm)", min_value=0, value=1000)
        nitrogen_used = st.number_input("Nitrogen Used (kg/ha)", min_value=0, value=120)
    
    with col2:
        if st.button("Calculate Efficiency"):
            # Calculate water efficiency
            wue = yield_achieved / water_used if water_used > 0 else 0
            nue = yield_achieved / nitrogen_used if nitrogen_used > 0 else 0
            
            # Benchmark values
            wue_benchmarks = {"rice": 3.5, "wheat": 4.0, "maize": 4.5, "cotton": 2.0}
            nue_benchmarks = {"rice": 30, "wheat": 35, "maize": 25, "cotton": 15}
            
            wue_benchmark = wue_benchmarks.get(crop_type, 3.0)
            nue_benchmark = nue_benchmarks.get(crop_type, 25)
            
            wue_percent = (wue / wue_benchmark) * 100
            nue_percent = (nue / nue_benchmark) * 100
            
            # Display results
            st.subheader("📊 Efficiency Results")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    "Water Use Efficiency",
                    f"{wue:.2f} kg/mm",
                    delta=f"{wue_percent:.1f}% of benchmark"
                )
            
            with col_b:
                st.metric(
                    "Nitrogen Use Efficiency",
                    f"{nue:.2f} kg/kg N",
                    delta=f"{nue_percent:.1f}% of benchmark"
                )
            
            # Efficiency ratings
            wue_rating = get_efficiency_rating(wue_percent)
            nue_rating = get_efficiency_rating(nue_percent)
            
            st.write(f"**Water Efficiency Rating:** {wue_rating}")
            st.write(f"**Nitrogen Efficiency Rating:** {nue_rating}")


def get_efficiency_rating(percentage: float) -> str:
    """Get efficiency rating."""
    if percentage >= 120:
        return "🌟 Excellent"
    elif percentage >= 100:
        return "✅ Good"
    elif percentage >= 80:
        return "⚠️ Average"
    elif percentage >= 60:
        return "❌ Below Average"
    else:
        return "🚨 Poor"


if __name__ == "__main__":
    main()

