"""
Yield prediction models with water and nitrogen efficiency estimation.
Integrates crop science models with AI for enhanced predictions.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

from .config import settings


class YieldPredictor:
    def __init__(self):
        """Initialize yield prediction models."""
        self.crop_models = {}
        self.training_data = {}
        self.prediction_history = {}
        
    def load_training_data(self, crop_type: str, data_source: str = None) -> Dict[str, Any]:
        """
        Load training data for yield prediction models.
        
        Args:
            crop_type: Type of crop (rice, wheat, maize, etc.)
            data_source: Path to training data file
            
        Returns:
            Training data summary
        """
        # Mock training data - in production, this would load from CSV/database
        if crop_type.lower() == "rice":
            training_data = {
                "crop": "rice",
                "varieties": ["IR64", "Pusa Basmati", "Samba"],
                "parameters": ["planting_date", "water_input", "nitrogen_input", "soil_type", "temperature"],
                "yield_range": (2000, 6000),  # kg/hectare
                "data_points": 1500,
                "features": [
                    "planting_date_days_from_jan1",
                    "total_water_mm",
                    "total_nitrogen_kg_ha",
                    "soil_ph",
                    "avg_temperature_c",
                    "rainfall_mm",
                    "pest_pressure_index"
                ]
            }
        elif crop_type.lower() == "wheat":
            training_data = {
                "crop": "wheat",
                "varieties": ["HD2967", "PBW343", "DBW17"],
                "parameters": ["planting_date", "water_input", "nitrogen_input", "soil_type", "temperature"],
                "yield_range": (2500, 5500),
                "data_points": 1200,
                "features": [
                    "planting_date_days_from_oct1",
                    "total_water_mm",
                    "total_nitrogen_kg_ha",
                    "soil_organic_matter",
                    "avg_temperature_c",
                    "frost_days",
                    "disease_pressure_index"
                ]
            }
        else:
            # Generic crop model
            training_data = {
                "crop": crop_type,
                "varieties": ["Generic"],
                "parameters": ["planting_date", "water_input", "nitrogen_input"],
                "yield_range": (1500, 4000),
                "data_points": 500,
                "features": [
                    "planting_date_encoded",
                    "total_water_mm",
                    "total_nitrogen_kg_ha",
                    "soil_quality_index"
                ]
            }
        
        self.training_data[crop_type.lower()] = training_data
        return training_data
    
    def predict_yield(self, crop_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict crop yield based on input parameters.
        
        Args:
            crop_type: Type of crop
            parameters: Input parameters (planting_date, water, nitrogen, etc.)
            
        Returns:
            Yield prediction with confidence intervals
        """
        crop_type = crop_type.lower()
        
        if crop_type not in self.training_data:
            self.load_training_data(crop_type)
        
        # Mock prediction model - in production, this would use trained ML models
        base_yield = self._calculate_base_yield(crop_type, parameters)
        
        # Apply adjustments based on parameters
        yield_adjustment = self._calculate_yield_adjustments(crop_type, parameters)
        
        predicted_yield = base_yield * yield_adjustment
        
        # Add uncertainty
        uncertainty = self._calculate_uncertainty(crop_type, parameters)
        
        # Generate recommendations
        recommendations = self._generate_yield_recommendations(crop_type, parameters, predicted_yield)
        
        prediction = {
            "crop_type": crop_type,
            "predicted_yield_kg_ha": round(predicted_yield, 2),
            "confidence_interval": {
                "lower": round(predicted_yield - uncertainty, 2),
                "upper": round(predicted_yield + uncertainty, 2)
            },
            "uncertainty_percent": round(uncertainty / predicted_yield * 100, 1),
            "input_parameters": parameters,
            "recommendations": recommendations,
            "model_version": "1.0",
            "prediction_date": datetime.now().isoformat()
        }
        
        # Store prediction history
        if crop_type not in self.prediction_history:
            self.prediction_history[crop_type] = []
        self.prediction_history[crop_type].append(prediction)
        
        return prediction
    
    def _calculate_base_yield(self, crop_type: str, parameters: Dict[str, Any]) -> float:
        """Calculate base yield for crop type."""
        base_yields = {
            "rice": 3500,
            "wheat": 3200,
            "maize": 3800,
            "cotton": 1800,
            "sugarcane": 45000
        }
        return base_yields.get(crop_type, 2500)
    
    def _calculate_yield_adjustments(self, crop_type: str, parameters: Dict[str, Any]) -> float:
        """Calculate yield adjustments based on parameters."""
        adjustment = 1.0
        
        # Planting date adjustment
        planting_date = parameters.get("planting_date")
        if planting_date:
            if crop_type == "rice" and planting_date in ["early", "optimal"]:
                adjustment *= 1.1
            elif crop_type == "wheat" and planting_date == "optimal":
                adjustment *= 1.05
        
        # Water adjustment
        water_input = parameters.get("water_input_mm", 0)
        if water_input > 0:
            optimal_water = {"rice": 1200, "wheat": 800, "maize": 900}
            optimal = optimal_water.get(crop_type, 800)
            if optimal * 0.8 <= water_input <= optimal * 1.2:
                adjustment *= 1.05
            elif water_input < optimal * 0.6 or water_input > optimal * 1.5:
                adjustment *= 0.8
        
        # Nitrogen adjustment
        nitrogen_input = parameters.get("nitrogen_input_kg_ha", 0)
        if nitrogen_input > 0:
            optimal_nitrogen = {"rice": 120, "wheat": 100, "maize": 150}
            optimal = optimal_nitrogen.get(crop_type, 100)
            if optimal * 0.9 <= nitrogen_input <= optimal * 1.1:
                adjustment *= 1.08
            elif nitrogen_input < optimal * 0.7:
                adjustment *= 0.85
        
        # Soil quality adjustment
        soil_quality = parameters.get("soil_quality_index", 0.5)
        adjustment *= (0.7 + soil_quality * 0.6)  # Range: 0.7 to 1.3
        
        return adjustment
    
    def _calculate_uncertainty(self, crop_type: str, parameters: Dict[str, Any]) -> float:
        """Calculate prediction uncertainty."""
        base_uncertainty = 200  # kg/hectare
        
        # Increase uncertainty for missing parameters
        required_params = ["planting_date", "water_input_mm", "nitrogen_input_kg_ha"]
        missing_params = [p for p in required_params if p not in parameters]
        uncertainty = base_uncertainty + len(missing_params) * 100
        
        return uncertainty
    
    def _generate_yield_recommendations(self, crop_type: str, parameters: Dict[str, Any], predicted_yield: float) -> List[str]:
        """Generate recommendations based on yield prediction."""
        recommendations = []
        
        # Planting date recommendations
        if parameters.get("planting_date") == "late":
            recommendations.append("🌱 Consider earlier planting next season for 10-15% yield increase")
        
        # Water efficiency recommendations
        water_input = parameters.get("water_input_mm", 0)
        if water_input > 0:
            optimal_water = {"rice": 1200, "wheat": 800, "maize": 900}
            optimal = optimal_water.get(crop_type, 800)
            if water_input > optimal * 1.3:
                recommendations.append("💧 Reduce water input by 20% - current usage exceeds optimal")
            elif water_input < optimal * 0.7:
                recommendations.append("💧 Increase water input by 25% for better yield potential")
        
        # Nitrogen efficiency recommendations
        nitrogen_input = parameters.get("nitrogen_input_kg_ha", 0)
        if nitrogen_input > 0:
            optimal_nitrogen = {"rice": 120, "wheat": 100, "maize": 150}
            optimal = optimal_nitrogen.get(crop_type, 100)
            if nitrogen_input < optimal * 0.8:
                recommendations.append("🌿 Increase nitrogen application by 20-30 kg/ha")
        
        # General recommendations
        if predicted_yield > 4000:
            recommendations.append("🌟 Excellent yield potential - maintain current practices")
        elif predicted_yield < 2500:
            recommendations.append("⚠️ Low yield potential - review soil health and nutrient management")
        
        recommendations.extend([
            "📊 Monitor crop growth stages for timely interventions",
            "🔬 Consider soil testing for precise nutrient recommendations"
        ])
        
        return recommendations
    
    def estimate_water_efficiency(self, crop_type: str, predicted_yield: float, water_input: float) -> Dict[str, Any]:
        """
        Estimate water use efficiency based on yield prediction.
        
        Args:
            crop_type: Type of crop
            predicted_yield: Predicted yield in kg/ha
            water_input: Water input in mm
            
        Returns:
            Water efficiency metrics
        """
        if water_input <= 0:
            return {"error": "Water input must be greater than 0"}
        
        # Calculate water use efficiency (kg yield per mm water)
        wue = predicted_yield / water_input
        
        # Benchmark WUE values (kg/mm/ha)
        benchmark_wue = {
            "rice": 3.5,
            "wheat": 4.0,
            "maize": 4.5,
            "cotton": 2.0
        }
        
        benchmark = benchmark_wue.get(crop_type.lower(), 3.0)
        efficiency_percent = (wue / benchmark) * 100
        
        return {
            "crop_type": crop_type,
            "water_use_efficiency_kg_per_mm": round(wue, 2),
            "benchmark_wue": benchmark,
            "efficiency_percent": round(efficiency_percent, 1),
            "efficiency_rating": self._get_efficiency_rating(efficiency_percent),
            "recommendations": self._get_water_efficiency_recommendations(wue, benchmark)
        }
    
    def estimate_nitrogen_efficiency(self, crop_type: str, predicted_yield: float, nitrogen_input: float) -> Dict[str, Any]:
        """
        Estimate nitrogen use efficiency based on yield prediction.
        
        Args:
            crop_type: Type of crop
            predicted_yield: Predicted yield in kg/ha
            nitrogen_input: Nitrogen input in kg/ha
            
        Returns:
            Nitrogen efficiency metrics
        """
        if nitrogen_input <= 0:
            return {"error": "Nitrogen input must be greater than 0"}
        
        # Calculate nitrogen use efficiency (kg yield per kg N)
        nue = predicted_yield / nitrogen_input
        
        # Benchmark NUE values (kg yield per kg N)
        benchmark_nue = {
            "rice": 30,
            "wheat": 35,
            "maize": 25,
            "cotton": 15
        }
        
        benchmark = benchmark_nue.get(crop_type.lower(), 25)
        efficiency_percent = (nue / benchmark) * 100
        
        return {
            "crop_type": crop_type,
            "nitrogen_use_efficiency_kg_per_kg_n": round(nue, 2),
            "benchmark_nue": benchmark,
            "efficiency_percent": round(efficiency_percent, 1),
            "efficiency_rating": self._get_efficiency_rating(efficiency_percent),
            "recommendations": self._get_nitrogen_efficiency_recommendations(nue, benchmark)
        }
    
    def _get_efficiency_rating(self, efficiency_percent: float) -> str:
        """Get efficiency rating based on percentage."""
        if efficiency_percent >= 120:
            return "Excellent"
        elif efficiency_percent >= 100:
            return "Good"
        elif efficiency_percent >= 80:
            return "Average"
        elif efficiency_percent >= 60:
            return "Below Average"
        else:
            return "Poor"
    
    def _get_water_efficiency_recommendations(self, wue: float, benchmark: float) -> List[str]:
        """Get water efficiency recommendations."""
        recommendations = []
        
        if wue < benchmark * 0.8:
            recommendations.extend([
                "💧 Implement drip irrigation for better water efficiency",
                "🌱 Use mulching to reduce water evaporation",
                "📊 Monitor soil moisture levels regularly"
            ])
        elif wue > benchmark * 1.2:
            recommendations.append("🌟 Excellent water efficiency - maintain current practices")
        
        return recommendations
    
    def _get_nitrogen_efficiency_recommendations(self, nue: float, benchmark: float) -> List[str]:
        """Get nitrogen efficiency recommendations."""
        recommendations = []
        
        if nue < benchmark * 0.8:
            recommendations.extend([
                "🌿 Split nitrogen applications throughout growing season",
                "🔬 Use soil testing for precise nitrogen recommendations",
                "🌱 Consider nitrogen-fixing cover crops in rotation"
            ])
        elif nue > benchmark * 1.2:
            recommendations.append("🌟 Excellent nitrogen efficiency - maintain current practices")
        
        return recommendations
    
    def compare_varieties(self, crop_type: str, varieties: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare yield predictions across different crop varieties.
        
        Args:
            crop_type: Type of crop
            varieties: List of varieties to compare
            parameters: Common parameters for all varieties
            
        Returns:
            Variety comparison results
        """
        comparison = {
            "crop_type": crop_type,
            "varieties": varieties,
            "comparison_results": {},
            "rankings": {}
        }
        
        # Predict yield for each variety
        for variety in varieties:
            variety_params = parameters.copy()
            variety_params["variety"] = variety
            
            prediction = self.predict_yield(crop_type, variety_params)
            comparison["comparison_results"][variety] = prediction
        
        # Rank varieties by predicted yield
        yield_rankings = sorted(
            comparison["comparison_results"].items(),
            key=lambda x: x[1]["predicted_yield_kg_ha"],
            reverse=True
        )
        comparison["rankings"]["by_yield"] = [variety for variety, _ in yield_rankings]
        
        return comparison


# Global instance
yield_predictor_instance = None

def get_yield_predictor() -> YieldPredictor:
    """Get or create global yield predictor instance."""
    global yield_predictor_instance
    if yield_predictor_instance is None:
        yield_predictor_instance = YieldPredictor()
    return yield_predictor_instance

def predict_crop_yield(crop_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Predict crop yield with efficiency analysis."""
    predictor = get_yield_predictor()
    
    # Get yield prediction
    prediction = predictor.predict_yield(crop_type, parameters)
    
    # Add efficiency analysis if inputs available
    water_input = parameters.get("water_input_mm")
    nitrogen_input = parameters.get("nitrogen_input_kg_ha")
    
    if water_input:
        water_efficiency = predictor.estimate_water_efficiency(
            crop_type, prediction["predicted_yield_kg_ha"], water_input
        )
        prediction["water_efficiency"] = water_efficiency
    
    if nitrogen_input:
        nitrogen_efficiency = predictor.estimate_nitrogen_efficiency(
            crop_type, prediction["predicted_yield_kg_ha"], nitrogen_input
        )
        prediction["nitrogen_efficiency"] = nitrogen_efficiency
    
    return prediction

