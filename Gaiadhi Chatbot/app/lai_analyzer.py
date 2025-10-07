"""
Leaf Area Index (LAI) analyzer for satellite data processing.
Integrates with Sentinel satellite data for agricultural monitoring.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

from .config import settings


class LAIAnalyzer:
    def __init__(self):
        """Initialize LAI analyzer for satellite data processing."""
        self.satellite_data = {}
        self.lai_timeseries = {}
        
    def load_satellite_data(self, region: str, date_range: tuple) -> Dict[str, Any]:
        """
        Load Sentinel satellite data for LAI analysis.
        
        Args:
            region: Geographic region (e.g., "Punjab", "Maharashtra")
            date_range: Tuple of (start_date, end_date)
            
        Returns:
            Processed satellite data
        """
        # Mock implementation - in production, this would connect to Sentinel API
        start_date, end_date = date_range
        
        # Generate mock LAI time series data
        dates = pd.date_range(start=start_date, end=end_date, freq='W')
        lai_values = np.random.uniform(0.5, 6.0, len(dates))  # LAI typically ranges 0-6
        
        # Add seasonal patterns
        for i, date in enumerate(dates):
            month = date.month
            if month in [6, 7, 8]:  # Monsoon - higher LAI
                lai_values[i] += 1.5
            elif month in [11, 12, 1, 2]:  # Winter - lower LAI
                lai_values[i] -= 0.8
        
        satellite_data = {
            "region": region,
            "date_range": date_range,
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "lai_values": lai_values.tolist(),
            "mean_lai": float(np.mean(lai_values)),
            "max_lai": float(np.max(lai_values)),
            "min_lai": float(np.min(lai_values)),
            "data_points": len(dates)
        }
        
        self.satellite_data[region] = satellite_data
        return satellite_data
    
    def analyze_lai_trends(self, region: str) -> Dict[str, Any]:
        """
        Analyze LAI trends and patterns.
        
        Args:
            region: Geographic region
            
        Returns:
            LAI trend analysis
        """
        if region not in self.satellite_data:
            return {"error": "No satellite data available for region"}
        
        data = self.satellite_data[region]
        lai_values = np.array(data["lai_values"])
        
        # Calculate trends
        x = np.arange(len(lai_values))
        trend = np.polyfit(x, lai_values, 1)[0]  # Linear trend slope
        
        # Seasonal analysis
        seasonal_patterns = self._analyze_seasonal_patterns(data)
        
        # Health indicators
        health_score = self._calculate_crop_health_score(lai_values)
        
        return {
            "region": region,
            "trend_slope": float(trend),
            "trend_direction": "increasing" if trend > 0.01 else "decreasing" if trend < -0.01 else "stable",
            "mean_lai": float(np.mean(lai_values)),
            "variability": float(np.std(lai_values)),
            "seasonal_patterns": seasonal_patterns,
            "health_score": health_score,
            "recommendations": self._generate_lai_recommendations(trend, health_score)
        }
    
    def _analyze_seasonal_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze seasonal patterns in LAI data."""
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in data["dates"]]
        lai_values = data["lai_values"]
        
        # Group by season
        seasons = {"Monsoon": [], "Winter": [], "Summer": [], "Spring": []}
        
        for date, lai in zip(dates, lai_values):
            month = date.month
            if month in [6, 7, 8, 9]:
                seasons["Monsoon"].append(lai)
            elif month in [10, 11, 12, 1, 2]:
                seasons["Winter"].append(lai)
            elif month in [3, 4, 5]:
                seasons["Spring"].append(lai)
            else:
                seasons["Summer"].append(lai)
        
        seasonal_means = {}
        for season, values in seasons.items():
            if values:
                seasonal_means[season] = {
                    "mean_lai": float(np.mean(values)),
                    "count": len(values)
                }
        
        return seasonal_means
    
    def _calculate_crop_health_score(self, lai_values: np.ndarray) -> float:
        """Calculate crop health score based on LAI values."""
        # Normalize LAI values (0-1 scale)
        normalized_lai = np.clip(lai_values / 6.0, 0, 1)
        
        # Calculate health score (higher is better)
        mean_health = float(np.mean(normalized_lai))
        consistency = 1.0 - float(np.std(normalized_lai))  # Lower std = more consistent
        
        health_score = (mean_health * 0.7 + consistency * 0.3) * 100
        return round(health_score, 2)
    
    def _generate_lai_recommendations(self, trend: float, health_score: float) -> List[str]:
        """Generate recommendations based on LAI analysis."""
        recommendations = []
        
        if trend < -0.05:
            recommendations.append("⚠️ Declining LAI trend detected - check for pest/disease issues")
        elif trend > 0.05:
            recommendations.append("✅ Positive LAI growth trend - crop development looks good")
        
        if health_score < 60:
            recommendations.append("🔍 Low crop health score - consider soil testing and fertilizer application")
        elif health_score > 80:
            recommendations.append("🌟 Excellent crop health - maintain current practices")
        
        recommendations.extend([
            "📊 Monitor LAI weekly for early problem detection",
            "🌱 Consider precision agriculture techniques for optimization"
        ])
        
        return recommendations
    
    def compare_regions(self, regions: List[str]) -> Dict[str, Any]:
        """
        Compare LAI data across multiple regions.
        
        Args:
            regions: List of regions to compare
            
        Returns:
            Comparative analysis
        """
        comparison = {
            "regions": regions,
            "comparison_data": {},
            "rankings": {}
        }
        
        for region in regions:
            if region in self.satellite_data:
                analysis = self.analyze_lai_trends(region)
                comparison["comparison_data"][region] = analysis
        
        # Rank regions by health score
        health_scores = {
            region: data.get("health_score", 0) 
            for region, data in comparison["comparison_data"].items()
        }
        sorted_regions = sorted(health_scores.items(), key=lambda x: x[1], reverse=True)
        comparison["rankings"]["by_health"] = [region for region, _ in sorted_regions]
        
        return comparison
    
    def export_analysis(self, region: str, output_dir: str = None) -> str:
        """
        Export LAI analysis to file.
        
        Args:
            region: Region to export
            output_dir: Output directory (defaults to data/lai)
            
        Returns:
            Path to exported file
        """
        if output_dir is None:
            output_dir = os.path.join("data", "lai")
        
        os.makedirs(output_dir, exist_ok=True)
        
        analysis = self.analyze_lai_trends(region)
        filename = f"lai_analysis_{region}_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return filepath


# Global instance
lai_analyzer_instance = None

def get_lai_analyzer() -> LAIAnalyzer:
    """Get or create global LAI analyzer instance."""
    global lai_analyzer_instance
    if lai_analyzer_instance is None:
        lai_analyzer_instance = LAIAnalyzer()
    return lai_analyzer_instance

def analyze_region_lai(region: str, days_back: int = 365) -> Dict[str, Any]:
    """Analyze LAI for a specific region."""
    analyzer = get_lai_analyzer()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Load and analyze data
    analyzer.load_satellite_data(region, (start_date, end_date))
    return analyzer.analyze_lai_trends(region)

