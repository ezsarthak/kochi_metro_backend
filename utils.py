"""
Utility functions for the train management API
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

def generate_mock_data():
    """Generate mock data for testing"""
    return {
        'trains': [f"Train_{i}" for i in range(1, 26)],
        'stations': [
            "ALVA", "PNCU", "CPPY", "ATTK", "MUTT", "KLMT", "CCUV", "PDPM", 
            "EDAP", "CGPP", "PARV", "JLSD", "KALR", "TNHL", "MGRD", "MACE", 
            "ERSH", "KVTR", "EMKM", "VYTA", "THYK", "PETT", "VAKK", "SNJN", "TPHT"
        ],
        'hub_stations': ["ALVA", "MUTT", "CGPP", "PARV", "VYTA", "TPHT"]
    }

def calculate_performance_metrics(schedule_data: Optional[Dict], priority_data: Optional[Dict], 
                                period: str, train_id: Optional[str] = None) -> Dict[str, Any]:
    """Calculate performance metrics for reporting"""
    
    # Mock performance data generation
    base_metrics = {
        'on_time_percentage': random.uniform(90, 98),
        'average_delay_minutes': random.uniform(1, 5),
        'total_trips_completed': random.randint(800, 1200),
        'maintenance_compliance': random.uniform(85, 100),
        'fuel_efficiency': random.uniform(7.5, 9.2),
        'passenger_satisfaction': random.uniform(4.2, 4.8)
    }
    
    # Adjust based on period
    period_multiplier = {
        'daily': 1,
        'weekly': 7,
        'monthly': 30
    }.get(period, 1)
    
    base_metrics['total_trips_completed'] *= period_multiplier
    
    # Generate trend data
    trend_data = []
    for i in range(period_multiplier):
        date = datetime.now() - timedelta(days=period_multiplier - i - 1)
        trend_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'on_time_percentage': base_metrics['on_time_percentage'] + random.uniform(-5, 5),
            'total_trips': random.randint(30, 50),
            'delays': random.randint(1, 8)
        })
    
    # Train-specific data if requested
    train_specific = None
    if train_id:
        train_specific = {
            'train_id': train_id,
            'total_distance': random.uniform(200, 400),
            'maintenance_score': random.uniform(70, 95),
            'breakdown_incidents': random.randint(0, 3),
            'last_maintenance': (datetime.now() - timedelta(days=random.randint(10, 90))).isoformat()
        }
    
    return {
        'period': period,
        'metrics': base_metrics,
        'trends': trend_data,
        'train_specific': train_specific,
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'performance_grade': 'A' if base_metrics['on_time_percentage'] > 95 else 'B',
            'key_insights': [
                f"On-time performance: {base_metrics['on_time_percentage']:.1f}%",
                f"Average delay: {base_metrics['average_delay_minutes']:.1f} minutes",
                f"Maintenance compliance: {base_metrics['maintenance_compliance']:.1f}%"
            ]
        }
    }

def create_what_if_simulation(schedule_data: Optional[Dict], trips_data: Optional[List], 
                            changes: Dict[str, Any]) -> Dict[str, Any]:
    """Create what-if simulation results"""
    
    if not schedule_data:
        return {
            'error': 'No schedule data available for simulation',
            'affected_trains': 0,
            'conflicts': 0,
            'efficiency_delta': 0,
            'recommendations': []
        }
    
    # Mock simulation logic
    change_type = changes.get('type', 'unknown')
    affected_trains = 0
    conflicts = 0
    efficiency_delta = 0
    recommendations = []
    
    if change_type == 'route_change':
        affected_trains = random.randint(3, 8)
        conflicts = random.randint(0, 2)
        efficiency_delta = random.uniform(-5, 10)
        recommendations = [
            "Consider adjusting departure times by 5-10 minutes",
            "Monitor passenger load on affected routes",
            "Prepare contingency plans for peak hours"
        ]
    
    elif change_type == 'maintenance_schedule':
        affected_trains = random.randint(1, 4)
        conflicts = random.randint(0, 1)
        efficiency_delta = random.uniform(-2, 5)
        recommendations = [
            "Ensure maintenance hub capacity is available",
            "Notify passengers of potential service changes",
            "Coordinate with maintenance teams"
        ]
    
    elif change_type == 'time_adjustment':
        affected_trains = random.randint(5, 12)
        conflicts = random.randint(1, 3)
        efficiency_delta = random.uniform(-3, 8)
        recommendations = [
            "Review platform availability at affected stations",
            "Update passenger information systems",
            "Check crew scheduling compatibility"
        ]
    
    else:
        recommendations = ["Unknown change type - manual review required"]
    
    return {
        'change_type': change_type,
        'affected_trains': affected_trains,
        'conflicts': conflicts,
        'efficiency_delta': efficiency_delta,
        'recommendations': recommendations,
        'simulation_details': {
            'total_scenarios_tested': random.randint(50, 200),
            'optimal_solution_found': conflicts == 0,
            'confidence_score': random.uniform(0.7, 0.95),
            'estimated_implementation_time': f"{random.randint(2, 8)} hours"
        },
        'impact_summary': {
            'passenger_impact': 'Low' if affected_trains < 5 else 'Medium' if affected_trains < 10 else 'High',
            'operational_complexity': 'Simple' if conflicts == 0 else 'Moderate' if conflicts < 3 else 'Complex',
            'cost_impact': f"${random.randint(1000, 10000):,}",
            'timeline': f"{random.randint(1, 7)} days"
        }
    }

def validate_train_id(train_id: str) -> bool:
    """Validate train ID format"""
    if not train_id.startswith('Train_'):
        return False
    
    try:
        train_num = int(train_id.split('_')[1])
        return 1 <= train_num <= 25
    except (IndexError, ValueError):
        return False

def format_time_duration(seconds: int) -> str:
    """Format duration in seconds to human readable format"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"

def calculate_distance_between_stations(station1: str, station2: str, station_order: List[str]) -> int:
    """Calculate distance between two stations"""
    try:
        idx1 = station_order.index(station1)
        idx2 = station_order.index(station2)
        return abs(idx2 - idx1)
    except ValueError:
        return 0

def get_station_info(station_id: str) -> Dict[str, Any]:
    """Get information about a station"""
    station_names = {
        "ALVA": "Aluva",
        "PNCU": "Pulinchodu", 
        "CPPY": "Companypady",
        "ATTK": "Ambattukavu",
        "MUTT": "Muttom",
        "KLMT": "Kalamassery",
        "CCUV": "Cochin University",
        "PDPM": "Pathadipalam",
        "EDAP": "Edapally",
        "CGPP": "Changampuzha Park",
        "PARV": "Palarivattom",
        "JLSD": "JLN Stadium",
        "KALR": "Kaloor",
        "TNHL": "Town Hall",
        "MGRD": "MG Road",
        "MACE": "Maharajas",
        "ERSH": "Ernakulam South",
        "KVTR": "Kadavanthra",
        "EMKM": "Elamkulam",
        "VYTA": "Vyttila",
        "THYK": "Thaikoodam",
        "PETT": "Petta",
        "VAKK": "Velloor",
        "SNJN": "SN Junction",
        "TPHT": "Tripunithura"
    }
    
    hub_stations = {"ALVA", "MUTT", "CGPP", "PARV", "VYTA", "TPHT"}
    
    return {
        'station_id': station_id,
        'station_name': station_names.get(station_id, station_id),
        'is_hub': station_id in hub_stations,
        'maintenance_facility': station_id in hub_stations,
        'platform_count': random.randint(2, 4) if station_id in hub_stations else random.randint(1, 2)
    }
