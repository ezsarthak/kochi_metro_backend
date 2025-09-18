from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from datetime import datetime, timedelta
import json
import os
import tempfile
import pandas as pd
from werkzeug.utils import secure_filename
import logging
from typing import Dict, List, Optional
import traceback

# Import our custom modules
from priority_model import main_advanced_pipeline
from schedule_engine import TrainScheduler, load_trips, validate_schedule, HUB_STOPS, STATION_ORDER
from utils import generate_mock_data, calculate_performance_metrics, create_what_if_simulation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATA_FOLDER'] = 'data'
app.config['BACKEND_EXCEL_FILE'] = 'data/Final_kochi_1.xlsx'  # Use existing Excel file

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)

# Global variables to store current data
current_priority_data = None
current_schedule_data = None
current_trips_data = None

def load_initial_data():
    """Load initial data on startup"""
    global current_trips_data
    try:
        current_trips_data = load_trips('data/trips.json')
        logger.info(f"Loaded {len(current_trips_data)} trips from data/trips.json")
    except Exception as e:
        logger.error(f"Failed to load initial trips data: {e}")
        current_trips_data = []

# Load data on startup
load_initial_data()

@app.errorhandler(Exception)
def handle_exception(e):
    """Global error handler"""
    logger.error(f"Unhandled exception: {str(e)}\n{traceback.format_exc()}")
    return jsonify({
        'error': 'Internal server error',
        'message': str(e),
        'timestamp': datetime.now().isoformat()
    }), 500

@app.route('/docs')
def api_documentation():
    """Serve API documentation"""
    try:
        from api_documentation import generate_api_docs, DOC_TEMPLATE
        docs = generate_api_docs()
        return render_template_string(
            DOC_TEMPLATE,
            title=docs['info']['title'],
            description=docs['info']['description'],
            version=docs['info']['version'],
            paths=docs['paths']
        )
    except Exception as e:
        return jsonify({
            "error": "Documentation generation failed",
            "message": str(e)
        }), 500

@app.route('/docs/json')
def api_docs_json():
    """Serve API documentation as JSON"""
    try:
        from api_documentation import generate_api_docs
        return jsonify(generate_api_docs())
    except Exception as e:
        return jsonify({
            "error": "Documentation generation failed",
            "message": str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'services': {
            'priority_model': 'available',
            'scheduler': 'available',
            'trips_data': len(current_trips_data) if current_trips_data else 0
        }
    })

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get dashboard overview data"""
    try:
        # Generate dashboard metrics
        total_trains = 0
        active_trains = 0
        maintenance_trains = 0
        
        if current_priority_data:
            priority_trains = current_priority_data.get('train_priorities', [])
            total_trains = len(priority_trains)
            
            # Get maintenance trains from current schedule
            if current_schedule_data:
                maintenance_train_list = current_schedule_data.get('maintenance_trains', [])
                maintenance_trains = len(maintenance_train_list)
                active_trains = total_trains - maintenance_trains
            else:
                active_trains = total_trains
                maintenance_trains = 0
        else:
            # Fallback: try to read Excel file directly for basic counts
            try:
                backend_file = app.config['BACKEND_EXCEL_FILE']
                if os.path.exists(backend_file):
                    df = pd.read_excel(backend_file)
                    if 'Train_ID' in df.columns:
                        total_trains = df['Train_ID'].nunique()
                        active_trains = total_trains
                        maintenance_trains = 0
            except Exception as e:
                logger.warning(f"Could not read Excel file for basic counts: {e}")
                total_trains = 0
                active_trains = 0
                maintenance_trains = 0
        
        # Calculate today's metrics from actual trips data
        today_trips = len(current_trips_data) if current_trips_data else 0
        
        # Calculate actual performance metrics from data
        on_time_percentage = 94.2  # This would need actual performance data
        avg_delay_minutes = 2.3    # This would need actual performance data
        
        if current_priority_data:
            # Calculate actual performance from priority data if available
            priority_trains = current_priority_data.get('train_priorities', [])
            if priority_trains:
                avg_score = sum(t.get('final_score', 0) for t in priority_trains) / len(priority_trains)
                # Convert score to performance percentage (higher score = worse performance)
                on_time_percentage = max(70, 100 - (avg_score * 10))
                avg_delay_minutes = max(0.5, avg_score * 0.5)
        
        # Generate alerts for worst performing trains
        alerts = []
        if current_priority_data:
            priority_trains = current_priority_data.get('train_priorities', [])
            worst_trains = sorted(priority_trains, key=lambda x: x['priority_rank'], reverse=True)[:3]
            
            for train in worst_trains:
                alerts.append({
                    'type': 'maintenance',
                    'severity': 'high',
                    'message': f"Train {train['train_id']} requires urgent maintenance (Priority rank {train['priority_rank']} - worst performing)",
                    'timestamp': datetime.now().isoformat()
                })
        
        # Calculate system efficiency based on actual data
        system_efficiency = 96.8
        if current_priority_data:
            priority_trains = current_priority_data.get('train_priorities', [])
            if priority_trains:
                avg_rank = sum(t.get('priority_rank', 0) for t in priority_trains) / len(priority_trains)
                # Convert average rank to efficiency (lower average rank = higher efficiency)
                system_efficiency = max(80, 100 - (avg_rank / total_trains * 20))
        
        return jsonify({
            'summary': {
                'total_trains': total_trains,
                'active_trains': active_trains,
                'maintenance_trains': maintenance_trains,
                'today_trips': today_trips,
                'on_time_percentage': round(on_time_percentage, 1),
                'avg_delay_minutes': round(avg_delay_minutes, 1)
            },
            'alerts': alerts,
            'quick_stats': {
                'trains_in_service': active_trains,
                'scheduled_maintenance': maintenance_trains,
                'emergency_alerts': len([a for a in alerts if a['severity'] == 'high']),
                'system_efficiency': round(system_efficiency, 1)
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in dashboard endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/priority/analyze', methods=['POST'])
def analyze_train_priorities():
    """Analyze train priorities from existing backend Excel file"""
    global current_priority_data
    
    try:
        backend_file = app.config['BACKEND_EXCEL_FILE']
        
        if not os.path.exists(backend_file):
            return jsonify({
                'error': f'Backend Excel file not found: {backend_file}',
                'message': 'Please ensure Final_kochi_1.xlsx exists in the data folder'
            }), 404
        
        logger.info(f"Running priority analysis on existing backend file: {backend_file}")
        result = main_advanced_pipeline(backend_file)
        
        if result:
            current_priority_data = result
            
            return jsonify({
                'status': 'success',
                'message': 'Priority analysis completed successfully using existing backend Excel file',
                'data': result,
                'file_used': backend_file,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Priority analysis failed'}), 500
            
    except Exception as e:
        logger.error(f"Error in priority analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/priority/current', methods=['GET'])
def get_current_priorities():
    """Get current priority analysis results"""
    if not current_priority_data:
        return jsonify({'error': 'No priority data available. Please run analysis first.'}), 404
    
    return jsonify({
        'status': 'success',
        'data': current_priority_data,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/schedule/generate', methods=['POST'])
def generate_schedule():
    """Generate train schedule with maintenance slots based on available slots"""
    global current_schedule_data, current_priority_data
    
    try:
        data = request.get_json()
        available_slots = data.get('available_slots', 4)
        
        if not current_priority_data:
            logger.info("No priority data available, running priority analysis first...")
            backend_file = app.config['BACKEND_EXCEL_FILE']
            
            if not os.path.exists(backend_file):
                return jsonify({
                    'error': f'Backend Excel file not found: {backend_file}',
                    'message': 'Please ensure Final_kochi_1.xlsx exists in the data folder'
                }), 404
            
            result = main_advanced_pipeline(backend_file)
            if result:
                current_priority_data = result
                logger.info("Priority analysis completed automatically")
            else:
                return jsonify({'error': 'Failed to run priority analysis required for schedule generation'}), 500
        
        maintenance_trains = []
        if current_priority_data:
            priority_trains = current_priority_data.get('train_priorities', [])
            sorted_trains = sorted(priority_trains, key=lambda x: x['priority_rank'], reverse=True)
            maintenance_trains = [f"Train_{t['train_id']}" for t in sorted_trains[:available_slots]]
            logger.info(f"Selected worst-ranked trains for maintenance: {maintenance_trains}")
        else:
            # This should never happen now, but keeping as fallback
            maintenance_trains = [f"Train_{i}" for i in range(25, 25-available_slots, -1)]
            logger.warning("No priority data available, using fallback train selection")
        
        logger.info(f"Generating schedule with {available_slots} available maintenance slots")
        
        scheduler = TrainScheduler(
            trips=current_trips_data,
            num_trains=25,
            maintenance_trains=set(maintenance_trains),
            hub_stops=HUB_STOPS,
            station_order=STATION_ORDER
        )
        
        schedule = scheduler.solve(time_limit_seconds=120)
        
        if schedule:
            is_valid, validation_report = validate_schedule(
                schedule, current_trips_data, set(maintenance_trains), HUB_STOPS
            )
            
            schedule_data = {
                'schedule': schedule,
                'maintenance_trains': maintenance_trains,
                'available_slots': available_slots,
                'selection_method': 'worst_priority_rank' if current_priority_data else 'fallback',
                'validation': {
                    'is_valid': is_valid,
                    'report': validation_report
                },
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_trains': len(schedule),
                    'maintenance_slots_used': len(maintenance_trains),
                    'available_slots_requested': available_slots,
                    'total_trips': sum(len(trips) for trips in schedule.values()),
                    'priority_data_used': current_priority_data is not None,
                    'auto_analysis_run': True  # Track if analysis was auto-run
                }
            }
            
            current_schedule_data = schedule_data
            
            return jsonify({
                'status': 'success',
                'message': f'Schedule generated successfully with {len(maintenance_trains)} worst-ranked trains selected for maintenance',
                'data': schedule_data,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to generate feasible schedule'}), 500
            
    except Exception as e:
        logger.error(f"Error generating schedule: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/schedule/current', methods=['GET'])
def get_current_schedule():
    """Get current schedule data"""
    if not current_schedule_data:
        return jsonify({'error': 'No schedule data available. Please generate schedule first.'}), 404
    
    return jsonify({
        'status': 'success',
        'data': current_schedule_data,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/trains', methods=['GET'])
def get_trains():
    """Get list of all trains with their current status"""
    try:
        trains = []
        
        if current_priority_data:
            priority_trains = current_priority_data.get('train_priorities', [])
            
            for train_priority in priority_trains:
                train_id = f"Train_{train_priority['train_id']}"
                
                schedule_info = None
                if current_schedule_data and train_id in current_schedule_data['schedule']:
                    trips = current_schedule_data['schedule'][train_id]
                    schedule_info = {
                        'total_trips': len(trips),
                        'first_trip': trips[0] if trips else None,
                        'last_trip': trips[-1] if trips else None
                    }
                
                status = 'active'
                if train_id in (current_schedule_data.get('maintenance_trains', []) if current_schedule_data else []):
                    status = 'maintenance'
                elif not schedule_info or schedule_info['total_trips'] == 0:
                    status = 'idle'
                
                trains.append({
                    'train_id': train_id,
                    'status': status,
                    'priority_rank': train_priority['priority_rank'],
                    'priority_score': train_priority['final_score'],
                    'scheduled_trips': schedule_info['total_trips'] if schedule_info else 0,
                    'maintenance_due': status == 'maintenance',
                    'last_updated': datetime.now().isoformat()
                })
        else:
            # Fallback: try to get train IDs from Excel file
            try:
                backend_file = app.config['BACKEND_EXCEL_FILE']
                if os.path.exists(backend_file):
                    df = pd.read_excel(backend_file)
                    if 'Train_ID' in df.columns:
                        unique_trains = df['Train_ID'].unique()
                        for train_id_num in unique_trains:
                            train_id = f"Train_{train_id_num}"
                            trains.append({
                                'train_id': train_id,
                                'status': 'unknown',
                                'priority_rank': None,
                                'priority_score': None,
                                'scheduled_trips': 0,
                                'maintenance_due': False,
                                'last_updated': datetime.now().isoformat()
                            })
            except Exception as e:
                logger.warning(f"Could not read Excel file for train list: {e}")
                return jsonify({'error': 'No train data available. Please run priority analysis first.'}), 404
        
        return jsonify({
            'status': 'success',
            'data': trains,
            'total_trains': len(trains),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting trains: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trains/<train_id>', methods=['GET'])
def get_train_details(train_id):
    """Get detailed information for a specific train"""
    try:
        priority_info = None
        if current_priority_data:
            for train_priority in current_priority_data.get('train_priorities', []):
                if f"Train_{train_priority['train_id']}" == train_id:
                    priority_info = train_priority
                    break
        
        if not priority_info:
            return jsonify({'error': f'Train {train_id} not found in priority data. Please run priority analysis first.'}), 404
        
        schedule_info = None
        if current_schedule_data and train_id in current_schedule_data['schedule']:
            trips = current_schedule_data['schedule'][train_id]
            
            trip_details = []
            if current_trips_data:
                trip_lookup = {trip['trip_id']: trip for trip in current_trips_data}
                for trip_id in trips:
                    if trip_id in trip_lookup:
                        trip = trip_lookup[trip_id]
                        if trip.get('stops'):
                            first_stop = trip['stops'][0]
                            last_stop = trip['stops'][-1]
                            trip_details.append({
                                'trip_id': trip_id,
                                'start_time': first_stop['departure_time'],
                                'end_time': last_stop['arrival_time'],
                                'start_station': first_stop['stop_name'],
                                'end_station': last_stop['stop_name'],
                                'total_stops': len(trip['stops'])
                            })
            
            schedule_info = {
                'total_trips': len(trips),
                'trips': trip_details,
                'first_departure': trip_details[0]['start_time'] if trip_details else None,
                'last_arrival': trip_details[-1]['end_time'] if trip_details else None
            }
        
        status = 'active'
        current_location = 'Unknown'
        if train_id in (current_schedule_data.get('maintenance_trains', []) if current_schedule_data else []):
            status = 'maintenance'
            current_location = 'Maintenance Hub'
        elif schedule_info and schedule_info['trips']:
            current_time = datetime.now().time()
            for trip in schedule_info['trips']:
                trip_start = datetime.strptime(trip['start_time'], '%H:%M:%S').time()
                trip_end = datetime.strptime(trip['end_time'], '%H:%M:%S').time()
                if trip_start <= current_time <= trip_end:
                    current_location = f"En route: {trip['start_station']} → {trip['end_station']}"
                    break
            else:
                current_location = schedule_info['trips'][-1]['end_station'] if schedule_info['trips'] else 'Depot'
        
        performance_metrics = {
            'priority_rank': priority_info['priority_rank'],
            'priority_score': priority_info['final_score'],
            'performance_rating': 'Poor' if priority_info['priority_rank'] > 20 else 'Average' if priority_info['priority_rank'] > 10 else 'Good',
            'maintenance_urgency': 'High' if priority_info['priority_rank'] > 20 else 'Medium' if priority_info['priority_rank'] > 10 else 'Low'
        }
        
        train_details = {
            'train_id': train_id,
            'status': status,
            'current_location': current_location,
            'priority_info': priority_info,
            'schedule_info': schedule_info,
            'maintenance_info': {
                'is_maintenance_train': train_id in (current_schedule_data.get('maintenance_trains', []) if current_schedule_data else []),
                'priority_rank': priority_info['priority_rank'],
                'urgency_level': performance_metrics['maintenance_urgency']
            },
            'performance_metrics': performance_metrics,
            'last_updated': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'data': train_details,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting train details for {train_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/maintenance/hub', methods=['GET'])
def get_maintenance_hub_data():
    """Get maintenance hub information"""
    try:
        hub_data = {}
        
        if current_schedule_data:
            maintenance_trains = current_schedule_data.get('maintenance_trains', [])
            
            for hub in HUB_STOPS:
                hub_data[hub] = {
                    'hub_name': hub,
                    'trains_scheduled': [],
                    'capacity': 5,
                    'current_occupancy': 0
                }
            
            for train_id in maintenance_trains:
                if train_id in current_schedule_data['schedule']:
                    trips = current_schedule_data['schedule'][train_id]
                    if trips and current_trips_data:
                        trip_lookup = {trip['trip_id']: trip for trip in current_trips_data}
                        last_trip_id = trips[-1]
                        if last_trip_id in trip_lookup:
                            last_trip = trip_lookup[last_trip_id]
                            if last_trip.get('stops'):
                                end_station = last_trip['stops'][-1]['stop_id']
                                if end_station in hub_data:
                                    hub_data[end_station]['trains_scheduled'].append({
                                        'train_id': train_id,
                                        'arrival_time': last_trip['stops'][-1]['arrival_time'],
                                        'maintenance_type': 'scheduled',
                                        'estimated_duration': '4 hours',
                                        'status': 'pending'
                                    })
                                    hub_data[end_station]['current_occupancy'] += 1
        
        return jsonify({
            'status': 'success',
            'data': {
                'hubs': list(hub_data.values()),
                'total_maintenance_trains': len(current_schedule_data.get('maintenance_trains', []) if current_schedule_data else []),
                'alerts': [
                    {
                        'type': 'capacity',
                        'message': 'MUTT hub approaching capacity',
                        'severity': 'warning'
                    }
                ] if any(hub['current_occupancy'] >= hub['capacity'] * 0.8 for hub in hub_data.values()) else []
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting maintenance hub data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulation/what-if', methods=['POST'])
def what_if_simulation():
    """Run what-if simulation for schedule changes"""
    try:
        data = request.get_json()
        
        changes = data.get('changes', {})
        scenario_name = data.get('scenario_name', 'Unnamed Scenario')
        
        simulation_result = create_what_if_simulation(
            current_schedule_data,
            current_trips_data,
            changes
        )
        
        return jsonify({
            'status': 'success',
            'data': {
                'scenario_name': scenario_name,
                'simulation_result': simulation_result,
                'impact_analysis': {
                    'affected_trains': simulation_result.get('affected_trains', 0),
                    'schedule_conflicts': simulation_result.get('conflicts', 0),
                    'efficiency_change': simulation_result.get('efficiency_delta', 0),
                    'recommendations': simulation_result.get('recommendations', [])
                }
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in what-if simulation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/notifications', methods=['GET'])
def get_notifications():
    """Get system notifications and alerts"""
    try:
        notifications = []
        
        if current_priority_data:
            # Get worst performing trains (highest priority ranks)
            priority_trains = current_priority_data.get('train_priorities', [])
            worst_trains = sorted(priority_trains, key=lambda x: x['priority_rank'], reverse=True)[:3]
            
            for train in worst_trains:
                notifications.append({
                    'id': f"priority_{train['train_id']}",
                    'type': 'maintenance',
                    'severity': 'high',
                    'title': f"Urgent Maintenance Required",
                    'message': f"Train {train['train_id']} has priority rank {train['priority_rank']} (worst performing) and requires immediate maintenance attention",
                    'timestamp': datetime.now().isoformat(),
                    'read': False
                })
        
        if current_schedule_data:
            validation = current_schedule_data.get('validation', {})
            if not validation.get('is_valid', True):
                notifications.append({
                    'id': 'schedule_validation',
                    'type': 'system',
                    'severity': 'warning',
                    'title': 'Schedule Validation Issues',
                    'message': 'Current schedule has validation warnings. Please review.',
                    'timestamp': datetime.now().isoformat(),
                    'read': False
                })
        
        notifications.extend([
            {
                'id': 'weather_alert',
                'type': 'operational',
                'severity': 'medium',
                'title': 'Weather Advisory',
                'message': 'Heavy rain expected between 14:00-16:00. Monitor train delays.',
                'timestamp': (datetime.now() - timedelta(hours=1)).isoformat(),
                'read': False
            },
            {
                'id': 'system_update',
                'type': 'system',
                'severity': 'low',
                'title': 'System Update Complete',
                'message': 'Scheduling system updated to version 1.2.3',
                'timestamp': (datetime.now() - timedelta(hours=3)).isoformat(),
                'read': True
            }
        ])
        
        return jsonify({
            'status': 'success',
            'data': {
                'notifications': notifications,
                'unread_count': len([n for n in notifications if not n['read']]),
                'total_count': len(notifications)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting notifications: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/performance', methods=['GET'])
def get_performance_report():
    """Generate performance analytics report"""
    try:
        period = request.args.get('period', 'daily')
        train_id = request.args.get('train_id', None)
        
        performance_data = calculate_performance_metrics(
            current_schedule_data,
            current_priority_data,
            period,
            train_id
        )
        
        return jsonify({
            'status': 'success',
            'data': performance_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/schedule', methods=['GET'])
def export_schedule():
    """Export current schedule to various formats"""
    try:
        format_type = request.args.get('format', 'json')
        
        if not current_schedule_data:
            return jsonify({'error': 'No schedule data available'}), 404
        
        if format_type == 'json':
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(current_schedule_data, f, indent=2)
                temp_path = f.name
            
            return send_file(
                temp_path,
                as_attachment=True,
                download_name=f'train_schedule_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                mimetype='application/json'
            )
        
        elif format_type == 'csv':
            rows = []
            for train_id, trips in current_schedule_data['schedule'].items():
                for i, trip_id in enumerate(trips):
                    rows.append({
                        'Train_ID': train_id,
                        'Sequence': i + 1,
                        'Trip_ID': trip_id,
                        'Is_Maintenance_Train': train_id in current_schedule_data.get('maintenance_trains', [])
                    })
            
            df = pd.DataFrame(rows)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df.to_csv(f.name, index=False)
                temp_path = f.name
            
            return send_file(
                temp_path,
                as_attachment=True,
                download_name=f'train_schedule_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mimetype='text/csv'
            )
        
        else:
            return jsonify({'error': f'Unsupported format: {format_type}'}), 400
            
    except Exception as e:
        logger.error(f"Error exporting schedule: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/maintenance/recommendations', methods=['GET'])
def get_maintenance_recommendations():
    """Get maintenance recommendations based on priority analysis"""
    try:
        if not current_priority_data:
            return jsonify({'error': 'No priority data available. Please run analysis first.'}), 404
        
        max_slots = request.args.get('max_slots', 10, type=int)
        
        priority_trains = current_priority_data.get('train_priorities', [])
        sorted_trains = sorted(priority_trains, key=lambda x: x['priority_rank'], reverse=True)
        
        recommendations = []
        for i, train in enumerate(sorted_trains[:max_slots]):
            recommendations.append({
                'train_id': f"Train_{train['train_id']}",
                'priority_rank': train['priority_rank'],
                'priority_score': train['final_score'],
                'recommendation_order': i + 1,
                'urgency': 'high' if train['priority_rank'] >= 20 else 'medium' if train['priority_rank'] >= 15 else 'low',
                'estimated_maintenance_hours': 4 + (train['priority_rank'] * 0.2)
            })
        
        return jsonify({
            'status': 'success',
            'data': {
                'recommendations': recommendations,
                'total_trains_analyzed': len(priority_trains),
                'worst_trains_shown': len(recommendations),
                'analysis_timestamp': current_priority_data.get('analysis_metadata', {}).get('timestamp')
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting maintenance recommendations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trains/running', methods=['GET'])
def get_running_trains():
    """Get trains running at specific time, optionally filtered by station and train_id"""
    try:
        # Get query parameters
        time_param = request.args.get('time')  # Required - format HH:MM:SS
        station_param = request.args.get('station')  # Optional
        train_id_param = request.args.get('train_id')  # Optional
        
        if not time_param:
            return jsonify({'error': 'time parameter is required (format: HH:MM:SS)'}), 400
        
        # Validate and convert time to seconds
        try:
            query_time_seconds = time_to_seconds(time_param)
        except ValueError:
            return jsonify({'error': 'Invalid time format. Use HH:MM:SS'}), 400
        
        if not current_schedule_data:
            return jsonify({'error': 'No schedule data available. Please generate schedule first.'}), 404
        
        # Create trip lookup for detailed information
        trip_lookup = {trip['trip_id']: trip for trip in current_trips_data} if current_trips_data else {}
        
        running_trains = []
        
        # Check each train in the schedule
        for train_name, trip_ids in current_schedule_data['schedule'].items():
            # Filter by train_id if specified - show current or next trip for this specific train
            if train_id_param and train_name == train_id_param:
                current_trip = None
                next_trip = None
                
                # Find current trip or next upcoming trip
                for trip_id in trip_ids:
                    if trip_id not in trip_lookup:
                        continue
                    
                    trip = trip_lookup[trip_id]
                    start_time, end_time = calculate_trip_duration(trip)
                    
                    # Check if this is the current trip
                    if start_time <= query_time_seconds <= end_time:
                        current_trip = (trip_id, trip, start_time, end_time)
                        break
                    # Check if this is a future trip
                    elif start_time > query_time_seconds:
                        if not next_trip or start_time < next_trip[2]:
                            next_trip = (trip_id, trip, start_time, end_time)
                
                # Use current trip if available, otherwise next trip
                selected_trip = current_trip or next_trip
                if selected_trip:
                    trip_id, trip, start_time, end_time = selected_trip
                    start_stop, end_stop = get_trip_endpoints(trip)
                    
                    # Determine status and location
                    if current_trip:
                        trip_progress = (query_time_seconds - start_time) / (end_time - start_time) * 100
                        current_location = f"En route: {start_stop} → {end_stop}"
                        if trip_progress < 10:
                            current_location = f"Departing {start_stop}"
                        elif trip_progress > 90:
                            current_location = f"Approaching {end_stop}"
                        trip_status = "current"
                    else:
                        current_location = f"Next trip: {start_stop} → {end_stop}"
                        trip_status = "next"
                    
                    # Get priority information
                    priority_info = None
                    if current_priority_data:
                        for train_priority in current_priority_data.get('train_priorities', []):
                            if f"Train_{train_priority['train_id']}" == train_name:
                                priority_info = {
                                    'priority_rank': train_priority['priority_rank'],
                                    'priority_score': train_priority['final_score']
                                }
                                break
                    
                    running_train_info = {
                        'train_id': train_name,
                        'trip_id': trip_id,
                        'trip_status': trip_status,
                        'current_trip': {
                            'start_time': seconds_to_time(start_time),
                            'end_time': seconds_to_time(end_time),
                            'start_station': start_stop,
                            'end_station': end_stop,
                            'progress_percentage': round((query_time_seconds - start_time) / (end_time - start_time) * 100, 1) if current_trip else 0
                        },
                        'current_location': current_location,
                        'status': 'maintenance' if train_name in current_schedule_data.get('maintenance_trains', []) else 'active',
                        'priority_info': priority_info,
                        'trip_details': {
                            'total_stops': len(trip.get('stops', [])),
                            'stops': trip.get('stops', [])
                        }
                    }
                    
                    running_trains.append(running_train_info)
                continue
            
            # If train_id not specified or doesn't match, continue with station filtering
            if train_id_param and train_name != train_id_param:
                continue
            
            # Check each trip for this train
            for trip_id in trip_ids:
                if trip_id not in trip_lookup:
                    continue
                
                trip = trip_lookup[trip_id]
                start_time, end_time = calculate_trip_duration(trip)
                
                if station_param:
                    trip_stations = [stop['stop_id'] for stop in trip.get('stops', [])]
                    start_stop, end_stop = get_trip_endpoints(trip)
                    
                    # Only show trains that are currently at the station or will be at the station at the given time
                    station_involved = station_param in trip_stations
                    
                    if not station_involved:
                        continue
                    
                    # Check if train is currently running through this station
                    if start_time <= query_time_seconds <= end_time:
                        # Calculate which stop the train is currently at or approaching
                        trip_progress = (query_time_seconds - start_time) / (end_time - start_time)
                        stops = trip.get('stops', [])
                        
                        # Determine current position relative to the queried station
                        station_index = None
                        for i, stop in enumerate(stops):
                            if stop['stop_id'] == station_param:
                                station_index = i
                                break
                        
                        if station_index is not None:
                            # Calculate approximate position in the trip
                            current_stop_index = int(trip_progress * len(stops))
                            
                            if current_stop_index <= station_index:
                                # Train hasn't reached the station yet or is currently at it
                                if current_stop_index == station_index:
                                    current_location = f"Currently at {station_param}"
                                else:
                                    current_location = f"Approaching {station_param} (currently near stop {current_stop_index + 1})"
                            else:
                                # Train has passed the station
                                current_location = f"Departed from {station_param} (en route to {end_stop})"
                            
                            # Get priority information
                            priority_info = None
                            if current_priority_data:
                                for train_priority in current_priority_data.get('train_priorities', []):
                                    if f"Train_{train_priority['train_id']}" == train_name:
                                        priority_info = {
                                            'priority_rank': train_priority['priority_rank'],
                                            'priority_score': train_priority['final_score']
                                        }
                                        break
                            
                            running_train_info = {
                                'train_id': train_name,
                                'trip_id': trip_id,
                                'trip_status': 'current',
                                'current_trip': {
                                    'start_time': seconds_to_time(start_time),
                                    'end_time': seconds_to_time(end_time),
                                    'start_station': start_stop,
                                    'end_station': end_stop,
                                    'progress_percentage': round(trip_progress * 100, 1)
                                },
                                'current_location': current_location,
                                'station_status': 'at_station' if current_stop_index == station_index else ('approaching' if current_stop_index < station_index else 'departed'),
                                'status': 'maintenance' if train_name in current_schedule_data.get('maintenance_trains', []) else 'active',
                                'priority_info': priority_info,
                                'trip_details': {
                                    'total_stops': len(stops),
                                    'stops': stops,
                                    'station_position': station_index + 1,
                                    'current_position': current_stop_index + 1
                                }
                            }
                            
                            running_trains.append(running_train_info)
                            break
                    
                    # Check for future trips that will pass through this station
                    elif start_time > query_time_seconds:
                        # Get priority information
                        priority_info = None
                        if current_priority_data:
                            for train_priority in current_priority_data.get('train_priorities', []):
                                if f"Train_{train_priority['train_id']}" == train_name:
                                    priority_info = {
                                        'priority_rank': train_priority['priority_rank'],
                                        'priority_score': train_priority['final_score']
                                    }
                                    break
                        
                        # Find station position in the trip
                        stops = trip.get('stops', [])
                        station_index = None
                        for i, stop in enumerate(stops):
                            if stop['stop_id'] == station_param:
                                station_index = i
                                break
                        
                        running_train_info = {
                            'train_id': train_name,
                            'trip_id': trip_id,
                            'trip_status': 'upcoming',
                            'current_trip': {
                                'start_time': seconds_to_time(start_time),
                                'end_time': seconds_to_time(end_time),
                                'start_station': start_stop,
                                'end_station': end_stop,
                                'progress_percentage': 0
                            },
                            'current_location': f"Scheduled to pass through {station_param}",
                            'station_status': 'scheduled',
                            'status': 'maintenance' if train_name in current_schedule_data.get('maintenance_trains', []) else 'active',
                            'priority_info': priority_info,
                            'trip_details': {
                                'total_stops': len(stops),
                                'stops': stops,
                                'station_position': (station_index + 1) if station_index is not None else None
                            }
                        }
                        
                        running_trains.append(running_train_info)
                        break
                else:
                    # No station filter - show all currently running trains
                    if start_time <= query_time_seconds <= end_time:
                        start_stop, end_stop = get_trip_endpoints(trip)
                        trip_progress = (query_time_seconds - start_time) / (end_time - start_time) * 100
                        
                        current_location = f"En route: {start_stop} → {end_stop}"
                        if trip_progress < 10:
                            current_location = f"Departing {start_stop}"
                        elif trip_progress > 90:
                            current_location = f"Approaching {end_stop}"
                        
                        # Get priority information
                        priority_info = None
                        if current_priority_data:
                            for train_priority in current_priority_data.get('train_priorities', []):
                                if f"Train_{train_priority['train_id']}" == train_name:
                                    priority_info = {
                                        'priority_rank': train_priority['priority_rank'],
                                        'priority_score': train_priority['final_score']
                                    }
                                    break
                        
                        running_train_info = {
                            'train_id': train_name,
                            'trip_id': trip_id,
                            'trip_status': 'current',
                            'current_trip': {
                                'start_time': seconds_to_time(start_time),
                                'end_time': seconds_to_time(end_time),
                                'start_station': start_stop,
                                'end_station': end_stop,
                                'progress_percentage': round(trip_progress, 1)
                            },
                            'current_location': current_location,
                            'status': 'maintenance' if train_name in current_schedule_data.get('maintenance_trains', []) else 'active',
                            'priority_info': priority_info,
                            'trip_details': {
                                'total_stops': len(trip.get('stops', [])),
                                'stops': trip.get('stops', [])
                            }
                        }
                        
                        running_trains.append(running_train_info)
                        break  # Only one trip per train can be active at a given time
        
        # Sort by train_id for consistent output
        running_trains.sort(key=lambda x: x['train_id'])
        
        return jsonify({
            'status': 'success',
            'data': {
                'query_time': time_param,
                'station_filter': station_param,
                'train_id_filter': train_id_param,
                'running_trains': running_trains,
                'total_running_trains': len(running_trains)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting running trains: {e}")
        return jsonify({'error': str(e)}), 500

def time_to_seconds(time_str):
    """Convert time string in HH:MM:SS format to seconds"""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def seconds_to_time(seconds):
    """Convert seconds to time string in HH:MM:SS format"""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02}"

def calculate_trip_duration(trip):
    """Calculate start and end time of a trip in seconds"""
    start_time = time_to_seconds(trip['stops'][0]['departure_time'])
    end_time = time_to_seconds(trip['stops'][-1]['arrival_time'])
    return start_time, end_time

def get_trip_endpoints(trip):
    """Get start and end station IDs of a trip"""
    start_stop = trip['stops'][0]['stop_id']
    end_stop = trip['stops'][-1]['stop_id']
    return start_stop, end_stop

if __name__ == '__main__':
    print("🚂 Train Management API Server Starting...")
    print("📊 API Server: http://localhost:5000")
    print("📚 API Documentation: http://localhost:5000/docs")
    print("📋 API JSON Schema: http://localhost:5000/docs/json")
    app.run(debug=True, host='0.0.0.0', port=5000)
