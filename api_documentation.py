"""
API Documentation Generator for Train Management System
Similar to FastAPI's automatic documentation but for Flask
"""

from flask import Flask, render_template_string
import json
from datetime import datetime

def generate_api_docs():
    """Generate API documentation"""
    
    api_docs = {
        "info": {
            "title": "Train Management System API",
            "version": "1.0.0",
            "description": "Comprehensive API for train scheduling, priority management, and operational analytics",
            "contact": {
                "name": "Train Management Team",
                "email": "support@trainmanagement.com"
            }
        },
        "servers": [
            {
                "url": "http://localhost:5000",
                "description": "Development server"
            }
        ],
        "paths": {
            "/api/health": {
                "get": {
                    "summary": "Health Check",
                    "description": "Check API health and service status",
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "example": {
                                "status": "healthy",
                                "timestamp": "2025-01-17T10:30:00Z",
                                "version": "1.0.0",
                                "services": {
                                    "priority_model": "available",
                                    "scheduler": "available",
                                    "trips_data": 256
                                }
                            }
                        }
                    }
                }
            },
            "/api/dashboard": {
                "get": {
                    "summary": "Dashboard Overview",
                    "description": "Get dashboard summary with key metrics and alerts",
                    "responses": {
                        "200": {
                            "description": "Dashboard data retrieved successfully",
                            "example": {
                                "summary": {
                                    "total_trains": 25,
                                    "active_trains": 21,
                                    "maintenance_trains": 4,
                                    "today_trips": 180,
                                    "on_time_percentage": 94.2,
                                    "avg_delay_minutes": 2.3
                                },
                                "alerts": [
                                    {
                                        "type": "maintenance",
                                        "severity": "high",
                                        "message": "Train 18 requires urgent maintenance (Priority 1)",
                                        "timestamp": "2025-01-17T10:30:00Z"
                                    }
                                ]
                            }
                        }
                    }
                }
            },
            "/api/priority/analyze": {
                "post": {
                    "summary": "Analyze Train Priorities (Backend Data)",
                    "description": "Analyze train priorities using existing backend Excel file (data/Final_kochi_1.xlsx). No file upload required - uses your pre-existing Excel data.",
                    "responses": {
                        "200": {
                            "description": "Priority analysis completed using backend data",
                            "example": {
                                "status": "success",
                                "message": "Priority analysis completed successfully using backend data",
                                "data": {
                                    "analysis_metadata": {
                                        "total_trains": 25,
                                        "processing_time_seconds": 2.17,
                                        "file_processed": "data/Final_kochi_1.xlsx"
                                    },
                                    "train_priorities": [
                                        {
                                            "train_id": "18",
                                            "priority_rank": 25,
                                            "final_score": 0.481,
                                            "weighted_score": 0.596
                                        }
                                    ]
                                }
                            }
                        },
                        "404": {
                            "description": "Backend Excel file not found",
                            "example": {
                                "error": "Backend Excel file not found: data/Final_kochi_1.xlsx"
                            }
                        }
                    }
                }
            },
            "/api/schedule/generate": {
                "post": {
                    "summary": "Generate Train Schedule with Available Slots",
                    "description": "Generate optimized train schedule by selecting worst-ranked trains for available maintenance slots",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "available_slots": {
                                            "type": "integer",
                                            "description": "Number of available maintenance slots",
                                            "default": 4,
                                            "example": 6
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Schedule generated with worst-ranked trains selected for maintenance",
                            "example": {
                                "status": "success",
                                "message": "Schedule generated successfully with 4 worst-ranked trains selected for maintenance",
                                "data": {
                                    "schedule": {
                                        "Train_1": ["WK_15", "WK_45", "WK_78"],
                                        "Train_2": ["WK_23", "WK_56"]
                                    },
                                    "maintenance_trains": ["Train_22", "Train_23", "Train_24", "Train_25"],
                                    "available_slots": 4,
                                    "selected_criteria": "worst_priority_rank",
                                    "validation": {
                                        "is_valid": True,
                                        "report": {
                                            "overall_accuracy": 98.5
                                        }
                                    },
                                    "metadata": {
                                        "maintenance_slots_used": 4,
                                        "available_slots_requested": 4
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/maintenance/recommendations": {
                "get": {
                    "summary": "Get Maintenance Recommendations",
                    "description": "Get maintenance recommendations based on priority analysis, showing worst-ranked trains first",
                    "parameters": [
                        {
                            "name": "max_slots",
                            "in": "query",
                            "schema": {
                                "type": "integer",
                                "default": 10
                            },
                            "description": "Maximum number of trains to recommend for maintenance"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Maintenance recommendations retrieved",
                            "example": {
                                "status": "success",
                                "data": {
                                    "recommendations": [
                                        {
                                            "train_id": "Train_25",
                                            "priority_rank": 25,
                                            "priority_score": 0.123,
                                            "recommendation_order": 1,
                                            "urgency": "high",
                                            "estimated_maintenance_hours": 9.0
                                        },
                                        {
                                            "train_id": "Train_24",
                                            "priority_rank": 24,
                                            "priority_score": 0.156,
                                            "recommendation_order": 2,
                                            "urgency": "high",
                                            "estimated_maintenance_hours": 8.8
                                        }
                                    ],
                                    "total_trains_analyzed": 25,
                                    "worst_trains_shown": 10
                                }
                            }
                        }
                    }
                }
            },
            "/api/trains": {
                "get": {
                    "summary": "Get All Trains",
                    "description": "Retrieve list of all trains with current status",
                    "responses": {
                        "200": {
                            "description": "Trains retrieved successfully",
                            "example": {
                                "status": "success",
                                "data": [
                                    {
                                        "train_id": "Train_1",
                                        "status": "active",
                                        "priority_rank": 1,
                                        "priority_score": 0.481,
                                        "scheduled_trips": 11,
                                        "maintenance_due": False
                                    },
                                    {
                                        "train_id": "Train_25",
                                        "status": "maintenance",
                                        "priority_rank": 25,
                                        "priority_score": 0.123,
                                        "scheduled_trips": 0,
                                        "maintenance_due": True
                                    }
                                ],
                                "total_trains": 25
                            }
                        }
                    }
                }
            },
            "/api/trains/{train_id}": {
                "get": {
                    "summary": "Get Train Details",
                    "description": "Get detailed information for a specific train",
                    "parameters": [
                        {
                            "name": "train_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Train identifier (e.g., Train_1)"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Train details retrieved successfully",
                            "example": {
                                "status": "success",
                                "data": {
                                    "train_id": "Train_25",
                                    "status": "maintenance",
                                    "current_location": "Maintenance Hub",
                                    "priority_info": {
                                        "priority_rank": 25,
                                        "final_score": 0.123
                                    },
                                    "schedule_info": {
                                        "total_trips": 0,
                                        "trips": []
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/trains/running": {
                "get": {
                    "summary": "Get Running Trains by Time and Location",
                    "description": "Filter trains based on time, station, and train_id. When station is provided, shows only trains that are currently at that station or will pass through that station at the given time. When train_id is provided, shows current trip or next upcoming trip for that specific train.",
                    "parameters": [
                        {
                            "name": "time",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Time in HH:MM:SS format (e.g., 14:30:00) - REQUIRED"
                        },
                        {
                            "name": "station",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                            "description": "Station ID to filter by (optional). Shows ONLY trains that are currently at this station or will pass through this station at the specified time. Returns empty if no trains are at/passing through this station at the given time. If null, returns trains from all stations."
                        },
                        {
                            "name": "train_id",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                            "description": "Train ID to filter by (optional). Shows current trip if train is running, or next upcoming trip if not currently active. If null, returns all trains."
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Running trains retrieved successfully",
                            "example": {
                                "status": "success",
                                "data": {
                                    "query_time": "14:30:00",
                                    "station_filter": "ALVA",
                                    "train_id_filter": None,
                                    "running_trains": [
                                        {
                                            "train_id": "Train_1",
                                            "trip_id": "WK_15",
                                            "trip_status": "current",
                                            "current_trip": {
                                                "start_time": "14:25:00",
                                                "end_time": "15:45:00",
                                                "start_station": "ALVA",
                                                "end_station": "ERNAKULAM",
                                                "progress_percentage": 6.2
                                            },
                                            "current_location": "Currently at ALVA",
                                            "station_status": "at_station",
                                            "status": "active",
                                            "priority_info": {
                                                "priority_rank": 1,
                                                "priority_score": 0.481
                                            },
                                            "trip_details": {
                                                "total_stops": 8,
                                                "stops": [
                                                    {
                                                        "stop_id": "ALVA",
                                                        "stop_name": "Aluva",
                                                        "arrival_time": "14:25:00",
                                                        "departure_time": "14:25:00"
                                                    }
                                                ],
                                                "station_position": 1,
                                                "current_position": 1
                                            }
                                        }
                                    ],
                                    "total_running_trains": 1
                                },
                                "timestamp": "2025-01-17T14:30:00Z"
                            }
                        },
                        "400": {
                            "description": "Invalid time format or missing required parameter",
                            "example": {
                                "error": "time parameter is required (format: HH:MM:SS)"
                            }
                        },
                        "404": {
                            "description": "No schedule data available",
                            "example": {
                                "error": "No schedule data available. Please generate schedule first."
                            }
                        }
                    }
                }
            },
            "/api/maintenance/hub": {
                "get": {
                    "summary": "Maintenance Hub Status",
                    "description": "Get maintenance hub information and train allocations",
                    "responses": {
                        "200": {
                            "description": "Maintenance hub data retrieved",
                            "example": {
                                "status": "success",
                                "data": {
                                    "hubs": [
                                        {
                                            "hub_name": "ALVA",
                                            "trains_scheduled": [
                                                {
                                                    "train_id": "Train_25",
                                                    "arrival_time": "22:49:58",
                                                    "maintenance_type": "scheduled",
                                                    "status": "pending"
                                                }
                                            ],
                                            "capacity": 5,
                                            "current_occupancy": 1
                                        }
                                    ],
                                    "total_maintenance_trains": 4
                                }
                            }
                        }
                    }
                }
            },
            "/api/simulation/what-if": {
                "post": {
                    "summary": "What-If Simulation",
                    "description": "Run simulation for proposed schedule changes",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "scenario_name": {
                                            "type": "string",
                                            "description": "Name for the simulation scenario"
                                        },
                                        "changes": {
                                            "type": "object",
                                            "properties": {
                                                "type": {
                                                    "type": "string",
                                                    "enum": ["route_change", "maintenance_schedule", "time_adjustment"]
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Simulation completed successfully",
                            "example": {
                                "status": "success",
                                "data": {
                                    "scenario_name": "Route Optimization Test",
                                    "impact_analysis": {
                                        "affected_trains": 5,
                                        "schedule_conflicts": 1,
                                        "efficiency_change": 7.2,
                                        "recommendations": [
                                            "Consider adjusting departure times by 5-10 minutes"
                                        ]
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/notifications": {
                "get": {
                    "summary": "Get Notifications",
                    "description": "Retrieve system notifications and alerts",
                    "responses": {
                        "200": {
                            "description": "Notifications retrieved successfully",
                            "example": {
                                "status": "success",
                                "data": {
                                    "notifications": [
                                        {
                                            "id": "priority_25",
                                            "type": "maintenance",
                                            "severity": "high",
                                            "title": "Urgent Maintenance Required",
                                            "message": "Train 25 has priority rank 25 and requires immediate attention",
                                            "timestamp": "2025-01-17T10:30:00Z",
                                            "read": False
                                        }
                                    ],
                                    "unread_count": 3,
                                    "total_count": 8
                                }
                            }
                        }
                    }
                }
            },
            "/api/reports/performance": {
                "get": {
                    "summary": "Performance Report",
                    "description": "Generate performance analytics report",
                    "parameters": [
                        {
                            "name": "period",
                            "in": "query",
                            "schema": {
                                "type": "string",
                                "enum": ["daily", "weekly", "monthly"],
                                "default": "daily"
                            }
                        },
                        {
                            "name": "train_id",
                            "in": "query",
                            "schema": {"type": "string"},
                            "description": "Filter by specific train"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Performance report generated",
                            "example": {
                                "status": "success",
                                "data": {
                                    "period": "daily",
                                    "metrics": {
                                        "on_time_percentage": 94.2,
                                        "average_delay_minutes": 2.3,
                                        "total_trips_completed": 1050,
                                        "maintenance_compliance": 96.8
                                    },
                                    "summary": {
                                        "performance_grade": "A",
                                        "key_insights": [
                                            "On-time performance: 94.2%",
                                            "Average delay: 2.3 minutes"
                                        ]
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/export/schedule": {
                "get": {
                    "summary": "Export Schedule",
                    "description": "Export current schedule in various formats",
                    "parameters": [
                        {
                            "name": "format",
                            "in": "query",
                            "schema": {
                                "type": "string",
                                "enum": ["json", "csv", "excel"],
                                "default": "json"
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Schedule exported successfully",
                            "content": {
                                "application/json": {},
                                "text/csv": {},
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {}
                            }
                        }
                    }
                }
            }
        }
    }
    
    return api_docs

# HTML template for documentation
DOC_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .endpoint { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .method { display: inline-block; padding: 5px 10px; border-radius: 3px; color: white; font-weight: bold; }
        .get { background: #27ae60; }
        .post { background: #3498db; }
        .put { background: #f39c12; }
        .delete { background: #e74c3c; }
        .example { background: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto; }
        pre { margin: 0; }
        .parameter { background: #ecf0f1; padding: 8px; margin: 5px 0; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>{{ description }}</p>
        <p><strong>Version:</strong> {{ version }}</p>
    </div>
    
    {% for path, methods in paths.items() %}
        {% for method, details in methods.items() %}
        <div class="endpoint">
            <h3>
                <span class="method {{ method }}">{{ method.upper() }}</span>
                <code>{{ path }}</code>
            </h3>
            <p><strong>{{ details.summary }}</strong></p>
            <p>{{ details.description }}</p>
            
            {% if details.parameters %}
            <h4>Parameters:</h4>
            {% for param in details.parameters %}
            <div class="parameter">
                <strong>{{ param.name }}</strong> ({{ param.in }}) - {{ param.description or 'No description' }}
                {% if param.required %}<em>Required</em>{% endif %}
            </div>
            {% endfor %}
            {% endif %}
            
            {% if details.requestBody %}
            <h4>Request Body:</h4>
            <div class="example">
                <pre>{{ details.requestBody | tojson(indent=2) }}</pre>
            </div>
            {% endif %}
            
            {% if details.responses %}
            <h4>Response Example:</h4>
            {% for status, response in details.responses.items() %}
            <p><strong>{{ status }}:</strong> {{ response.description }}</p>
            {% if response.example %}
            <div class="example">
                <pre>{{ response.example | tojson(indent=2) }}</pre>
            </div>
            {% endif %}
            {% endfor %}
            {% endif %}
        </div>
        {% endfor %}
    {% endfor %}
    
    <div style="margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 5px;">
        <h3>Getting Started</h3>
        <ol>
            <li>Start the Flask server: <code>python app.py</code></li>
            <li>Check health: <code>GET /api/health</code></li>
            <li>Ensure your Excel file exists at <code>data/Final_kochi_1.xlsx</code></li>
            <li>Analyze train priorities using backend data: <code>POST /api/priority/analyze</code></li>
            <li>Generate schedule with worst-ranked trains selected for maintenance: <code>POST /api/schedule/generate</code></li>
            <li>Access dashboard data: <code>GET /api/dashboard</code></li>
            <li>Get maintenance recommendations: <code>GET /api/maintenance/recommendations</code></li>
            <li>Get running trains by time and location: <code>GET /api/trains/running?time=14:30:00&station=ALVA</code></li>
        </ol>
        
        <h3>Running Trains Endpoint Usage Examples</h3>
        <ul>
            <li><strong>All trains at specific time:</strong> <code>GET /api/trains/running?time=14:30:00</code></li>
            <li><strong>Trains at specific station at specific time:</strong> <code>GET /api/trains/running?time=14:30:00&station=ALVA</code> - Shows only trains currently at ALVA station or passing through ALVA at 14:30:00</li>
            <li><strong>Specific train status:</strong> <code>GET /api/trains/running?time=14:30:00&train_id=Train_5</code></li>
            <li><strong>Combined filters:</strong> <code>GET /api/trains/running?time=14:30:00&station=ERNAKULAM&train_id=Train_1</code></li>
        </ul>
        
        <h3>Station Filtering Behavior</h3>
        <p>When using the station parameter:</p>
        <ul>
            <li><strong>Current trains:</strong> Shows trains currently running through the specified station at the given time</li>
            <li><strong>Future trains:</strong> Shows trains scheduled to pass through the station at or after the given time</li>
            <li><strong>Station status:</strong> Indicates if train is 'at_station', 'approaching', 'departed', or 'scheduled'</li>
            <li><strong>Position tracking:</strong> Shows train's current position relative to the queried station</li>
        </ul>
        
        <h3>Backend Data Requirements</h3>
        <p>The system requires an Excel file at <code>data/Final_kochi_1.xlsx</code> with the following sheets:</p>
        <ul>
            <li><strong>Fitness_Performance</strong> - Train fitness scores, critical/non-critical systems, revenue, performance ratings</li>
            <li><strong>Maintenance_Issues</strong> - Maintenance requirements, cleaning needs, breakdown frequency, repair costs</li>
            <li><strong>Operational_Metrics</strong> - Efficiency scores, utilization rates, passenger load factors, service quality</li>
        </ul>
        
        <h3>Authentication</h3>
        <p>Currently, no authentication is required. In production, implement JWT or API key authentication.</p>
        
        <h3>Error Handling</h3>
        <p>All endpoints return consistent error responses:</p>
        <div class="example">
            <pre>{
  "error": "Error description",
  "message": "Detailed error message",
  "timestamp": "2025-01-17T10:30:00Z"
}</pre>
        </div>
    </div>
</body>
</html>
"""

def create_docs_app():
    """Create Flask app for serving documentation"""
    docs_app = Flask(__name__)
    
    @docs_app.route('/')
    def api_documentation():
        docs = generate_api_docs()
        return render_template_string(
            DOC_TEMPLATE,
            title=docs['info']['title'],
            description=docs['info']['description'],
            version=docs['info']['version'],
            paths=docs['paths']
        )
    
    @docs_app.route('/json')
    def api_docs_json():
        return json.dumps(generate_api_docs(), indent=2)
    
    return docs_app

def test_module():
    """Test function to ensure module loads correctly"""
    return "API Documentation module loaded successfully"

if __name__ == '__main__':
    docs_app = create_docs_app()
    docs_app.run(debug=True, port=5001)
