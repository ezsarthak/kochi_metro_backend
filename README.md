# Train Management System API

A comprehensive Flask-based API system for train scheduling, priority management, and operational analytics.

## Features

- **Priority Analysis**: ML-based train priority ranking using Excel data
- **Schedule Generation**: Optimized train scheduling with constraint satisfaction
- **Dashboard Analytics**: Real-time operational metrics and alerts
- **Maintenance Management**: Hub-based maintenance scheduling
- **What-If Simulation**: Test schedule changes before implementation
- **Performance Reports**: Detailed analytics and trend analysis
- **Export Capabilities**: Schedule export in multiple formats

## Quick Start

1. **Install Dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

2. **Start the System**
   \`\`\`bash
   chmod +x start_servers.sh
   ./start_servers.sh
   \`\`\`

3. **Access Services**
   - Main API: http://localhost:5000
   - Documentation: http://localhost:5001

## API Endpoints

### Core Operations
- `GET /api/health` - Health check
- `GET /api/dashboard` - Dashboard overview
- `POST /api/priority/analyze` - Analyze train priorities
- `POST /api/schedule/generate` - Generate optimized schedule

### Train Management
- `GET /api/trains` - List all trains
- `GET /api/trains/{train_id}` - Get train details
- `GET /api/maintenance/hub` - Maintenance hub status

### Analytics & Simulation
- `POST /api/simulation/what-if` - Run what-if scenarios
- `GET /api/reports/performance` - Performance analytics
- `GET /api/notifications` - System alerts

### Data Export
- `GET /api/export/schedule` - Export schedule (JSON/CSV/Excel)

## Usage Examples

### 1. Analyze Train Priorities
\`\`\`bash
curl -X POST -F "file=@train_data.xlsx" http://localhost:5000/api/priority/analyze
\`\`\`

### 2. Generate Schedule
\`\`\`bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"maintenance_slots": 4}' \
  http://localhost:5000/api/schedule/generate
\`\`\`

### 3. Get Dashboard Data
\`\`\`bash
curl http://localhost:5000/api/dashboard
\`\`\`

### 4. Run What-If Simulation
\`\`\`bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"scenario_name": "Route Test", "changes": {"type": "route_change"}}' \
  http://localhost:5000/api/simulation/what-if
\`\`\`

## Data Flow

1. **Upload Excel** → Priority Analysis → Train Rankings
2. **Priority Data** → Schedule Generation → Optimized Schedule
3. **Schedule** → Dashboard → Real-time Monitoring
4. **Changes** → What-If Simulation → Impact Analysis

## Architecture

\`\`\`
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask API      │    │   Algorithms    │
│   Dashboard     │◄──►│   (app.py)       │◄──►│   - Priority    │
│                 │    │                  │    │   - Scheduler   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   Data Storage   │
                       │   - JSON Files   │
                       │   - Excel Data   │
                       └──────────────────┘
\`\`\`

## Configuration

Key settings in `app.py`:
- `MAX_CONTENT_LENGTH`: File upload limit (16MB)
- `NUM_TRAINS`: Total trains in system (25)
- `HUB_STOPS`: Maintenance hub stations
- `STATION_ORDER`: Linear route definition

## Error Handling

All endpoints return consistent error responses:
\`\`\`json
{
  "error": "Error type",
  "message": "Detailed description",
  "timestamp": "2025-01-17T10:30:00Z"
}
\`\`\`

## Development

### Adding New Endpoints
1. Add route handler in `app.py`
2. Update documentation in `api_documentation.py`
3. Add utility functions in `utils.py`

### Testing
\`\`\`bash
# Health check
curl http://localhost:5000/api/health

# Get all trains
curl http://localhost:5000/api/trains

# Get specific train
curl http://localhost:5000/api/trains/Train_1
\`\`\`

## Production Deployment

1. **Security**: Add authentication (JWT/API keys)
2. **Database**: Replace JSON files with proper database
3. **Caching**: Add Redis for performance
4. **Monitoring**: Add logging and metrics
5. **Load Balancing**: Use Gunicorn + Nginx

## Support

For issues or questions:
- Check the API documentation at http://localhost:5001
- Review logs in the console output
- Ensure all dependencies are installed correctly
