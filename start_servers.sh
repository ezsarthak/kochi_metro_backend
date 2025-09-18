#!/bin/bash

echo "Starting Train Management System..."

echo "Starting API server with integrated documentation on port 5000..."
python app.py &
API_PID=$!

echo ""
echo "🚆 Train Management System is running!"
echo ""
echo "📊 Main API: http://localhost:5000"
echo "📚 API Documentation: http://localhost:5000/docs"
echo "📋 API JSON Schema: http://localhost:5000/docs/json"
echo ""
echo "Quick test: curl http://localhost:5000/api/health"
echo ""
echo "Press Ctrl+C to stop the server"

# Wait for interrupt
trap "echo 'Stopping server...'; kill $API_PID; exit" INT
wait
