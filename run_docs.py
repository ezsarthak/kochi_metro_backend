"""
Run the API documentation server
"""

from api_documentation import create_docs_app

if __name__ == '__main__':
    print("Starting API Documentation Server...")
    print("Documentation will be available at: http://localhost:5001")
    print("JSON format available at: http://localhost:5001/json")
    
    docs_app = create_docs_app()
    docs_app.run(debug=True, host='0.0.0.0', port=5001)
