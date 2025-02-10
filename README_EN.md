# Bike-fit System

## Introduction
Bike-fit is an innovative bicycle fitting system designed to help cyclists achieve optimal riding posture and bicycle setup. Using advanced computer vision technology and professional cycling knowledge, it provides personalized bicycle adjustment recommendations.

## System Architecture
The project adopts a front-end and back-end separated architecture:
- Frontend (`frontend/`): User interface and interaction layer
- Backend (`backend/`): Business logic and data processing layer
- Demo Module (`demo/`): Feature demonstrations and test cases
- Data Model (`thunder model/`): System data model design

## Key Features
- Riding Posture Analysis
- Bicycle Parameter Measurement
- Personalized Adjustment Recommendations
- User Data Management
- Professional Report Generation

## Tech Stack
### Frontend
- HTML5 + CSS3
- JavaScript (ES6+)
- Bootstrap for responsive layout
- WebSocket for real-time communication
- File upload and image processing capabilities

### Backend
- Python 3.8+
- Flask Web Framework
- OpenCV for image processing
- MediaPipe for human pose detection
- NumPy for mathematical computations
- SQLite Database
- WebSocket for real-time communication

## Environment Requirements
### System Requirements
- Python 3.8 or higher
- pip package manager
- Modern browsers (Chrome, Firefox, Safari, etc.)
- Camera (for real-time pose analysis)

### Installation
1. Clone the project
```bash
git clone https://github.com/Gzmomo001/Bike-fit.git
cd Bike-fit
```

2. Install backend dependencies
```bash
cd backend
pip install -r requirements.txt
```

Main dependencies include:
- Web Framework:
  * Flask==2.0.1
  * Werkzeug==2.0.1
  * Jinja2==3.0.1

- Image Processing:
  * opencv-python==4.5.3.56
  * mediapipe==0.8.7.3
  * Pillow==8.3.2

- Scientific Computing:
  * numpy==1.21.2
  * scipy==1.7.1

- Real-time Communication:
  * websockets==10.0
  * python-socketio==5.4.0

- Database:
  * SQLAlchemy==1.4.23

For a complete list of dependencies, please check `backend/requirements.txt`

## Quick Start
### Frontend Setup
1. Navigate to frontend directory
```bash
cd frontend
```

2. Open main.html in browser
```bash
# MacOS
open main.html
# Linux
xdg-open main.html
# Windows
start main.html
# Or directly drag main.html into your browser
```

3. Frontend Features
- Homepage offers two modes: image upload and real-time camera analysis
- Supports JPG, PNG, JPEG image formats
- Recommend using clear side-view cycling photos for best analysis results
- Real-time analysis requires browser camera access permission

### Backend Setup
1. Create and activate virtual environment (recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# MacOS/Linux
source venv/bin/activate
```

2. Install Dependencies
```bash
cd backend
# Create requirements.txt if not exists
pip freeze > requirements.txt
# Install dependencies
pip install -r requirements.txt
```

3. Start Flask Server
```bash
# Development mode (recommended)
export FLASK_ENV=development  # MacOS/Linux
set FLASK_ENV=development    # Windows
python main.py

# Or direct start
python main.py
```

Server will start at http://localhost:5000 by default

### Installation Verification
1. Confirm Service Start
- Check if backend server is running (http://localhost:5000)
- Confirm no error messages in terminal
- Verify all dependencies are installed successfully

2. Feature Testing
- Image Upload Test:
  * Prepare a clear side-view cycling photo
  * Click "Upload Image" button
  * Wait for analysis results (typically 3-5 seconds)
  * View analysis report and recommendations

- Real-time Analysis Test:
  * Click "Enable Camera" button
  * Allow browser camera access
  * Adjust position to show complete side view of bicycle
  * View real-time analysis results

3. Common Issues Resolution
- If webpage fails to load, check browser console for error messages
- Ensure backend service is running properly
- Check if firewall allows local server access
- Ensure Python version compatibility (3.8+ recommended)

## Development Mode
For development:
1. Create new git branch
```bash
git checkout -b feature/your-feature-name
```

2. Enable debug mode
```bash
export FLASK_DEBUG=1  # MacOS/Linux
set FLASK_DEBUG=1    # Windows
python main.py
```

3. Real-time log viewing
```bash
# In another terminal window
tail -f backend/app.log
```

## Common Issues
1. If encountering dependency installation issues, ensure Python version compatibility
2. Make sure camera permissions are enabled
3. If image processing errors occur, check image format support

## Development Team
[To be added]

## License
[To be added]

## Contact
For any questions or suggestions, please contact us through:
- Email: [To be added]
- Project Repository: https://github.com/Gzmomo001/Bike-fit

## Backend Code Documentation
### Project Structure
```
backend/
├── __init__.py          # Package initialization
├── app.py              # Flask application configuration
├── main.py             # Main program entry
├── model.py            # Core model definition
├── preprocessing.py    # Image preprocessing module
├── postprocessing.py   # Result post-processing module
├── cropping.py         # Image cropping module
├── keypoints.py        # Keypoint detection module
├── unit_test.py        # Unit tests
└── requirements.txt    # Project dependencies
```

### Core Modules
#### 1. Main Program Entry (main.py)
- Flask application and route initialization
- HTTP request and WebSocket connection handling
- Module workflow coordination
- Main API endpoints:
  * `/upload`: Handle image uploads
  * `/analyze`: Posture analysis
  * `/realtime`: WebSocket real-time analysis
  * `/report`: Generate analysis reports

#### 2. Application Configuration (app.py)
- Flask application configuration
- Database connection setup
- Cross-Origin Resource Sharing (CORS) configuration
- Logging system setup
- Error handling mechanisms

#### 3. Core Model (model.py)
- Bicycle posture analysis core algorithm
- MediaPipe integration for human pose detection
- Key angle and distance calculations
- Adjustment recommendation generation
- Main features:
  * Pose detection
  * Angle calculation
  * Parameter measurement
  * Recommendation generation

#### 4. Image Processing
##### Preprocessing (preprocessing.py)
- Image format conversion
- Size adjustment
- Color space conversion
- Image enhancement
- Noise handling

##### Post-processing (postprocessing.py)
- Analysis result optimization
- Data filtering and smoothing
- Visualization generation
- Report formatting
- Recommendation optimization

##### Image Cropping (cropping.py)
- Automatic ROI (Region of Interest) detection
- Smart cropping
- Boundary handling
- Image alignment

#### 5. Keypoint Detection (keypoints.py)
- Human body keypoint detection
- Bicycle component recognition
- Coordinate transformation
- Feature point tracking

#### 6. Testing Module (unit_test.py)
- Unit test cases
- Integration tests
- Performance tests
- Edge case testing

### Data Flow
1. Image Input
   - Upload image or video stream input
   - Format validation and initialization

2. Preprocessing Stage
   - Image standardization
   - Quality optimization
   - Size adjustment

3. Core Processing
   - Human pose detection
   - Bicycle component recognition
   - Parameter calculation
   - Posture analysis

4. Post-processing Stage
   - Result optimization
   - Data validation
   - Recommendation generation
   - Report compilation

5. Output Results
   - JSON format data
   - Visualization results
   - Adjustment recommendations
   - Professional report

### API Documentation
#### 1. Image Analysis API
```http
POST /api/analyze
Content-Type: multipart/form-data

Parameters:
- image: Image file
- options: Analysis options (optional)

Response:
{
    "status": "success",
    "data": {
        "angles": {...},
        "distances": {...},
        "suggestions": [...]
    }
}
```

#### 2. Real-time Analysis API
```websocket
WebSocket: ws://localhost:5000/realtime

Message Format:
{
    "type": "frame",
    "data": "base64 encoded image data"
}

Response:
{
    "type": "analysis",
    "data": {
        "pose": {...},
        "suggestions": [...]
    }
}
```

### Error Handling
- Input validation errors
- Processing timeouts
- Resource unavailability
- Unsupported formats
- System errors

Each error type has corresponding error codes and handling mechanisms to ensure system stability and user experience.

### Performance Optimization
1. Image Processing Optimization
   - Optimized OpenCV algorithms
   - Image caching mechanism
   - Parallel processing

2. Real-time Analysis Optimization
   - Frame rate control
   - Data compression
   - Asynchronous processing

3. Resource Management
   - Memory usage optimization
   - CPU load balancing
   - Concurrent connection management