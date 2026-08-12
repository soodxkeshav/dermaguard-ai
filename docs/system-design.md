# System Design

## User Flow

1. User uploads skin lesion image.
2. Frontend sends image to backend API.
3. AI model analyzes image.
4. Model predicts lesion category.
5. Risk level is generated.
6. Explanation is displayed.
7. Prediction is stored in database.

## Components

### Frontend
- React
- Image Upload
- Results Dashboard

### Backend
- FastAPI
- API Endpoints
- Prediction Service

### AI Model
- TensorFlow
- Skin Cancer Classifier

### Database
- SQLite
- Prediction History