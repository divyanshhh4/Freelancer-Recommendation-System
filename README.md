# Freelancer Recommendation System

## Overview
The Freelancer Recommendation System is an AI-powered solution designed to match job requirements with the most suitable freelancers. Using machine learning techniques like content-based filtering and collaborative filtering, this system provides personalized recommendations for clients. The application is built with Python, Scikit-learn, and FastAPI and is deployed on a scalable platform.

---

## Model Selection and Training Process

### 1. Data Preprocessing
- The freelancer dataset and interaction data were cleaned and preprocessed.
- Freelancer skills, completed projects, and experience were vectorized using the **TF-IDF Vectorizer**.
- Interaction ratings were aggregated to create a matrix for collaborative filtering.

### 2. Content-Based Filtering
- **TF-IDF Vectorizer** was used to calculate similarity scores between freelancers' skills, projects, and experience against job requirements.
- Availability was scored based on compatibility with the job timeline.

### 3. Collaborative Filtering
- A **Truncated SVD model** was trained on the interaction matrix to predict freelancer ratings based on clients' past preferences.

### 4. Final Scoring
A weighted scoring system combined:
- Skills similarity (30%)
- Projects similarity (30%)
- Experience similarity (20%)
- Availability score (20%)

The top 5 freelancers with the highest final scores are returned as recommendations.

---

## API Functionality and How to Test It

### API Endpoints
#### Root Endpoint (`/`)
- Returns a message confirming that the API is running.

#### Recommendation Endpoint (`/recommend`)
- Accepts job details as input.
- Returns the top 5 recommended freelancers based on the input criteria.

### Input Format
The `/recommend` endpoint expects a JSON payload in the following format:
{
"skills": ["Python", "Machine Learning"],
"budget": 50,
"timeline": "2 weeks",
"client_id": "C1"
}

### Output Format
The API returns a JSON object containing the top 5 recommended freelancers:
{
"recommendations": [
{
"Freelancer_ID": "F1",
"Hourly_Rate": 45,
"Skills": ["Python", "Machine Learning"],
"Completed_Projects": ["Project A", "Project B"],
"Experience": ["Role X at Company Y"],
"Availability": "1 Week"
},
...
]
}

---

## Testing Instructions
1. Visit the hosted API link.
2. Navigate to `/docs` to access the interactive Swagger UI.
3. Expand the `/recommend` tab.
4. Click **Try it out**, fill in the input fields (`skills`, `budget`, `timeline`, `client_id`), and press **Execute**.
5. View the recommendations in the response section.

---

## Deployment Steps

### Deployment on Railway

#### Create a Railway Project:
1. Log in to your Railway account and create a new project.

#### Upload Files:
2. Upload all required files (`main.py`, `model.py`, `datasets`) to your Railway project repository.

#### Configure Settings:
3. Specify the runtime environment (`python 3.12.4`) in `runtime.txt`.
4. Install dependencies by including `requirement.txt` in your project.

#### Start Deployment:
5. Deploy your FastAPI application by linking it to your Railway project.
6. Railway will automatically build and host your application.

#### Access API:
7. Once deployed, access your hosted API link provided by Railway.
8. Test functionality using Swagger UI (`/docs`) or any HTTP client like Postman.
