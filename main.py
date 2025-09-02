from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import FreelancerRecommender

app = FastAPI()
recommender = FreelancerRecommender()

@app.on_event("startup")
def startup_event():
    try:
        freelancers_path = r"./Datasets/synthetic_freelancers_dataset.csv"
        interactions_path = r"./Datasets/interactions_data.csv"
        recommender.load_data(freelancers_path, interactions_path)
        recommender.preprocess()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Startup error: {str(e)}")

class JobDetails(BaseModel):
    skills: list[str]
    budget: float | None = None
    timeline: str | None = None
    client_id: str | None = None

@app.post("/recommend")
def get_recommendations(job: JobDetails):
    try:
        recommendations = recommender.recommend(
            required_skills=job.skills,
            job_budget=job.budget,
            timeline=job.timeline,
            client_id=job.client_id
        )

        if self.freelancers_df.empty:
            return {"message": "no data loaded"}
        
        if recommendations.empty:
            return {"message": "No freelancers match your criteria."}
            
        return {"recommendations": recommendations.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Freelancer Recommendation API is running!"}



