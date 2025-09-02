import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

class FreelancerRecommender:
    def __init__(self):
        self.freelancers_df = None
        self.interactions_df = None
        self.vectorizer = None
        self.svd_model = None
        self.reconstructed_df = None

    def load_data(self, freelancers_path, interactions_path=None):
        try:
            self.freelancers_df = pd.read_csv(freelancers_path)
            if interactions_path:
                self.interactions_df = pd.read_csv(interactions_path)
        except FileNotFoundError as e:
            raise ValueError(f"Dataset error: {e}")

    def preprocess(self):
        # Preprocess freelancers data
        self.freelancers_df['Skills'] = self.freelancers_df['Skills'].str.split(', ')
        self.freelancers_df['Completed_Projects'] = self.freelancers_df['Completed_Projects'].str.split(', ')
        self.freelancers_df['Projects_Combined'] = self.freelancers_df['Completed_Projects'].apply(lambda x: ' '.join(x))
        self.freelancers_df['Experience'] = self.freelancers_df['Experience'].str.split(', ')

        # Train TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(min_df=1, max_df=0.8, ngram_range=(1, 1))
        self.skills_tfidf = self.vectorizer.fit_transform(
            self.freelancers_df['Skills'].apply(lambda x: ' '.join(x))
        )
        self.projects_tfidf = self.vectorizer.transform(self.freelancers_df['Projects_Combined'])
        self.experience_tfidf = self.vectorizer.transform(
            self.freelancers_df['Experience'].apply(lambda x: ' '.join(x))
        )

        # Collaborative filtering preprocessing
        if self.interactions_df is not None:
            # Handle duplicate ratings
            self.interactions_df = self.interactions_df.groupby(['Client_ID', 'Freelancer_ID'], as_index=False).mean()
            interaction_matrix = self.interactions_df.pivot(index='Client_ID', columns='Freelancer_ID', values='Rating').fillna(0)
            interaction_matrix_np = interaction_matrix.to_numpy()

            # Train SVD model
            self.svd_model = TruncatedSVD(n_components=5)
            latent_matrix = self.svd_model.fit_transform(interaction_matrix_np)
            reconstructed_matrix = self.svd_model.inverse_transform(latent_matrix)
            self.reconstructed_df = pd.DataFrame(
                reconstructed_matrix,
                index=interaction_matrix.index,
                columns=interaction_matrix.columns
            )

    def recommend(self, required_skills, job_budget=None, timeline=None, client_id=None):
        try:
            # Skills matching
            job_skills_tfidf = self.vectorizer.transform([" ".join(required_skills)])
            skills_similarity = cosine_similarity(job_skills_tfidf, self.skills_tfidf).flatten()
            self.freelancers_df['Skills_Similarity_Score'] = skills_similarity

            # Projects matching
            projects_similarity = cosine_similarity(job_skills_tfidf, self.projects_tfidf).flatten()
            self.freelancers_df['Projects_Similarity_Score'] = projects_similarity

            # Experience matching
            experience_similarity = cosine_similarity(job_skills_tfidf, self.experience_tfidf).flatten()
            self.freelancers_df['Experience_Similarity_Score'] = experience_similarity

            # Availability scoring
            def availability_score(freelancer_availability, timeline):
                if not timeline or not freelancer_availability:
                    return 0.5
                timeline_weeks = int(timeline.split()[0]) if "week" in timeline.lower() else int(timeline.split()[0]) * 4
                availability_weeks = int(freelancer_availability.split()[0]) if "week" in freelancer_availability.lower() else int(freelancer_availability.split()[0]) * 4
                return 1.0 if availability_weeks >= timeline_weeks else 0.2

            if timeline:
                self.freelancers_df['Availability_Score'] = self.freelancers_df['Availability'].apply(
                    lambda x: availability_score(x, timeline)
                )
            else:
                self.freelancers_df['Availability_Score'] = 0.5

            # Collaborative filtering integration
            cf_weight = 0.0  # Default weight
            if client_id and (self.reconstructed_df is not None) and (client_id in self.reconstructed_df.index):
                predicted_ratings = self.reconstructed_df.loc[client_id]
                predicted_ratings_normalized = predicted_ratings / (predicted_ratings.max() or 1)
                self.freelancers_df['Predicted_Rating'] = predicted_ratings_normalized.reindex(self.freelancers_df.index).fillna(0)
                cf_weight = 0.15  # 15% weight for collaborative filtering
            else:
                self.freelancers_df['Predicted_Rating'] = 0

            # Dynamic weighted scoring
            content_weight = 1 - cf_weight
            self.freelancers_df['Final_Score'] = (
                (0.35 * content_weight) * self.freelancers_df['Skills_Similarity_Score'] +
                (0.35 * content_weight) * self.freelancers_df['Projects_Similarity_Score'] +
                (0.20 * content_weight) * self.freelancers_df['Experience_Similarity_Score'] +
                (0.10 * content_weight) * self.freelancers_df['Availability_Score'] +
                cf_weight * self.freelancers_df['Predicted_Rating']
            )

            # Apply budget filter
            filtered_freelancers = self.freelancers_df.copy()
            if job_budget:
                filtered_freelancers = filtered_freelancers[filtered_freelancers['Hourly_Rate'] <= job_budget]

            # Return top 5 recommendations with all required columns
            return filtered_freelancers.nlargest(5, 'Final_Score')[[
                'Freelancer_ID', 'Hourly_Rate', 'Skills', 'Completed_Projects', 
                'Experience', 'Availability'
            ]]
        except Exception as e:
            print(f"Error in recommend(): {str(e)}")
            return pd.DataFrame()
