import os
from src.preprocess import extract_text
from src.embeddings import get_similarity_scores
from src.scorer import calculate_weighted_score
from src.reranker import get_llm_evaluation # Assuming this exists

def run_ranking_pipeline(job_description, resume_path):
    # 1. Extraction
    text = extract_text(resume_path)
    
    # 2. Semantic Score (Fast Filter)
    # Returns a score between 0 and 1
    semantic_score = get_similarity_scores(job_description, text)
    
    # 3. Weighted Calculation
    # We assign weights based on the formula we discussed
    final_score = calculate_weighted_score(
        semantic_score=semantic_score,
        skill_score=0.85, # In a full system, parse this from text
        experience_years=5,
        target_years=5
    )
    
    # 4. LLM Re-Ranking (The "Judge")
    # Only run this if the semantic score is above a threshold (e.g., 0.6)
    if semantic_score > 0.6:
        evaluation = get_llm_evaluation(job_description, text, "Candidate_ID_001")
        return evaluation
    
    return {"score": final_score, "status": "Filtered Out"}

# Example Usage
if __name__ == "__main__":
    JD = "We need a Senior Python Developer with experience in AWS and Fintech."
    RESUME = "data/uploads/alice_resume.pdf"
    
    result = run_ranking_pipeline(JD, RESUME)
    print(result)