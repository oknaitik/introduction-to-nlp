import pandas as pd
import os

def save_results(word_pairs, human_scores, model_similarities, save_path='./'):
    """Save evaluation results to a CSV file."""
    
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    # Create a DataFrame with results
    results_df = pd.DataFrame({
        "Word 1": [pair[0] for pair in word_pairs],
        "Word 2": [pair[1] for pair in word_pairs],
        "Human (Mean)": human_scores,
        "Model Similarity": model_similarities
    })
    
    csv_path = os.path.join(save_path, "evaluation_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    print(f"Results saved to: {csv_path}")
