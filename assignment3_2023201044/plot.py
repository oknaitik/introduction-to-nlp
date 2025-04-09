import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_plots(model_similarities, human_scores, spearman_corr):
    # Plot Cosine Similarity vs Human Mean Scores
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=model_similarities, y=human_scores, alpha=0.7)
    sns.regplot(x=model_similarities, y=human_scores, scatter=False, color="red")  # Trendline
    plt.xlabel("Model Cosine Similarity")
    plt.ylabel("Human Mean Score")
    plt.title(f"Cosine Similarity vs Human Mean (Spearman: {spearman_corr:.4f})")

    # Save the figure
    save_path = "."
    cos_sim_plot_path = f"{save_path}/cosine_vs_human.png"
    plt.savefig(cos_sim_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {cos_sim_plot_path}")
    plt.show()

    # Plot Ranked Data for Spearman Correlation
    ranked_model_sim = pd.Series(model_similarities).rank().tolist()
    ranked_human_scores = pd.Series(human_scores).rank().tolist()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=ranked_model_sim, y=ranked_human_scores, alpha=0.7)
    sns.regplot(x=ranked_model_sim, y=ranked_human_scores, scatter=False, color="red")  # Trendline
    plt.xlabel("Ranked Model Similarity")
    plt.ylabel("Ranked Human Mean Score")
    plt.title(f"Spearman Rank Correlation (ρ = {spearman_corr:.4f})")

    # Save the figure
    spearman_rank_plot_path = f"{save_path}/spearman_rank.png"
    plt.savefig(spearman_rank_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {spearman_rank_plot_path}")
    plt.show()