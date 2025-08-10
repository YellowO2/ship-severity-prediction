# Ship Deficiency Severity Prediction

A predictive model for a hackathon that classifies the severity of ship deficiencies using a hybrid machine learning approach. The model combines deep learning-based NLP with a gradient boosting classifier to achieve high accuracy on mixed data types.

## The Steps

1.  **Data Preprocessing:** 
	The initial dataset contained multiple severity ratings for single incidents. A "consensus severity" was derived for each incident to create a clean, reliable target label for training. The raw text from incident reports was also cleaned and standardized.

2.  **NLP Feature Extraction with DistilBERT:**
    -   A pre-trained **DistilBERT** model (a smaller, faster variant of BERT) was fine-tuned on the cleaned incident report texts.
    -   Instead of using the model for direct classification, it was used as a sophisticated feature extractor to generate rich, contextual **embeddings** (numerical representations) from the text.

3.  **Hybrid Model with XGBoost:**
    -   The text embeddings from DistilBERT were combined with the structured data from the dataset (e.g., ship age, vessel group).
    -   This combined feature set was then used to train an **XGBoost** classifier, which made the final prediction on the incident's severity.

## Tech Stack

- **Language & Environment:** Python, Jupyter Notebook
- **Data Handling:** Pandas, Scikit-learn
- **NLP / Deep Learning:** PyTorch, Hugging Face Transformers (DistilBERT)
- **Prediction Model:** XGBoost
