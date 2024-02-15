# Problem Statement
Analyse a [dataset](https://drive.google.com/drive/folders/1IxCF-QLFi02knTmUYllnPgedZNzXT_hy?usp=sharing) with a column called description, which is the POS stamp for the transaction, and two other columns for Category and Sub Category. 
Use a notebook of your choice (Jupyter, Google, etc.) and preferably Python as a coding language, to answer the following questions:

1. Given the transaction details columns, can you build a simple model that reads a portion of this data, and without seeing the hold out set, predicts the categories and subcategories? first start by predicting the category and see if you can expand.

2. Can you also build a model which takes in a whole new transaction description (none of those that your model has seen for training and testing), and gives the closest possible category?

# Results
1. Issues:
    - Lack of data has put subcategory prediction is out of reach.
    - Oversampling techniques need a minimum of a few entries to work effectively.
    - Synthetic data generation through LLMs is infeasible for me due to cost of compute.
2. Outcomes:
    - All models are only predicting the top 2 categories.
    - Subcategories could not be reliably predicted.
    - MiniLM embeddings instantly improved performance for even the worst models.
    - Predicting just 'Food', 'Entertainment' and 'Other' improved generalisation, but not anything else.
3. Prediction on unseen data:
    - Results on the test set imply that the performance does generalise to unseen, if the performance was there to begin with.
    - The models cannot come up with new categories for completely unseen data, but they will attempt to classify it as the next best category.


# Analysis Journey
## TL:DR - Full journal on main branch
- Intial Thoughts (13/02/2024): 
    - Identified that all features are categorical and considered using embeddings.
    - Planned to test basic sklearn models and pretrained VAEs.
    - Cautious of using anything too complicated.
- Data Exploration (13/02/2024):
    - Limited exploration due to only 3 features.
    - 400 rows with mostly unique transaction descriptions.
    - Skew in categories, with Entertainment and Food being dominant.
    - Noticed overlap in contents and subcategories unique to their category.
    - Considered using distance measures and pretrained text classification models.
- Model Building (13/02/2024):
    - Explored distance measures like cosine similarity.
    - Considered basic vectorizers like CountVectorizer and TfidfVectorizer.
    - Implemented basic SKlearn models to limited success (SVM, KNN, Bayes).
- Model Building (14/02/2024):
    - Explored LLM embedders like facebook/Data2Vec-text-base and RoBERTa.
    - Data preparation challenges with imbalanced classes.
    - Explored undersampling and oversampling techniques.
    - Tried SMOTE and ADASYN for oversampling.
- Model Building (15/02/2024):
    - Attempted generating synthetic data through LLM queries.
    - Attempted class weights when oversampling techniques didn't work.
    - Subcategory prediction was not impressive.
- Results (15/02/2024):
    - Models generalised their performance well for the test set, and were able to classify unseen categories to the 'next best' category.
    - Models performance was not good however, at best I was able to predict 4 categories with a precision of 0.33.
    - KNN was compettive with SVM when using embeddings, so the majority of future improments will be in the data prep.
