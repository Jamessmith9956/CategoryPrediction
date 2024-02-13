# Problem Statement
Analyse a [dataset](https://drive.google.com/drive/folders/1IxCF-QLFi02knTmUYllnPgedZNzXT_hy?usp=sharing) with a column called description, which is the POS stamp for the transaction, and two other columns for Category and Sub Category. 
Use a notebook of your choice (Jupyter, Google, etc.) and preferably Python as a coding language, to answer the following questions:

1. Given the transaction details columns, can you build a simple model that reads a portion of this data, and without seeing the hold out set, predicts the categories and subcategories? first start by predicting the category and see if you can expand.

2. Can you also build a model which takes in a whole new transaction description (none of those that your model has seen for training and testing), and gives the closest possible category?



# Journey

## 1. Intial Thoughts - 13/02/2024
Looking at the data, I notice that all the features are categorical, so k-means and similar clustering algorithms aren't possible unless I find an embedding of some kind. 
That would take some time and might not work on a dataset of this size if I have to train my own embedding.

Immediately, I'm thinking I can finish a DT or random forrest using SK learn by tonight, and explore other methods tomorrow.

For tomorrow, there are pre-trained VAEs (typically used for embeddings) that I can fine tune for this problem, . I'm cautious of using anything too complicated for a dataset this small.



# Solution

