# Problem Statement
Analyse a [dataset](https://drive.google.com/drive/folders/1IxCF-QLFi02knTmUYllnPgedZNzXT_hy?usp=sharing) with a column called description, which is the POS stamp for the transaction, and two other columns for Category and Sub Category. 
Use a notebook of your choice (Jupyter, Google, etc.) and preferably Python as a coding language, to answer the following questions:

1. Given the transaction details columns, can you build a simple model that reads a portion of this data, and without seeing the hold out set, predicts the categories and subcategories? first start by predicting the category and see if you can expand.

2. Can you also build a model which takes in a whole new transaction description (none of those that your model has seen for training and testing), and gives the closest possible category?



# Journey

## 1. Intial Thoughts - 13/02/2024
Looking at the data, I notice that all the features are categorical, so alot of algorithms aren't possible unless I find an embedding of some kind, while there are pre-trained embeddings available, they may not work, or fine tuning might not be effective on such a small dataset. 

Immediately, I'm thinking I can finish a DT or random forrest using SK learn by tonight, and explore other methods tomorrow. For tomorrow, there are pre-trained VAEs (typically used for embeddings) that I can fine tune for this problem, . I'm cautious of using anything too complicated.

## 2. Data Exploration - 13/02/2024
With only 3 features, exploration is limited.
There are 400 rows, and almost exclusively unique transaction descriptions; mostly nonsense but some overlap in contents. 

looking at the 13 existing categories(keeping in mind unseen categories may exist), Entertainment and Food make up 80% of the data. this makes me worried that naive methods might not really work. Similarly Books make up 95% of the entertaiment category, while food is split 57:43 between groceries and fast food.

All subcategories, except Maintenance, are unique to their category.

Another thing I notices is that many of the descriptions have the same words, except in a different order or with additional words/characters. I could try finding a simple distance measure but not sure if that is any better than using an off the shelf model.

Given the skew in the data, I need to keep an eye on the false positive rate and the precision, and be consious of the fact that some categories may not have been included in this dataset (eg. entertainment:movies)

looking at these as a human, there is a lot of intuitive information in the descriptions, so a pretrained  model text classification model is probably going to work the best. DT and Forrests are not going to be robust to unseen data, or handle the skew well.


# Solution

