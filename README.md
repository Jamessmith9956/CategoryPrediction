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

## 3. Model Building - 13/02/2024
### Distance Measures
My basic model to vectorise the data without using any complex NN model.
I'm most familiar with cosine similarity, which requires I vectorise the data, typically through word frequency. 

other [options](https://flavien-vidal.medium.com/similarity-distances-for-natural-language-processing-16f63cd5ba55) I am less experienced using include: LCS, Edit Distance and Hamming Distance. 
Hamming and Edit distance out, however LCS might be usable with some finicking.

overall the efficeincy of this method is O(n^2), so another method will be needed to scale (DT, PageRank).

### basic vectorisers 
[Medium](https://medium.com/geekculture/how-sklearns-countvectorizer-and-tfidftransformer-compares-with-tfidfvectorizer-a42a2d6d15a2) provides good articles detailing the difference between basic vectorizers availiable in SK learn.
Count vectoriser: Tokenises the text and counts the frequency of each word.
Tfidf vectoriser: does the above + normalises the frequency of each word in the corpus.

Tfidf is better for our case as it will decrease the importance of common words. That said I'm mindful of the bias I'm intoducing.

## Data preperation
I've pretty much imediately hit a problem, there are 2 catgories with only 1 record, meaning I cannot seperate them into train test and val sets evenly. I have the option of a number of [oversampling methods](https://machinelearningmastery.com/data-sampling-methods-for-imbalanced-classification/), but I might save these for tomorrow and just drop the class for now. looking at the case, I could even generate new entries, but I would have to do the same for the subcategories with similar issues.

For now: Drop the categories and pick a oversampling method for tomorrow.


# Solutions

## Cosine Similarity
the heatmap from the basic cosine similarity on Tfidf vector representations was more sparse than I expected. I think this is mostly on the vectorisation and sampling method. 
![heatmap showing ./analysis/naive_cosine_similarity](./analysis/naive_cosine_similarity.png)

