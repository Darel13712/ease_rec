#EASE recommender

This is a repository for a model from 
[Embarrassingly Shallow Autoencoders for Sparse Data](https://arxiv.org/abs/1905.03375).

This model is cool because it is a closed-form solution.

### Input
`pandas.DataFrame` with columns `user_id` and `item_id` both for fit and predict.
It may also use ratings from column `rating` if implicit parameter is set to `False`.

### Output
List of ratings