# EASE recommender

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/190503375/collaborative-filtering-on-netflix)](https://paperswithcode.com/sota/collaborative-filtering-on-netflix?p=190503375)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/190503375/collaborative-filtering-on-movielens-20m)](https://paperswithcode.com/sota/collaborative-filtering-on-movielens-20m?p=190503375)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/190503375/collaborative-filtering-on-million-song)](https://paperswithcode.com/sota/collaborative-filtering-on-million-song?p=190503375)


This is a repository for a model from 
[Embarrassingly Shallow Autoencoders for Sparse Data](https://arxiv.org/abs/1905.03375).

This model is cool because it is a closed-form solution.

Similiarity matrix is calculated on CPU with numpy.

### Input
`pandas.DataFrame` with columns `user_id` and `item_id` both for fit and predict.

It may also use ratings from column `rating` if `implicit` parameter is set to `False`.

### Output
`pandas.DataFrame` with columns `user_id, item_id, score`