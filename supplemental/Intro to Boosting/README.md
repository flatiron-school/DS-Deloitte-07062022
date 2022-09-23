# Boosting

## Learning Goals

- describe boosting algorithms
- implement boosting models with `sklearn` and with `XGBoost`

## Lecture Materials

Jupyter Notebooks (there are two, unclear which is intended to be the lecture notebook):
- [Boosting Illustration](boosting_illustration.ipynb)
- [Boosting in SKLearn](boosting_in_sklearn.ipynb)

## Lesson Plan

### Introduction (5 Mins)

Once again, boosting is an extremely popular and powerful idea. Boosters routinely win modeling competitions etc. Two types, but AdaBoost is more of historical than practical value.

### Adaboost Algorithm (10 Mins)

Two series of weights: one on the data points (larger weights on points wrongly classified) and another on the models (larger weights on more accurate models).

### Gradient Boosting Algorithm (10 Mins)

More efficient idea: Train a model on the last model's *residuals*: If we now take the first model's predicions and add to them predictions about how that model *misses the mark*, we should have a superior model.

### Boosting Illustration Notebook (15 Mins)

This notebook walks through the idea of successively fitting models to previous models' residuals. Using the function at the end you can loop through the algorithm as many times as you like.

### Further Illustration of Gradient Boosting (10 Mins)

The `gradient_boost_and_plot()` function shows residuals and ensemble predictions as we repeat the algorithm.

### XGBoost (5 Mins)

Separate library for boosting. Syntax is different from but similar to `sklearn`'s. The algorithm is more sophisiticated in terms of regularization.

### Conclusion (5 Mins)

End of Phase 3 content!

## Tips

Greg generally starts with the "Boosting in SKLearn" ntbk, gets through the description of the algorithm of gradient boosting, hops over to the "Boosting Illustration" ntbk (since that is an application of gradient boosting), and then finishes up the "SKLearn" ntbk. The lesson plan above assumes this workflow.

# Additional Resources

## Lecture Supplements

[This post](https://towardsdatascience.com/a-brief-introduction-to-xgboost-3eaee2e3e5d6) may be helpful for describing the difference between standard and "extreme" gradient boosting.