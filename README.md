# Machine Learning R Project 

This was a project I did for a data mining class at TAMU, STAT639 with Professor Yang Ni. The gist of the first part of the project is as follows:

1. We are given the `class_data.RData` file
2. The `class_data` has an outcome vector `y`, as well as two design matrices, `x` and `x_new`.
3. `x` and `y` are training data 
4. We then are required to use `x` and `y` to train a set of classifiers, and finally make predictions using `x_new`
5. We then submit our predictions to the professor, and are scored on how accurate our classifications are

**NOTE:** We were not given the true class labels; there is no `y_new` data provided to us. The professor has that information privately, so as to mimick a real-world ML prediction problem. 

For part one of the project, I scored 6th out of 20 groups (I worked alone, as a singleton group). My estimated prediction error was somewhere around 22%, and my actual error was 16%. The lowest prediction error (that is, the best group's error) was 13%. I made sure in this part not to use any packages that do cross validation for me; I orginally intended to use some, but decided against it as I couldn't ensure the same cross-validation folds were being used across procedures. Moreover, it wasn't clear to me how some packages were doing their nested, repeat CV, and it also wasn't clear if any of the packages made parallel processing possible. So, ultimately I wrote my own nested, repeated k-fold cross validation scheme, all done in parallel. Lastly, for this part of the project, one area I could improve on is to do variable selection within each cross-validation set. Because of the high-dimensionality, including variable selection---either by trees or LASSO---during CV would likely have improved my prediction error. Additionally, I should have considered the Relaxed LASSO, but ran out of time. In any case, the ensemble methods likely would have outperformed the Relaxed LASSO.  

In the second part of the project, we had to cluster the `cluster_data`, and then decide based on our algorithm a number of clusters. For this, I used PCA and then KMeans, Gaussian mixtures, DBScan, and hierarchical clustering. However, this ended up being not the best approach, as a linear dimensionality reduction did not work out too well. Instead, what would have been better was t-distributed stochastic neighbor embeddings, as groups that did this recovered the correct number of clusters. I also attempted kernel PCA to address nonlinearity, but this did not offer much improvement. We were not told where we placed in this part of the project, as our reasoning behind the clustering decisions was what was important. But, given another opportunity, I would explore TSNE or other more recent algorithms. 

* See `final_pres.pdf` and `report.pdf` for more details, as well as the included R-script for code. 
