# DocumentClassification
北航2020数据挖掘导论：DocumentClassification


In this case, we have two categories of emails, in which one category is about hockey and the other is about baseball. The data is in the folder classification.
1. Firstly preprocess the documents into numerical data (Record data). The preprocessing guidelines can be found in the introduction slides (SMO), consider using tf-idf
2. Use SVMs to classify the documents and test the classification results with 5-fold cross validation. You should report the precision, recall, and F1-measure of each fold and the average values. (Recommend LIBSVM to implement SVMs. You can refer to the tutorial slides in evaluating the results.)
3. Bonus (5 extra points). Implement Sequential Minimal Optimization (SMO) by following the introductive slides.
