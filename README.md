
# 1. Problem Statement
This project focuses on predicting the price range of mobile phones based on their technical and hardware features. The aim is to build and compare several machine learning models that can classify a mobile phone into one of four price categories: low, medium, high, or very high.

Mobile price prediction is a classification problem, and it is useful for understanding how device specifications such as RAM, battery power, camera quality, screen size, connectivity, and memory influence the overall price of the phone. This project uses a Streamlit-based application to load trained models and compare their performance on the same dataset.

# 2. Dataset Description
The dataset used in this project is the Mobile Price Classification dataset. It contains multiple mobile phone features `(20)` and a target label named **_price_range_**.

| Category |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Input Features | battery_power | blue | clock_speed | dual_sim | fc | four_g | int_memory | m_dep | mobile_wt | n_cores | pc | px_height | px_width | ram | sc_h | sc_w | talk_time | three_g | touch_screen | wifi |
| Target Variable | price_range (4classes) | | | | | | | | | | | | | | | | | | | |


- 0 = Low cost
- 1 = Medium cost
- 2 = High cost
- 3 = Very high cost

The dataset file used in this project is: mobile_price_class_train.csv. This available publicly in kaggle (https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data?select=train.csv).

# 3. GitHub Repository Link
https://github.com/NMTechie/MLStreamLitApp

# 4. Models Used
The following five machine learning models were trained and evaluated:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN). Taken k=5 while training.
- Naive Bayes
- Support Vector Machine (SVM)
  
These models were trained on the same dataset and tested using common evaluation metrics to maintain fairness in comparison.

### Comparison Table
| Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC Score | 
|---|---|---|---|---|---|---| 
| Logistics Regression | 0.9745 | 0.9991 | 0.9745 | 0.9745 | 0.9744 | 0.9661 |
| Decision Tree | 0.9660 | 0.9773 | 0.9661 | 0.9660 | 0.9660 | 0.9547|
| KNN | 0.6610 | 0.8745 | 0.6707 | 0.6610 | 0.6618 | 0.5498 |
| Naive Bayes | 0.8170 | 0.9544 | 0.8178 | 0.8170 | 0.8173 | 0.7560 |
| Support Vector Machine | 0.9650 | 0.9896 | 0.9653 | 0.9650 | 0.9651 | 0.9534 |

# 5. Observations on Model Performance
Model Name | Observation about model performance|
|---|---|
| Logistic Regression | Top performer across every metric — Accuracy 97.45%, AUC 0.9991, F1 0.9744, MCC 0.9661. Precision and Recall are nearly identical (0.9745 each), meaning it rarely favors one price class over another and misclassifications are minimal and evenly spread. Its near-perfect AUC shows the four price classes are almost linearly separable using the given hardware features (notably RAM, battery_power, px_height/px_width), which plays directly to Logistic Regression's strength as a linear classifier. |
| Decision Tree | Second-best overall (Accuracy 96.60%, F1 0.9660, MCC 0.9547), with Precision and Recall tightly matched, indicating consistent behavior across classes. However, its AUC (0.9773) trails Logistic Regression and SVM by a noticeable margin — trees produce less smooth probability estimates, so while hard class predictions are strong, confidence ranking between classes is weaker. This also hints at mild overfitting to training-data splits, a known trait of single decision trees versus ensemble or margin-based methods. |
| KNN | Clear outlier with the weakest results (Accuracy 66.10%, MCC 0.5498) — roughly 30 points below every other model. The relatively higher AUC (0.8745) compared to its accuracy suggests the model can rank classes reasonably but struggles to pick the correct final class, likely because distance calculations are dominated by unscaled, large-range features like battery_power and px_height/px_width, drowning out smaller-scale but informative features like m_dep or clock_speed. This points to a feature-scaling gap rather than a fundamental flaw in the KNN approach. |
| Naive Bayes | Mid-tier performance (Accuracy 81.70%, MCC 0.7560), sitting clearly behind the top three but well ahead of KNN. Its AUC (0.9544) is relatively strong despite the accuracy gap, meaning the model ranks classes reasonably well but the independence assumption (treating features like ram, px_height, and px_width as uncorrelated) causes probability estimates to be miscalibrated at the decision boundary, which costs it accuracy, precision, and recall uniformly (all ~0.817). |
| Support Vector Machine | Near-identical to the top two models (Accuracy 96.50%, AUC 0.9896, MCC 0.9534), with Precision/Recall balance just as tight as Logistic Regression's. This confirms the price classes have well-defined, largely separable boundaries, since SVM's margin-maximizing approach performs almost as well as a simple linear model — reinforcing that the dataset's relationship between hardware specs and price_range is close to linear or cleanly separable with minimal nonlinear complexity. |
| Overall, Winner for Your Dataset? | **Winner: Logistic Regression.** It leads on every single metric (Accuracy, AUC, Precision, Recall, F1, MCC) rather than just one — unlike Decision Tree and SVM, which are strong but trail slightly on AUC/MCC. Combined with its simplicity, interpretability, and low computational cost, Logistic Regression is the most reliable and practical choice for deployment on this dataset. |




# Reference
bare minimum list of packages only not the transitive ones
```python
pip list --not-required --format=freeze > requirements.txt
```

With all dependencies 
```python
pip freeze > requirements.txt
```
This is best interms automation to scan the py files and actually built requirements.txt with needed one with exact version as well for reproducibility 
```python
pip install pipreqs
pipreqs . --force --ignore streamlitenv
```



