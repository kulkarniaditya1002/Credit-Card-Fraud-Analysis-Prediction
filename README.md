
# **Credit Card Fraud Detection using Machine Learning**

## **Overview**
This project focuses on detecting fraudulent credit card transactions using various machine learning models. The dataset contains transactions made by European cardholders in September 2013, with only 492 fraudulent transactions out of 284,807 total transactions. Due to the significant class imbalance, sophisticated techniques were employed to ensure accurate and reliable predictions.

### **Dataset**
- **Transactions**: 284,807
- **Frauds**: 492 (0.172%)
- **Period**: Two days in September 2013
- **Features**: 30 total, including 28 principal components derived from PCA, and two original features: `Time` and `Amount`.
- **Class Label**: 
  - `1` for fraud
  - `0` for non-fraud

### **Metadata & Pre-processing**
- **Principal Components**: Features `V1` to `V28` are PCA-derived.
- **`Time`**: Seconds elapsed between each transaction and the first transaction in the dataset.
- **`Amount`**: Transaction cost.
- **`Class`**: Binary response variable indicating fraud (1) or not (0).
- **Data Imbalance**: Significant skew with frauds making up only 0.172% of the transactions.
- **Missing Values**: Verified for missing transactions using `colSums(is.na(df))`.
- **Class Distribution**: Checked using `table(df$Class)` to confirm imbalance.

### **Exploratory Data Analysis (EDA)**
- **Class Distribution**: Visualized the skewness in the data.
- **Time Distribution**: Analyzed transaction time by class.
- **Amount Distribution**: Compared transaction amounts by class.
- **Feature Correlation**: Utilized Pearson correlation to understand relationships between features.

### **Model Performance**
Three machine learning models were trained and evaluated:
| Model               | AUC-ROC Score  |
|---------------------|---------------|
| **XGBoost**         | 97.1%         |
| **Logistic Regression** | 97.1%     |
| **Random Forest**   | **97.7%**     |

#### **Top 3 Features (Importance Scores)**
- **V1**: 38%
- **V2**: 18%
- **V3**: 10%

### **Conclusion**
The project successfully tackles the challenges associated with imbalanced datasets, such as the fraud detection dataset where fraudulent cases are rare compared to normal transactions. 

#### **Key Takeaways:**
- **Metrics**: Accuracy isn't appropriate for imbalanced data; instead, AUC-ROC and F1 scores were used.
- **Sampling Techniques**: Oversampling provided significant improvement, leading to better model performance.
- **Model Performance**: 
  - The **Random Forest** model achieved the highest AUC-ROC score of 0.977, though XGBoost and Logistic Regression also performed well.
  - Tuning the Random Forest model's parameters may further improve performance.
  
This project underscores the importance of proper sampling, modeling, and evaluation techniques when dealing with imbalanced datasets.

---

## **Getting Started**

### **Prerequisites**
- Python 3.7+
- Libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

### **Installation**
Clone the repository and install the required packages:

```bash
git clone https://github.com/kulkarniaditya1002/Credit-Card-Fraud-Analysis-Prediction.git
cd Credit-Card-Fraud-Analysis-Prediction
pip install -r requirements.txt
```

### **Usage**
To run the analysis and train models:

```bash
python fraud_detection.py
```

## **Future Work**
- **Parameter Tuning**: Further tuning of the Random Forest model could yield even better results.
- **Feature Engineering**: Exploring additional feature engineering techniques may enhance model accuracy.
- **Advanced Techniques**: Incorporating ensemble methods or deep learning models to improve fraud detection.

## **Contributing**
Contributions are welcome! Please open an issue or submit a pull request for any improvements or features you'd like to see.


---
