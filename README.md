# Credit Card Fraud Detection Project

![Credit Card Fraud Detection](https://raw.githubusercontent.com/github/explore/main/topics/fraud-detection/fraud-detection.png)

## Overview
This project demonstrates a complete workflow for credit card fraud detection using Python. It covers data analysis, handling class imbalance, visualization, and machine learning model comparison. The solution is designed to be professional, reproducible, and ready for public sharing on GitHub.

---

## Project Structure
```
Fraud Detection/
├── creditcard/
│   └── creditcard.csv
├── frauddetection.py
└── README.md
```

---

## Step-by-Step Guide

### 1. Data Loading
- The dataset (`creditcard.csv`) is loaded using pandas.
- Basic info and class distribution are printed.

```
import pandas as pd
...
df = pd.read_csv('creditcard/creditcard.csv')
print('Data shape:', df.shape)
print(df['Class'].value_counts())
```

---

### 2. Exploratory Data Analysis (EDA)
- Visualize class imbalance using a count plot.
- Apply PCA for dimensionality reduction and visualize data distribution.

![Class Distribution](screenshots/class_distribution.png)
![PCA Scatter](screenshots/pca_scatter.png)

---

### 3. Handling Imbalanced Data
- Use Random Over Sampling (ROS), Random Under Sampling (AOS), and SMOTE to balance the dataset.
- Visualize the effect of each technique using scatter plots.

![Sampling Comparison](screenshots/sampling_comparison.png)

---

### 4. Model Training & Comparison
- Train RandomForest and XGBoost classifiers using imbalanced-learn pipelines.
- Compare models using Accuracy, ROC-AUC, F1-score, and Recall.
- Print a summary table and recommend the best model based on majority metrics.

```
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
...
results_df = pd.DataFrame(results)
print(results_df)
```

---

### 5. Results & Recommendation
- The script prints the best model for each metric and the overall best model.

![Model Comparison Table](screenshots/model_comparison.png)

---

## Screenshots & Images
Add the following images to a `screenshots/` folder:
- `class_distribution.png`: Bar plot of class distribution.
- `pca_scatter.png`: PCA scatter plot of original data.
- `sampling_comparison.png`: 2x2 grid of scatter plots for Original, ROS, AOS, SMOTE.
- `model_comparison.png`: Screenshot of the model comparison table output.

---


## Quick Start: How to Run This Project

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Install Python (Recommended: Python 3.8+)**
   - Download and install from [python.org](https://www.python.org/downloads/).

3. **Create a Virtual Environment (Optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install Dependencies**
   ```bash
   pip install pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
   ```

5. **Add the Dataset**
   - Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Place it in the `creditcard/` folder.

6. **Run the Analysis Script**
   ```bash
   python frauddetection.py
   ```

7. **View Results and Plots**
   - The script will print model comparison results and recommendations in the terminal.
   - Plots will be displayed in pop-up windows (ensure your environment supports GUI for matplotlib).

---

### Troubleshooting
- If you encounter memory or performance issues, try running on a smaller sample of the data or reduce model complexity (e.g., set `n_estimators=50` for RandomForest/XGBoost).
- For Jupyter Notebook users, you can copy code blocks into a notebook for stepwise execution and visualization.

---

---

## Notes
- The solution is designed for large, imbalanced datasets.
- All code is commented and modular for easy understanding.
- Visualizations and results are saved for reporting and sharing.

---

## License
This project is open-source and free to use under the MIT License.

---

## Author
Created by [Your Name].

---

## Contributing
Pull requests and issues are welcome!

---

## References
- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
