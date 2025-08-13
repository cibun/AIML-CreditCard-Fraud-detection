# Credit Card Fraud Detection Project
## Overview
This project demonstrates a complete workflow for credit card fraud detection using Python. It covers data analysis, handling class imbalance, visualization, and machine learning model comparison. The solution is designed to be professional, reproducible, and ready for public sharing on GitHub.

---

## Project Structure
```
Fraud Detection/
├── creditcard/
│   └── creditcard.csv
├── Images/
│   ├── Screenshot 2025-08-14 010810.jpg
│   └── Compare Analysis.png
├── src/
│   ├── frauddetection.py
│   └── README.md
```

---

## Step-by-Step Guide

### 1. Data Loading
- The dataset (`creditcard.csv`) is loaded using pandas.
- Basic info and class distribution are printed.

```
import pandas as pd
...
df = pd.read_csv('src/creditcard/creditcard.csv')
print('Data shape:', df.shape)
print(df['Class'].value_counts())
```

---

### 2. Exploratory Data Analysis (EDA)
- Visualize class imbalance using a count plot.
- Apply PCA for dimensionality reduction and visualize data distribution.

![Class Distribution](../Images/Screenshot%202025-08-14%20010810.jpg)
![PCA Scatter](../Images/Compare%20Analysis.png)

---

### 3. Handling Imbalanced Data
- Use Random Over Sampling (ROS), Random Under Sampling (AOS), and SMOTE to balance the dataset.
- Visualize the effect of each technique using scatter plots.


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

![Model Comparison Table](../Images/Compare%20Analysis.png)
---

### 5. Results & Recommendation
- The script prints the best model for each metric and the overall best model.


---

## Screenshots & Images
Below are key images from the analysis for better readability and understanding:
- **Class Distribution:**
  ![Class Distribution](../Images/Screenshot%202025-08-14%20010810.jpg)
- **PCA & Sampling Comparison:**
  ![PCA and Sampling Comparison](../Images/Compare%20Analysis.png)
- **Model Comparison Table:**
  ![Model Comparison Table](../Images/Compare%20Analysis.png)

---

## Quick Start: How to Run This Project

1. **Clone the Repository**
   ```bash
   git clone https://github.com/cibun/AIML-CreditCard-Fraud-detection.git
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
   - Download `creditcard.csv` from [the project)
   - Place it in the `creditcard/` folder.

6. **Run the Analysis Script**
   ```bash
   python src/frauddetection.py
   ```

7. **View Results and Plots**
   - The script will print model comparison results and recommendations in the terminal.
   - Plots will be displayed in pop-up windows (ensure your environment supports GUI for matplotlib).

---

### Troubleshooting
- If you encounter memory or performance issues, try running on a smaller sample of the data or reduce model complexity (e.g., set `n_estimators=50` for RandomForest/XGBoost).
- For Jupyter Notebook users, you can copy code blocks into a notebook for stepwise execution and visualization.

---

## Notes
- The solution is designed for large, imbalanced datasets.
- All code is commented and modular for easy understanding.
- Visualizations and results are saved for reporting and sharing.

---

## License
This project is open-source and free to use and modify.

---

## Author
Created by [Biswakesan Swain].

---

## Contributing
Pull requests and issues are welcome!

---

## References

- [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)

---

## Detailed Step-by-Step Workflow

### Data Cleaning
- After loading, all rows with missing values are dropped to ensure clean input for analysis and modeling. This prevents errors in PCA and model training.

### Feature Selection
- All columns except 'Class' are used as features for PCA and model training. 'Class' is the target variable indicating fraud or not.

### Sampling Techniques Explained
- **Random Over Sampling (ROS):** Duplicates minority class samples to balance the dataset.
- **Random Under Sampling (AOS):** Removes majority class samples to balance the dataset.
- **SMOTE:** Synthesizes new minority class samples using nearest neighbors.
- These techniques help address severe class imbalance, which is common in fraud detection.

### Visualization
- PCA is used to reduce feature space to 2D for visualization.
- Side-by-side scatter plots show the effect of each sampling technique on the data distribution.

### Model Training
- Both RandomForest and XGBoost classifiers are trained using imbalanced-learn pipelines, which apply sampling before fitting the model.
- Train/test split is stratified to preserve class proportions.

### Evaluation Metrics
- Models are evaluated using:
  - **Accuracy:** Overall correctness (can be misleading for imbalanced data).
  - **ROC-AUC:** Ability to distinguish between classes.
  - **Precision:** Proportion of predicted frauds that are actual frauds.
  - **Recall:** Proportion of actual frauds detected.
  - **F1-score:** Harmonic mean of precision and recall.
- These metrics are printed for each model and sampling technique.

### Automated Recommendation
- The script prints the best model for each metric.
- The overall best model is recommended based on the majority of metrics (most wins).

### Screenshots
- Each image in the Images folder is referenced in the workflow and described for clarity.

### Advanced Usage
- Users can extend the script to include more models, metrics, or sampling techniques.
- For large datasets, cloud-based Jupyter environments are recommended for faster computation and visualization.

### Contact & Support
- For questions, feedback, or collaboration, reach out via GitHub Issues or LinkedIn.

---
