# EDA libraries import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
# Machine Learning Libraries import
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from imblearn.pipeline import Pipeline


# Load the dataset
df = pd.read_csv('creditcard/creditcard.csv')

# Basic info
print('Data shape:', df.shape)
print(df['Class'].value_counts())

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
#plt.show()


# PCA for 2D visualization
features = [col for col in df.columns if col not in ['Class']]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df[features])
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Class'] = df['Class']

# Random Over Sampling (ROS)
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(df[features], df['Class'])
X_ros_pca = pca.transform(X_ros)
df_ros_pca = pd.DataFrame(X_ros_pca, columns=['PC1', 'PC2'])
df_ros_pca['Class'] = y_ros

# Random Under Sampling (AOS)
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(df[features], df['Class'])
X_rus_pca = pca.transform(X_rus)
df_rus_pca = pd.DataFrame(X_rus_pca, columns=['PC1', 'PC2'])
df_rus_pca['Class'] = y_rus

# SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(df[features], df['Class'])
X_smote_pca = pca.transform(X_smote)
df_smote_pca = pd.DataFrame(X_smote_pca, columns=['PC1', 'PC2'])
df_smote_pca['Class'] = y_smote

# Side-by-side scatter plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.scatterplot(ax=axes[0,0], x='PC1', y='PC2', hue='Class', data=df_pca, alpha=0.5)
axes[0,0].set_title('Original Data')
axes[0,0].set_xlabel('PC1')
axes[0,0].set_ylabel('PC2')

sns.scatterplot(ax=axes[0,1], x='PC1', y='PC2', hue='Class', data=df_ros_pca, alpha=0.5)
axes[0,1].set_title('Random Over Sampling (ROS)')
axes[0,1].set_xlabel('PC1')
axes[0,1].set_ylabel('PC2')

sns.scatterplot(ax=axes[1,0], x='PC1', y='PC2', hue='Class', data=df_rus_pca, alpha=0.5)
axes[1,0].set_title('Random Under Sampling (AOS)')
axes[1,0].set_xlabel('PC1')
axes[1,0].set_ylabel('PC2')

sns.scatterplot(ax=axes[1,1], x='PC1', y='PC2', hue='Class', data=df_smote_pca, alpha=0.5)
axes[1,1].set_title('SMOTE')
axes[1,1].set_xlabel('PC1')
axes[1,1].set_ylabel('PC2')

for ax in axes.flat:
    ax.legend(title='Class')


plt.tight_layout()
#plt.show()

# Machine Learning Comparison: RandomForest vs XGBoost with SMOTE and ROS
results = []

X = df[features]
y = df['Class']

for sampler_name, sampler in [('SMOTE', SMOTE(random_state=42)), ('ROS', RandomOverSampler(random_state=42))]:
    for clf_name, clf in [
        ('RandomForest', RandomForestClassifier(random_state=42)),
        ('XGBoost', XGBClassifier(eval_metric='logloss', random_state=42))
    ]:
        pipe = Pipeline([
            ('sampler', sampler),
            ('clf', clf)
        ])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, 'predict_proba') else None
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append({
            'Sampler': sampler_name,
            'Classifier': clf_name,
            'Accuracy': acc,
            'ROC-AUC': roc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1
        })

# Show results
results_df = pd.DataFrame(results)
print('\nModel Comparison:')
print(results_df)


# Recommend best model based on multiple metrics

metrics = ['Accuracy', 'ROC-AUC', 'F1-score', 'Recall']
best_models = []
for metric in metrics:
    if metric in results_df.columns:
        best_row = results_df.loc[results_df[metric].idxmax()]
        best_models.append((metric, best_row['Classifier'], best_row['Sampler'], best_row[metric]))
        print(f"\nBest Model by {metric}: {best_row['Classifier']} with {best_row['Sampler']} ({metric}: {best_row[metric]:.4f})")

# Summarize overall best model by majority of metrics
from collections import Counter
model_votes = Counter([(m[1], m[2]) for m in best_models])
overall_best = model_votes.most_common(1)[0]
print(f"\nOverall Best Model (by majority of metrics): {overall_best[0][0]} with {overall_best[0][1]} (Votes: {overall_best[1]})")
