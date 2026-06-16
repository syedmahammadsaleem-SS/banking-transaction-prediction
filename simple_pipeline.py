"""
Simple Banking Pipeline - No matplotlib (avoids DLL security blocks)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import os

def generate_sample_data(n_samples=50000, n_features=200):
    np.random.seed(42)
    feature_names = [f'var_{i:03d}' for i in range(n_features)]
    X = np.random.randn(n_samples, n_features)
    weights = np.random.randn(10)
    signal = np.dot(X[:, :10], weights)
    prob = 1 / (1 + np.exp(-signal))
    prob = (prob - prob.min()) / (prob.max() - prob.min()) * 0.3 + 0.05
    target = np.random.binomial(1, prob)
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = target
    df['ID_code'] = [f'ID_{i:08d}' for i in range(n_samples)]
    return df[['ID_code'] + feature_names + ['target']]

def run_pipeline():
    print("="*60)
    print("BANKING TRANSACTION PREDICTION PIPELINE")
    print("="*60)
    
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    data_path = 'data/raw/train.csv'
    if os.path.exists(data_path):
        print("Loading existing data...")
        df = pd.read_csv(data_path)
    else:
        print("Generating sample data (50,000 customers)...")
        df = generate_sample_data(n_samples=50000)
        df.to_csv(data_path, index=False)
        print(f"Data saved to {data_path}")
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    feature_cols = [c for c in df.columns if c not in ['ID_code', 'target']]
    X = df[feature_cols]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    }
    
    results = []
    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'Model': name, 'ROC-AUC': round(auc,4), 'Accuracy': round(acc,4),
            'Precision': round(prec,4), 'Recall': round(rec,4), 'F1': round(f1,4)
        })
        
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['ROC-AUC'].idxmax()]
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best['Model']}")
    print(f"ROC-AUC: {best['ROC-AUC']}")
    print(f"{'='*60}")
    
    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    print(results_df.to_string(index=False))
    
    results_df.to_csv('reports/model_comparison.csv', index=False)
    print("\nResults saved to reports/model_comparison.csv")
    print("Pipeline complete!")

if __name__ == '__main__':
    run_pipeline()