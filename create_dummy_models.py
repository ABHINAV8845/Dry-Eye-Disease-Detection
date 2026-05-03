import os, joblib, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

genes = ['GAPDH','ACTB','PAX6','MUC1','TP53','EGFR','VEGFA','IL6','TNF','CXCL8']
os.makedirs('models', exist_ok=True)
X = np.random.rand(50, 10)
y = np.array([0]*25 + [1]*25)
scaler = StandardScaler().fit(X)
clf = LogisticRegression().fit(scaler.transform(X), y)
joblib.dump(clf, 'models/classifier.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(genes, 'models/selected_genes.joblib')
print("✅ Dummy .joblib files created successfully inside /models folder!")
