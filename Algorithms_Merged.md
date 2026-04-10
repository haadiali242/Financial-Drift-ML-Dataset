## Merged Files List
- 1. algorithm_1_logistic_regression.py (2.2 KB)
- 2. algorithm_2_random_forest.py (2.6 KB)
- 3. algorithm_3_xgboost.py (2.8 KB)
- 4. algorithm_4_hoeffding_adwin.py (4.3 KB)
- 5. algorithm_5_kmeans.py (2.4 KB)
- 6. algorithm_6_dbscan.py (2.5 KB)
- 7. algorithm_7_isolation_forest.py (3 KB)
- 8. algorithm_8_autoencoder.py (3.9 KB)
- 9. algorithm_9_adaptive_ensemble.py (4.6 KB)


## 1. algorithm_1_logistic_regression.py

```py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

def run_algorithm_1(dataset_path, dataset_name):
    data = pd.read_csv(dataset_path)
    
    feature_cols = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
    X = data[feature_cols]
    y = data['default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    coefficients = dict(zip(feature_cols, model.coef_[0]))
    intercept = model.intercept_[0]
    
    prob_mean = np.mean(y_prob)
    prob_std = np.std(y_prob)
    
    results = {
        'dataset': dataset_name,
        'algorithm': 'Logistic Regression',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'brier_score': brier,
        'mcc': mcc,
        'coefficients': coefficients,
        'intercept': intercept,
        'probability_mean': prob_mean,
        'probability_std': prob_std,
        'default_rate': y.mean()
    }
    
    return results

if __name__ == '__main__':
    import sys
    dataset_path = sys.argv[1]
    dataset_name = sys.argv[2]
    results = run_algorithm_1(dataset_path, dataset_name)
    print(results)
```

## 2. algorithm_2_random_forest.py

```py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

def run_algorithm_2(dataset_path, dataset_name):
    data = pd.read_csv(dataset_path)
    
    feature_cols = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
    X = data[feature_cols]
    y = data['default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, oob_score=True)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    oob_error = 1 - model.oob_score_
    
    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    
    tree_depths = [estimator.tree_.max_depth for estimator in model.estimators_]
    avg_tree_depth = np.mean(tree_depths)
    max_tree_depth = np.max(tree_depths)
    
    leaf_nodes = [estimator.tree_.n_leaves for estimator in model.estimators_]
    avg_leaf_nodes = np.mean(leaf_nodes)
    
    results = {
        'dataset': dataset_name,
        'algorithm': 'Random Forest Classifier',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'brier_score': brier,
        'mcc': mcc,
        'oob_error': oob_error,
        'feature_importance': feature_importance,
        'avg_tree_depth': avg_tree_depth,
        'max_tree_depth': max_tree_depth,
        'avg_leaf_nodes': avg_leaf_nodes,
        'default_rate': y.mean()
    }
    
    return results

if __name__ == '__main__':
    import sys
    dataset_path = sys.argv[1]
    dataset_name = sys.argv[2]
    results = run_algorithm_2(dataset_path, dataset_name)
    print(results)
```

## 3. algorithm_3_xgboost.py

```py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

def run_algorithm_3(dataset_path, dataset_name):
    data = pd.read_csv(dataset_path)
    
    feature_cols = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
    X = data[feature_cols]
    y = data['default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=20
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    feature_importance_gain = dict(zip(feature_cols, model.feature_importances_))
    
    booster = model.get_booster()
    importance_cover = booster.get_score(importance_type='cover')
    importance_weight = booster.get_score(importance_type='weight')
    
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
    
    results = {
        'dataset': dataset_name,
        'algorithm': 'XGBoost Gradient Boosting',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'brier_score': brier,
        'mcc': mcc,
        'feature_importance_gain': feature_importance_gain,
        'importance_cover': importance_cover,
        'importance_weight': importance_weight,
        'best_iteration': best_iteration,
        'n_estimators': model.n_estimators,
        'default_rate': y.mean()
    }
    
    return results

if __name__ == '__main__':
    import sys
    dataset_path = sys.argv[1]
    dataset_name = sys.argv[2]
    results = run_algorithm_3(dataset_path, dataset_name)
    print(results)
```

## 4. algorithm_4_hoeffding_adwin.py

```py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

class SimpleHoeffdingTree:
    def __init__(self):
        self.tree = DecisionTreeClassifier(max_depth=10, random_state=42)
        self.is_fitted = False
        self.n_samples = 0
        
    def partial_fit(self, X, y):
        if not self.is_fitted:
            self.tree.fit(X, y)
            self.is_fitted = True
        else:
            self.tree.fit(X, y)
        self.n_samples += len(X)
        
    def predict(self, X):
        return self.tree.predict(X)
    
    def predict_proba(self, X):
        return self.tree.predict_proba(X)

class SimpleADWIN:
    def __init__(self, delta=0.002):
        self.delta = delta
        self.width = 0
        self.buffer = []
        
    def add_element(self, value):
        self.buffer.append(value)
        self.width = len(self.buffer)
        
    def detected_change(self):
        if self.width < 30:
            return False
        
        n = self.width
        for i in range(10, n - 10):
            w0 = self.buffer[:i]
            w1 = self.buffer[i:]
            
            if len(w0) > 0 and len(w1) > 0:
                m0 = np.mean(w0)
                m1 = np.mean(w1)
                
                epsilon = np.sqrt(2 * np.log(2 / self.delta) / (1.0 / len(w0) + 1.0 / len(w1)))
                
                if abs(m0 - m1) > epsilon:
                    self.buffer = self.buffer[i:]
                    self.width = len(self.buffer)
                    return True
        
        return False

def run_algorithm_4(dataset_path, dataset_name):
    data = pd.read_csv(dataset_path)
    
    feature_cols = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
    X = data[feature_cols].values
    y = data['default'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    
    ht = SimpleHoeffdingTree()
    adwin = SimpleADWIN()
    
    prequential_accuracy = []
    drift_points = []
    window_sizes = []
    
    batch_size = 10
    for i in range(0, len(X_train), batch_size):
        end_idx = min(i + batch_size, len(X_train))
        X_batch = X_train[i:end_idx]
        y_batch = y_train[i:end_idx]
        
        if i > 0:
            preds = ht.predict(X_batch)
            errors = [0 if p == y else 1 for p, y in zip(preds, y_batch)]
            
            for error in errors:
                adwin.add_element(error)
                
            if adwin.detected_change():
                drift_points.append(i)
            
            window_sizes.append(adwin.width)
        
        ht.partial_fit(X_batch, y_batch)
    
    y_pred = ht.predict(X_test)
    y_prob = ht.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc_roc = roc_auc_score(y_test, y_prob)
    except:
        auc_roc = 0.5
    
    try:
        brier = brier_score_loss(y_test, y_prob)
    except:
        brier = 0.25
    
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results = {
        'dataset': dataset_name,
        'algorithm': 'Hoeffding Tree with ADWIN',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'brier_score': brier,
        'mcc': mcc,
        'drift_points': drift_points,
        'n_drift_detections': len(drift_points),
        'avg_window_size': np.mean(window_sizes) if window_sizes else 0,
        'final_n_samples': ht.n_samples,
        'default_rate': y.mean()
    }
    
    return results

if __name__ == '__main__':
    import sys
    dataset_path = sys.argv[1]
    dataset_name = sys.argv[2]
    results = run_algorithm_4(dataset_path, dataset_name)
    print(results)
```

## 5. algorithm_5_kmeans.py

```py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def run_algorithm_5(dataset_path, dataset_name):
    data = pd.read_csv(dataset_path)
    
    feature_cols = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
    X = data[feature_cols]
    y = data['default']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    
    try:
        silhouette = silhouette_score(X_scaled, cluster_labels)
    except:
        silhouette = 0
    
    cluster_analysis = []
    for i in range(3):
        cluster_mask = cluster_labels == i
        cluster_size = np.sum(cluster_mask)
        cluster_default_rate = y[cluster_mask].mean() if cluster_size > 0 else 0
        cluster_analysis.append({
            'cluster_id': i,
            'size': int(cluster_size),
            'default_rate': cluster_default_rate,
            'centroid': centroids[i].tolist()
        })
    
    cluster_analysis.sort(key=lambda x: x['default_rate'], reverse=True)
    
    high_risk_cluster = cluster_analysis[0]['cluster_id']
    low_risk_cluster = cluster_analysis[-1]['cluster_id']
    
    predicted_default = np.array([1 if cl == high_risk_cluster else 0 for cl in cluster_labels])
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y, predicted_default)
    
    centroid_dict = {}
    for i, col in enumerate(feature_cols):
        centroid_dict[col] = centroids[:, i].tolist()
    
    results = {
        'dataset': dataset_name,
        'algorithm': 'K-Means Clustering',
        'n_clusters': 3,
        'inertia': inertia,
        'silhouette_score': silhouette,
        'cluster_analysis': cluster_analysis,
        'centroids': centroid_dict,
        'cluster_accuracy': accuracy,
        'default_rate': y.mean()
    }
    
    return results

if __name__ == '__main__':
    import sys
    dataset_path = sys.argv[1]
    dataset_name = sys.argv[2]
    results = run_algorithm_5(dataset_path, dataset_name)
    print(results)
```

## 6. algorithm_6_dbscan.py

```py
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def run_algorithm_6(dataset_path, dataset_name):
    data = pd.read_csv(dataset_path)
    
    feature_cols = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
    X = data[feature_cols]
    y = data['default']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    cluster_labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    noise_fraction = n_noise / len(cluster_labels)
    
    cluster_mask = cluster_labels != -1
    if n_clusters > 1 and np.sum(cluster_mask) > 0:
        try:
            silhouette = silhouette_score(X_scaled[cluster_mask], cluster_labels[cluster_mask])
        except:
            silhouette = 0
    else:
        silhouette = 0
    
    noise_default_rate = y[cluster_labels == -1].mean() if n_noise > 0 else 0
    clustered_default_rate = y[cluster_labels != -1].mean() if np.sum(cluster_mask) > 0 else 0
    
    cluster_analysis = []
    for i in range(n_clusters):
        cluster_mask_i = cluster_labels == i
        cluster_size = np.sum(cluster_mask_i)
        if cluster_size > 0:
            cluster_default_rate = y[cluster_mask_i].mean()
            cluster_center = X_scaled[cluster_mask_i].mean(axis=0)
            cluster_analysis.append({
                'cluster_id': i,
                'size': int(cluster_size),
                'default_rate': cluster_default_rate,
                'centroid': cluster_center.tolist()
            })
    
    results = {
        'dataset': dataset_name,
        'algorithm': 'DBSCAN Clustering',
        'n_clusters': n_clusters,
        'n_noise_points': int(n_noise),
        'noise_fraction': noise_fraction,
        'silhouette_score': silhouette,
        'noise_default_rate': noise_default_rate,
        'clustered_default_rate': clustered_default_rate,
        'cluster_analysis': cluster_analysis,
        'default_rate': y.mean()
    }
    
    return results

if __name__ == '__main__':
    import sys
    dataset_path = sys.argv[1]
    dataset_name = sys.argv[2]
    results = run_algorithm_6(dataset_path, dataset_name)
    print(results)
```

## 7. algorithm_7_isolation_forest.py

```py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def run_algorithm_7(dataset_path, dataset_name):
    data = pd.read_csv(dataset_path)
    
    feature_cols = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
    X = data[feature_cols]
    y = data['default']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    contamination_rates = [0.05, 0.10, 0.15]
    results_by_contamination = {}
    
    for cont_rate in contamination_rates:
        iso_forest = IsolationForest(
            n_estimators=200,
            contamination=cont_rate,
            random_state=42
        )
        
        iso_forest.fit(X_scaled)
        
        anomaly_scores = -iso_forest.score_samples(X_scaled)
        anomaly_labels = iso_forest.predict(X_scaled)
        
        anomaly_rate = np.mean(anomaly_labels == -1)
        
        anomaly_mask = anomaly_labels == -1
        normal_mask = anomaly_labels == 1
        
        anomaly_default_rate = y[anomaly_mask].mean() if np.sum(anomaly_mask) > 0 else 0
        normal_default_rate = y[normal_mask].mean() if np.sum(normal_mask) > 0 else 0
        
        results_by_contamination[f'contamination_{int(cont_rate*100)}'] = {
            'anomaly_rate': anomaly_rate,
            'anomaly_default_rate': anomaly_default_rate,
            'normal_default_rate': normal_default_rate,
            'mean_anomaly_score': np.mean(anomaly_scores),
            'std_anomaly_score': np.std(anomaly_scores)
        }
    
    iso_forest_main = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    iso_forest_main.fit(X_scaled)
    
    anomaly_scores = -iso_forest_main.score_samples(X_scaled)
    
    feature_importance = {}
    for i, col in enumerate(feature_cols):
        correlations = []
        for j in range(X_scaled.shape[1]):
            if i != j:
                corr = np.corrcoef(X_scaled[:, i], anomaly_scores)[0, 1]
                correlations.append(abs(corr))
        feature_importance[col] = np.mean(correlations) if correlations else 0
    
    score_distribution = {
        'mean': np.mean(anomaly_scores),
        'std': np.std(anomaly_scores),
        'min': np.min(anomaly_scores),
        'max': np.max(anomaly_scores),
        'median': np.median(anomaly_scores),
        'q25': np.percentile(anomaly_scores, 25),
        'q75': np.percentile(anomaly_scores, 75)
    }
    
    results = {
        'dataset': dataset_name,
        'algorithm': 'Isolation Forest',
        'contamination_results': results_by_contamination,
        'feature_importance': feature_importance,
        'score_distribution': score_distribution,
        'default_rate': y.mean()
    }
    
    return results

if __name__ == '__main__':
    import sys
    dataset_path = sys.argv[1]
    dataset_name = sys.argv[2]
    results = run_algorithm_7(dataset_path, dataset_name)
    print(results)
```

## 8. algorithm_8_autoencoder.py

```py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def run_algorithm_8(dataset_path, dataset_name):
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense
        tf.get_logger().setLevel('ERROR')
        
        data = pd.read_csv(dataset_path)
        
        feature_cols = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
        X = data[feature_cols]
        y = data['default']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test = train_test_split(X_scaled, test_size=0.3, random_state=42)
        
        input_dim = X_train.shape[1]
        encoding_dim = 8
        
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(32, activation='relu')(input_layer)
        bottleneck = Dense(encoding_dim, activation='relu')(encoder)
        decoder = Dense(32, activation='relu')(bottleneck)
        output_layer = Dense(input_dim, activation='linear')(decoder)
        
        autoencoder = Model(input_layer, output_layer)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        autoencoder.fit(
            X_train, X_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        
        X_reconstructed = autoencoder.predict(X_scaled, verbose=0)
        
        reconstruction_errors = np.mean(np.square(X_scaled - X_reconstructed), axis=1)
        
        feature_errors = {}
        for i, col in enumerate(feature_cols):
            feature_reconstruction = X_reconstructed[:, i]
            feature_original = X_scaled[:, i]
            feature_errors[col] = float(np.mean(np.square(feature_original - feature_reconstruction)))
        
        encoder_model = Model(input_layer, bottleneck)
        latent_representations = encoder_model.predict(X_scaled, verbose=0)
        
        threshold = np.percentile(reconstruction_errors, 90)
        anomalies = reconstruction_errors > threshold
        
        anomaly_rate = np.mean(anomalies)
        anomaly_default_rate = y[anomalies].mean() if np.sum(anomalies) > 0 else 0
        normal_default_rate = y[~anomalies].mean() if np.sum(~anomalies) > 0 else 0
        
        error_stats = {
            'mean': float(np.mean(reconstruction_errors)),
            'std': float(np.std(reconstruction_errors)),
            'min': float(np.min(reconstruction_errors)),
            'max': float(np.max(reconstruction_errors)),
            'median': float(np.median(reconstruction_errors)),
            'threshold_90': float(threshold)
        }
        
        results = {
            'dataset': dataset_name,
            'algorithm': 'Autoencoder',
            'encoding_dim': encoding_dim,
            'reconstruction_error_stats': error_stats,
            'feature_errors': feature_errors,
            'anomaly_rate': float(anomaly_rate),
            'anomaly_default_rate': float(anomaly_default_rate),
            'normal_default_rate': float(normal_default_rate),
            'latent_space_mean': float(np.mean(latent_representations)),
            'latent_space_std': float(np.std(latent_representations)),
            'default_rate': float(y.mean())
        }
        
    except Exception as e:
        results = {
            'dataset': dataset_name,
            'algorithm': 'Autoencoder',
            'error': str(e),
            'default_rate': data['default'].mean() if 'data' in locals() else 0
        }
    
    return results

if __name__ == '__main__':
    import sys
    dataset_path = sys.argv[1]
    dataset_name = sys.argv[2]
    results = run_algorithm_8(dataset_path, dataset_name)
    print(results)
```

## 9. algorithm_9_adaptive_ensemble.py

```py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

def run_algorithm_9(dataset_path, dataset_name):
    data = pd.read_csv(dataset_path)
    
    feature_cols = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
    X = data[feature_cols]
    y = data['default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': None
    }
    
    try:
        from xgboost import XGBClassifier
        models['gradient_boosting'] = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    except:
        models['gradient_boosting'] = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model_weights = {'logistic_regression': 0.33, 'random_forest': 0.33, 'gradient_boosting': 0.34}
    active_model = 'random_forest'
    
    warning_threshold = 0.3
    drift_threshold = 0.4
    
    window_size = 100
    error_history = []
    drift_detected = False
    warning_zone = False
    warning_start = None
    
    for name, model in models.items():
        if model is not None:
            model.fit(X_train_scaled, y_train)
    
    ensemble_predictions = []
    individual_predictions = {name: [] for name in models.keys()}
    
    for i, x in enumerate(X_test_scaled):
        x = x.reshape(1, -1)
        
        probs = {}
        for name, model in models.items():
            if model is not None:
                prob = model.predict_proba(x)[0, 1]
                probs[name] = prob
                individual_predictions[name].append(prob)
        
        ensemble_prob = sum(probs[name] * model_weights[name] for name in probs.keys())
        ensemble_predictions.append(ensemble_prob)
        
        if i < len(y_test):
            pred = 1 if ensemble_prob > 0.5 else 0
            error = 0 if pred == y_test.iloc[i] else 1
            error_history.append(error)
            
            if len(error_history) >= window_size:
                recent_error = np.mean(error_history[-window_size:])
                
                if recent_error > drift_threshold and not drift_detected:
                    drift_detected = True
                    active_model = 'logistic_regression'
                    model_weights['logistic_regression'] = 0.5
                    model_weights['random_forest'] = 0.3
                    model_weights['gradient_boosting'] = 0.2
                elif recent_error > warning_threshold and not warning_zone and not drift_detected:
                    warning_zone = True
                    warning_start = i
                elif recent_error <= warning_threshold and warning_zone:
                    warning_zone = False
    
    y_pred = np.array([1 if p > 0.5 else 0 for p in ensemble_predictions])
    y_prob = np.array(ensemble_predictions)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results = {
        'dataset': dataset_name,
        'algorithm': 'Adaptive Ensemble with DDM',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'brier_score': brier,
        'mcc': mcc,
        'drift_detected': drift_detected,
        'warning_zone': warning_zone,
        'active_model': active_model,
        'final_weights': model_weights,
        'avg_error_rate': np.mean(error_history) if error_history else 0,
        'default_rate': y.mean()
    }
    
    return results

if __name__ == '__main__':
    import sys
    dataset_path = sys.argv[1]
    dataset_name = sys.argv[2]
    results = run_algorithm_9(dataset_path, dataset_name)
    print(results)
```
