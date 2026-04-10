## Merged Files List
- 1. algorithm_1_logistic_regression.md (556 B)
- 2. algorithm_2_random_forest_classifier.md (585 B)
- 3. algorithm_3_xgboost_gradient_boosting.md (804 B)
- 4. algorithm_4_hoeffding_tree_with_adwin.md (372 B)
- 5. algorithm_5_k-means_clustering.md (1.4 KB)
- 6. algorithm_6_dbscan_clustering.md (725 B)
- 7. algorithm_7_isolation_forest.md (1 KB)
- 8. algorithm_8_autoencoder.md (638 B)
- 9. algorithm_9_adaptive_ensemble_with_ddm.md (495 B)


## 1. algorithm_1_logistic_regression.md

```md
# Algorithm 1: Logistic Regression

**Dataset:** V3_Phase3_Stress

## Results

### accuracy
0.783333

### precision
0.800000

### recall
0.823529

### f1_score
0.811594

### auc_roc
0.867647

### brier_score
0.148298

### mcc
0.557142

### coefficients

  - credit_lines_outstanding: -0.066701
  - loan_amt_outstanding: -0.035481
  - total_debt_outstanding: 2.247700
  - income: 0.323020
  - years_employed: 0.135812
  - fico_score: -1.972317

### intercept
0.617618

### probability_mean
0.575222

### probability_std
0.340795

### default_rate
0.575000

```

## 2. algorithm_2_random_forest_classifier.md

```md
# Algorithm 2: Random Forest Classifier

**Dataset:** V3_Phase3_Stress

## Results

### accuracy
0.983333

### precision
1.000000

### recall
0.970588

### f1_score
0.985075

### auc_roc
1.000000

### brier_score
0.022631

### mcc
0.966768

### oob_error
0.028571

### feature_importance

  - credit_lines_outstanding: 0.029680
  - loan_amt_outstanding: 0.061280
  - total_debt_outstanding: 0.445298
  - income: 0.063332
  - years_employed: 0.044917
  - fico_score: 0.355492

### avg_tree_depth
5.515000

### max_tree_depth
10

### avg_leaf_nodes
11.215000

### default_rate
0.575000

```

## 3. algorithm_3_xgboost_gradient_boosting.md

```md
# Algorithm 3: XGBoost Gradient Boosting

**Dataset:** V3_Phase3_Stress

## Results

### accuracy
1.000000

### precision
1.000000

### recall
1.000000

### f1_score
1.000000

### auc_roc
1.000000

### brier_score
0.001347

### mcc
1.000000

### feature_importance_gain

  - credit_lines_outstanding: 0.053007156
  - loan_amt_outstanding: 0.067594856
  - total_debt_outstanding: 0.4322932
  - income: 0.035500456
  - years_employed: 0.055284712
  - fico_score: 0.35631964

### importance_cover

  - f0: 5.754553
  - f1: 6.659193
  - f2: 7.014223
  - f3: 4.154740
  - f4: 5.538127
  - f5: 5.536268

### importance_weight

  - f0: 15.000000
  - f1: 26.000000
  - f2: 102.000000
  - f3: 43.000000
  - f4: 22.000000
  - f5: 90.000000

### best_iteration
199

### n_estimators
200

### default_rate
0.575000

```

## 4. algorithm_4_hoeffding_tree_with_adwin.md

```md
# Algorithm 4: Hoeffding Tree with ADWIN

**Dataset:** V3_Phase3_Stress

## Results

### accuracy
0.650000

### precision
0.622642

### recall
0.970588

### f1_score
0.758621

### auc_roc
0.600679

### brier_score
0.350000

### mcc
0.310819

### drift_points
[]

### n_drift_detections
0

### avg_window_size
70.000000

### final_n_samples
140

### default_rate
0.575000

```

## 5. algorithm_5_k-means_clustering.md

```md
# Algorithm 5: K-Means Clustering

**Dataset:** V3_Phase3_Stress

## Results

### n_clusters
3

### inertia
920.417568

### silhouette_score
0.131286

### cluster_analysis

  - {'cluster_id': 1, 'size': 75, 'default_rate': np.float64(0.7333333333333333), 'centroid': [-0.11797139301186146, -0.2641454341541089, 0.573438362384349, -0.5464652597199986, 0.7160983925192586, 0.06356451722252077]}
  - {'cluster_id': 2, 'size': 72, 'default_rate': np.float64(0.5833333333333334), 'centroid': [-0.5076152591782775, 0.05684698396596958, -0.007994534937323454, 0.2204529749129202, -0.9695621014790727, -0.17156365886535657]}
  - {'cluster_id': 0, 'size': 53, 'default_rate': np.float64(0.33962264150943394), 'centroid': [0.8565311912589734, 0.29656461728317696, -0.8006088804403557, 0.4738166091560315, 0.3037941861801665, 0.1431178235210676]}

### centroids

  - credit_lines_outstanding: 
  - 0.8565311912589734
  - -0.11797139301186146
  - -0.5076152591782775
  - loan_amt_outstanding: 
  - 0.29656461728317696
  - -0.2641454341541089
  - 0.05684698396596958
  - total_debt_outstanding: 
  - -0.8006088804403557
  - 0.573438362384349
  - -0.007994534937323454
  - income: 
  - 0.4738166091560315
  - -0.5464652597199986
  - 0.2204529749129202
  - years_employed: 
  - 0.3037941861801665
  - 0.7160983925192586
  - -0.9695621014790727
  - fico_score: 
  - 0.1431178235210676
  - 0.06356451722252077
  - -0.17156365886535657

### cluster_accuracy
0.600000

### default_rate
0.575000

```

## 6. algorithm_6_dbscan_clustering.md

```md
# Algorithm 6: DBSCAN Clustering

**Dataset:** V3_Phase3_Stress

## Results

### n_clusters
2

### n_noise_points
71

### noise_fraction
0.355000

### silhouette_score
0.219274

### noise_default_rate
0.605634

### clustered_default_rate
0.558140

### cluster_analysis

  - {'cluster_id': 0, 'size': 124, 'default_rate': np.float64(0.5806451612903226), 'centroid': [-0.02050075195939747, -0.04895396219067511, 0.019680086840234957, -0.14664090387463585, 0.144679947705438, -0.05754939909505588]}
  - {'cluster_id': 1, 'size': 5, 'default_rate': np.float64(0.0), 'centroid': [0.9627987881290612, 1.3884064903767521, -1.2779628255634097, 1.5462493358658218, -0.3266345256341251, 1.428223012684906]}

### default_rate
0.575000

```

## 7. algorithm_7_isolation_forest.md

```md
# Algorithm 7: Isolation Forest

**Dataset:** V3_Phase3_Stress

## Results

### contamination_results

  - contamination_5: 
  - anomaly_rate: 0.050000
  - anomaly_default_rate: 0.800000
  - normal_default_rate: 0.563158
  - mean_anomaly_score: 0.509306
  - std_anomaly_score: 0.029841
  - contamination_10: 
  - anomaly_rate: 0.100000
  - anomaly_default_rate: 0.700000
  - normal_default_rate: 0.561111
  - mean_anomaly_score: 0.509306
  - std_anomaly_score: 0.029841
  - contamination_15: 
  - anomaly_rate: 0.150000
  - anomaly_default_rate: 0.700000
  - normal_default_rate: 0.552941
  - mean_anomaly_score: 0.509306
  - std_anomaly_score: 0.029841

### feature_importance

  - credit_lines_outstanding: 0.020597
  - loan_amt_outstanding: 0.007973
  - total_debt_outstanding: 0.052304
  - income: 0.150311
  - years_employed: 0.048729
  - fico_score: 0.002266

### score_distribution

  - mean: 0.509306
  - std: 0.029841
  - min: 0.446096
  - max: 0.597423
  - median: 0.507807
  - q25: 0.489906
  - q75: 0.529885

### default_rate
0.575000

```

## 8. algorithm_8_autoencoder.md

```md
# Algorithm 8: Autoencoder

**Dataset:** V3_Phase3_Stress

## Results

### encoding_dim
8

### reconstruction_error_stats

  - mean: 0.214500
  - std: 0.206576
  - min: 0.009037
  - max: 1.551443
  - median: 0.149862
  - threshold_90: 0.471666

### feature_errors

  - credit_lines_outstanding: 0.157161
  - loan_amt_outstanding: 0.256704
  - total_debt_outstanding: 0.150706
  - income: 0.273315
  - years_employed: 0.140167
  - fico_score: 0.308944

### anomaly_rate
0.100000

### anomaly_default_rate
0.800000

### normal_default_rate
0.550000

### latent_space_mean
0.889838

### latent_space_std
0.890601

### default_rate
0.575000

```

## 9. algorithm_9_adaptive_ensemble_with_ddm.md

```md
# Algorithm 9: Adaptive Ensemble with DDM

**Dataset:** V3_Phase3_Stress

## Results

### accuracy
1.000000

### precision
1.000000

### recall
1.000000

### f1_score
1.000000

### auc_roc
1.000000

### brier_score
0.031954

### mcc
1.000000

### drift_detected
False

### warning_zone
False

### active_model
random_forest

### final_weights

  - logistic_regression: 0.330000
  - random_forest: 0.330000
  - gradient_boosting: 0.340000

### avg_error_rate
0.000000

### default_rate
0.575000

```
