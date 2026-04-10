## Merged Files List
- 1. algorithm_1_logistic_regression.md (559 B)
- 2. algorithm_2_random_forest_classifier.md (587 B)
- 3. algorithm_3_xgboost_gradient_boosting.md (811 B)
- 4. algorithm_4_hoeffding_tree_with_adwin.md (375 B)
- 5. algorithm_5_k-means_clustering.md (1.4 KB)
- 6. algorithm_6_dbscan_clustering.md (515 B)
- 7. algorithm_7_isolation_forest.md (1 KB)
- 8. algorithm_8_autoencoder.md (640 B)
- 9. algorithm_9_adaptive_ensemble_with_ddm.md (497 B)


## 1. algorithm_1_logistic_regression.md

```md
# Algorithm 1: Logistic Regression

**Dataset:** V1_Phase1_Baseline

## Results

### accuracy
0.886667

### precision
0.878049

### recall
0.850394

### f1_score
0.864000

### auc_roc
0.955623

### brier_score
0.084787

### mcc
0.767188

### coefficients

  - credit_lines_outstanding: -0.069017
  - loan_amt_outstanding: 0.329795
  - total_debt_outstanding: 2.478791
  - income: -2.768167
  - years_employed: 0.045000
  - fico_score: -1.788739

### intercept
-1.135751

### probability_mean
0.414048

### probability_std
0.405403

### default_rate
0.425000

```

## 2. algorithm_2_random_forest_classifier.md

```md
# Algorithm 2: Random Forest Classifier

**Dataset:** V1_Phase1_Baseline

## Results

### accuracy
0.856667

### precision
0.868421

### recall
0.779528

### f1_score
0.821577

### auc_roc
0.946202

### brier_score
0.096970

### mcc
0.705242

### oob_error
0.108571

### feature_importance

  - credit_lines_outstanding: 0.030561
  - loan_amt_outstanding: 0.128270
  - total_debt_outstanding: 0.248022
  - income: 0.335643
  - years_employed: 0.044992
  - fico_score: 0.212512

### avg_tree_depth
9.985000

### max_tree_depth
10

### avg_leaf_nodes
65.220000

### default_rate
0.425000

```

## 3. algorithm_3_xgboost_gradient_boosting.md

```md
# Algorithm 3: XGBoost Gradient Boosting

**Dataset:** V1_Phase1_Baseline

## Results

### accuracy
0.853333

### precision
0.854701

### recall
0.787402

### f1_score
0.819672

### auc_roc
0.950799

### brier_score
0.098189

### mcc
0.698090

### feature_importance_gain

  - credit_lines_outstanding: 0.04926539
  - loan_amt_outstanding: 0.108336076
  - total_debt_outstanding: 0.2722051
  - income: 0.2870401
  - years_employed: 0.071334794
  - fico_score: 0.21181846

### importance_cover

  - f0: 8.457204
  - f1: 14.614194
  - f2: 19.524685
  - f3: 20.920101
  - f4: 10.927139
  - f5: 20.428173

### importance_weight

  - f0: 89.000000
  - f1: 152.000000
  - f2: 292.000000
  - f3: 317.000000
  - f4: 126.000000
  - f5: 237.000000

### best_iteration
50

### n_estimators
200

### default_rate
0.425000

```

## 4. algorithm_4_hoeffding_tree_with_adwin.md

```md
# Algorithm 4: Hoeffding Tree with ADWIN

**Dataset:** V1_Phase1_Baseline

## Results

### accuracy
0.753333

### precision
0.708661

### recall
0.708661

### f1_score
0.708661

### auc_roc
0.747394

### brier_score
0.246667

### mcc
0.494789

### drift_points
[]

### n_drift_detections
0

### avg_window_size
350.000000

### final_n_samples
700

### default_rate
0.425000

```

## 5. algorithm_5_k-means_clustering.md

```md
# Algorithm 5: K-Means Clustering

**Dataset:** V1_Phase1_Baseline

## Results

### n_clusters
3

### inertia
4248.840884

### silhouette_score
0.156012

### cluster_analysis

  - {'cluster_id': 2, 'size': 357, 'default_rate': np.float64(0.8319327731092437), 'centroid': [0.17119349763964348, 0.8039636404776511, 0.8287197033793047, -0.4430579624056465, 0.1536930270167886, -0.4290219849345829]}
  - {'cluster_id': 0, 'size': 377, 'default_rate': np.float64(0.27320954907161804), 'centroid': [0.12983154666017901, -0.9239011022009899, -0.9024390426111742, -0.2101092885990103, 0.03676689736551578, -0.1467681689252045]}
  - {'cluster_id': 1, 'size': 266, 'default_rate': np.float64(0.09398496240601503), 'centroid': [-0.4137690667227076, 0.23043494691448146, 0.16679167277444001, 0.8924168961678302, -0.2583816953074921, 0.7838061966407794]}

### centroids

  - credit_lines_outstanding: 
  - 0.12983154666017901
  - -0.4137690667227076
  - 0.17119349763964348
  - loan_amt_outstanding: 
  - -0.9239011022009899
  - 0.23043494691448146
  - 0.8039636404776511
  - total_debt_outstanding: 
  - -0.9024390426111742
  - 0.16679167277444001
  - 0.8287197033793047
  - income: 
  - -0.2101092885990103
  - 0.8924168961678302
  - -0.4430579624056465
  - years_employed: 
  - 0.03676689736551578
  - -0.2583816953074921
  - 0.1536930270167886
  - fico_score: 
  - -0.1467681689252045
  - 0.7838061966407794
  - -0.4290219849345829

### cluster_accuracy
0.812000

### default_rate
0.425000

```

## 6. algorithm_6_dbscan_clustering.md

```md
# Algorithm 6: DBSCAN Clustering

**Dataset:** V1_Phase1_Baseline

## Results

### n_clusters
1

### n_noise_points
0

### noise_fraction
0.000000

### silhouette_score
0

### noise_default_rate
0

### clustered_default_rate
0.425000

### cluster_analysis

  - {'cluster_id': 0, 'size': 1000, 'default_rate': np.float64(0.425), 'centroid': [-1.0436096431476472e-17, 1.481037514849959e-16, -1.5121237595394633e-16, 2.369215934550084e-16, -3.4305891460917337e-17, -5.505595979116151e-16]}

### default_rate
0.425000

```

## 7. algorithm_7_isolation_forest.md

```md
# Algorithm 7: Isolation Forest

**Dataset:** V1_Phase1_Baseline

## Results

### contamination_results

  - contamination_5: 
  - anomaly_rate: 0.050000
  - anomaly_default_rate: 0.320000
  - normal_default_rate: 0.430526
  - mean_anomaly_score: 0.505042
  - std_anomaly_score: 0.033710
  - contamination_10: 
  - anomaly_rate: 0.100000
  - anomaly_default_rate: 0.380000
  - normal_default_rate: 0.430000
  - mean_anomaly_score: 0.505042
  - std_anomaly_score: 0.033710
  - contamination_15: 
  - anomaly_rate: 0.150000
  - anomaly_default_rate: 0.326667
  - normal_default_rate: 0.442353
  - mean_anomaly_score: 0.505042
  - std_anomaly_score: 0.033710

### feature_importance

  - credit_lines_outstanding: 0.008940
  - loan_amt_outstanding: 0.142378
  - total_debt_outstanding: 0.164946
  - income: 0.154344
  - years_employed: 0.093045
  - fico_score: 0.020095

### score_distribution

  - mean: 0.505042
  - std: 0.033710
  - min: 0.431051
  - max: 0.624575
  - median: 0.503237
  - q25: 0.480229
  - q75: 0.526434

### default_rate
0.425000

```

## 8. algorithm_8_autoencoder.md

```md
# Algorithm 8: Autoencoder

**Dataset:** V1_Phase1_Baseline

## Results

### encoding_dim
8

### reconstruction_error_stats

  - mean: 0.006434
  - std: 0.004978
  - min: 0.000203
  - max: 0.060213
  - median: 0.005189
  - threshold_90: 0.012134

### feature_errors

  - credit_lines_outstanding: 0.007492
  - loan_amt_outstanding: 0.006223
  - total_debt_outstanding: 0.005409
  - income: 0.006225
  - years_employed: 0.006536
  - fico_score: 0.006720

### anomaly_rate
0.100000

### anomaly_default_rate
0.320000

### normal_default_rate
0.436667

### latent_space_mean
1.399789

### latent_space_std
0.967017

### default_rate
0.425000

```

## 9. algorithm_9_adaptive_ensemble_with_ddm.md

```md
# Algorithm 9: Adaptive Ensemble with DDM

**Dataset:** V1_Phase1_Baseline

## Results

### accuracy
0.866667

### precision
0.865546

### recall
0.811024

### f1_score
0.837398

### auc_roc
0.956670

### brier_score
0.089023

### mcc
0.725708

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
0.133333

### default_rate
0.425000

```
