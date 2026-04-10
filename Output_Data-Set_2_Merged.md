## Merged Files List
- 1. algorithm_1_logistic_regression.md (556 B)
- 2. algorithm_2_random_forest_classifier.md (585 B)
- 3. algorithm_3_xgboost_gradient_boosting.md (808 B)
- 4. algorithm_4_hoeffding_tree_with_adwin.md (372 B)
- 5. algorithm_5_k-means_clustering.md (1.5 KB)
- 6. algorithm_6_dbscan_clustering.md (510 B)
- 7. algorithm_7_isolation_forest.md (1 KB)
- 8. algorithm_8_autoencoder.md (637 B)
- 9. algorithm_9_adaptive_ensemble_with_ddm.md (494 B)


## 1. algorithm_1_logistic_regression.md

```md
# Algorithm 1: Logistic Regression

**Dataset:** V2_Phase2_Shift

## Results

### accuracy
0.726667

### precision
0.659574

### recall
0.553571

### f1_score
0.601942

### auc_roc
0.770327

### brier_score
0.181897

### mcc
0.399757

### coefficients

  - credit_lines_outstanding: -0.142849
  - loan_amt_outstanding: 0.103954
  - total_debt_outstanding: 0.891066
  - income: -0.649993
  - years_employed: -0.156829
  - fico_score: 0.010333

### intercept
-0.664775

### probability_mean
0.380256

### probability_std
0.223010

### default_rate
0.372000

```

## 2. algorithm_2_random_forest_classifier.md

```md
# Algorithm 2: Random Forest Classifier

**Dataset:** V2_Phase2_Shift

## Results

### accuracy
0.733333

### precision
0.690476

### recall
0.517857

### f1_score
0.591837

### auc_roc
0.731763

### brier_score
0.189083

### mcc
0.408885

### oob_error
0.287143

### feature_importance

  - credit_lines_outstanding: 0.074755
  - loan_amt_outstanding: 0.139136
  - total_debt_outstanding: 0.284344
  - income: 0.243060
  - years_employed: 0.109909
  - fico_score: 0.148796

### avg_tree_depth
10.000000

### max_tree_depth
10

### avg_leaf_nodes
99.520000

### default_rate
0.372000

```

## 3. algorithm_3_xgboost_gradient_boosting.md

```md
# Algorithm 3: XGBoost Gradient Boosting

**Dataset:** V2_Phase2_Shift

## Results

### accuracy
0.716667

### precision
0.651685

### recall
0.517857

### f1_score
0.577114

### auc_roc
0.749098

### brier_score
0.187077

### mcc
0.373750

### feature_importance_gain

  - credit_lines_outstanding: 0.12922338
  - loan_amt_outstanding: 0.1478836
  - total_debt_outstanding: 0.26066148
  - income: 0.19782121
  - years_employed: 0.1372157
  - fico_score: 0.12719478

### importance_cover

  - f0: 13.118142
  - f1: 24.044088
  - f2: 29.628504
  - f3: 24.401131
  - f4: 16.488985
  - f5: 23.185871

### importance_weight

  - f0: 108.000000
  - f1: 123.000000
  - f2: 258.000000
  - f3: 321.000000
  - f4: 150.000000
  - f5: 225.000000

### best_iteration
27

### n_estimators
200

### default_rate
0.372000

```

## 4. algorithm_4_hoeffding_tree_with_adwin.md

```md
# Algorithm 4: Hoeffding Tree with ADWIN

**Dataset:** V2_Phase2_Shift

## Results

### accuracy
0.690000

### precision
0.787879

### recall
0.232143

### f1_score
0.358621

### auc_roc
0.597454

### brier_score
0.310000

### mcc
0.301305

### drift_points
[]

### n_drift_detections
0

### avg_window_size
350.000000

### final_n_samples
700

### default_rate
0.372000

```

## 5. algorithm_5_k-means_clustering.md

```md
# Algorithm 5: K-Means Clustering

**Dataset:** V2_Phase2_Shift

## Results

### n_clusters
3

### inertia
4741.198649

### silhouette_score
0.116574

### cluster_analysis

  - {'cluster_id': 1, 'size': 364, 'default_rate': np.float64(0.5302197802197802), 'centroid': [-0.14236747050967086, 0.20775136520004175, 1.0524671484886854, 0.07506308864604856, -0.03803257474697341, 0.03229726284935349]}
  - {'cluster_id': 2, 'size': 308, 'default_rate': np.float64(0.37987012987012986), 'centroid': [-0.33630533071655, -0.09338932302761194, -0.6262635836703535, -0.9308850545178035, -0.05580800578672236, 0.06803572692618341]}
  - {'cluster_id': 0, 'size': 328, 'default_rate': np.float64(0.18902439024390244), 'centroid': [0.4737920766043206, -0.14285849219606941, -0.5799050557299164, 0.7908220503790303, 0.09461196033600301, -0.09972929137326153]}

### centroids

  - credit_lines_outstanding: 
  - 0.4737920766043206
  - -0.14236747050967086
  - -0.33630533071655
  - loan_amt_outstanding: 
  - -0.14285849219606941
  - 0.20775136520004175
  - -0.09338932302761194
  - total_debt_outstanding: 
  - -0.5799050557299164
  - 1.0524671484886854
  - -0.6262635836703535
  - income: 
  - 0.7908220503790303
  - 0.07506308864604856
  - -0.9308850545178035
  - years_employed: 
  - 0.09461196033600301
  - -0.03803257474697341
  - -0.05580800578672236
  - fico_score: 
  - -0.09972929137326153
  - 0.03229726284935349
  - 0.06803572692618341

### cluster_accuracy
0.650000

### default_rate
0.372000

```

## 6. algorithm_6_dbscan_clustering.md

```md
# Algorithm 6: DBSCAN Clustering

**Dataset:** V2_Phase2_Shift

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
0.372000

### cluster_analysis

  - {'cluster_id': 0, 'size': 1000, 'default_rate': np.float64(0.372), 'centroid': [6.650235917504687e-17, -7.038813976123493e-17, 1.2251311076738602e-16, 1.6275869541004795e-16, 1.0103029524088925e-16, -6.539768726554485e-16]}

### default_rate
0.372000

```

## 7. algorithm_7_isolation_forest.md

```md
# Algorithm 7: Isolation Forest

**Dataset:** V2_Phase2_Shift

## Results

### contamination_results

  - contamination_5: 
  - anomaly_rate: 0.050000
  - anomaly_default_rate: 0.440000
  - normal_default_rate: 0.368421
  - mean_anomaly_score: 0.517658
  - std_anomaly_score: 0.028319
  - contamination_10: 
  - anomaly_rate: 0.100000
  - anomaly_default_rate: 0.500000
  - normal_default_rate: 0.357778
  - mean_anomaly_score: 0.517658
  - std_anomaly_score: 0.028319
  - contamination_15: 
  - anomaly_rate: 0.150000
  - anomaly_default_rate: 0.406667
  - normal_default_rate: 0.365882
  - mean_anomaly_score: 0.517658
  - std_anomaly_score: 0.028319

### feature_importance

  - credit_lines_outstanding: 0.032662
  - loan_amt_outstanding: 0.006388
  - total_debt_outstanding: 0.042994
  - income: 0.002765
  - years_employed: 0.131319
  - fico_score: 0.108824

### score_distribution

  - mean: 0.517658
  - std: 0.028319
  - min: 0.450970
  - max: 0.610810
  - median: 0.516655
  - q25: 0.496802
  - q75: 0.537145

### default_rate
0.372000

```

## 8. algorithm_8_autoencoder.md

```md
# Algorithm 8: Autoencoder

**Dataset:** V2_Phase2_Shift

## Results

### encoding_dim
8

### reconstruction_error_stats

  - mean: 0.005013
  - std: 0.003378
  - min: 0.000174
  - max: 0.030212
  - median: 0.004289
  - threshold_90: 0.009315

### feature_errors

  - credit_lines_outstanding: 0.004765
  - loan_amt_outstanding: 0.004455
  - total_debt_outstanding: 0.003970
  - income: 0.006052
  - years_employed: 0.006199
  - fico_score: 0.004635

### anomaly_rate
0.100000

### anomaly_default_rate
0.310000

### normal_default_rate
0.378889

### latent_space_mean
1.483342

### latent_space_std
0.754548

### default_rate
0.372000

```

## 9. algorithm_9_adaptive_ensemble_with_ddm.md

```md
# Algorithm 9: Adaptive Ensemble with DDM

**Dataset:** V2_Phase2_Shift

## Results

### accuracy
0.730000

### precision
0.663158

### recall
0.562500

### f1_score
0.608696

### auc_roc
0.744681

### brier_score
0.188484

### mcc
0.407900

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
0.270000

### default_rate
0.372000

```
