# Custom Evaluators

## Overview

This guide explains how to create custom evaluators for the GENESIS Core Lib. Custom evaluators allow you to implement specialized quality assessment metrics and evaluation methods beyond the built-in evaluators.

## Base Evaluator Classes

All custom evaluators should inherit from the base classes in `src/sdg_core_lib/evaluate/`:

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Optional
from sdg_core_lib.dataset.datasets import Dataset
from sdg_core_lib.evaluate.metrics import MetricReport

class BaseEvaluator(ABC):
    """Abstract base class for all evaluators"""
    
    def __init__(self, real_data: Dataset, synthetic_data: Dataset):
        self._real_data = real_data
        self._synth_data = synthetic_data
        self.report = MetricReport()

    @abstractmethod
    def compute(self) -> Dict[str, Any]:
        """Compute evaluation metrics"""
        raise NotImplementedError

    def get_report(self) -> MetricReport:
        """Get evaluation report"""
        return self.report

    def add_metric(self, name: str, value: Any, description: str = ""):
        """Add a metric to the report"""
        self.report.add_metric(name, value, description)
```

## Creating Custom Evaluators

### Example 1: Statistical Similarity Evaluator

```python
import numpy as np
from scipy import stats
from typing import Dict, Any, List
from sdg_core_lib.evaluate.base_evaluator import BaseEvaluator
from sdg_core_lib.dataset.datasets import Dataset

class StatisticalSimilarityEvaluator(BaseEvaluator):
    """Evaluator for statistical similarity between real and synthetic data"""
    
    def __init__(self, real_data: Dataset, synthetic_data: Dataset, 
                 significance_level: float = 0.05):
        super().__init__(real_data, synthetic_data)
        self.significance_level = significance_level

    def compute(self) -> Dict[str, Any]:
        """Compute statistical similarity metrics"""
        results = {}
        
        # Get data arrays
        real_array = self._real_data.get_computing_data()
        synthetic_array = self._synth_data.get_computing_data()
        
        # Compute column-wise statistics
        if len(real_array.shape) > 1:
            # Multi-dimensional data
            for col_idx in range(real_array.shape[1]):
                col_results = self._evaluate_column(
                    real_array[:, col_idx], 
                    synthetic_array[:, col_idx],
                    f"column_{col_idx}"
                )
                results.update(col_results)
        else:
            # Single-dimensional data
            results.update(self._evaluate_column(real_array, synthetic_array, "data"))
        
        # Overall similarity score
        results["overall_similarity"] = self._compute_overall_similarity(results)
        
        # Add metrics to report
        for metric_name, metric_value in results.items():
            self.add_metric(metric_name, metric_value)
        
        return results

    def _evaluate_column(self, real_col: np.ndarray, synthetic_col: np.ndarray, 
                         column_prefix: str) -> Dict[str, Any]:
        """Evaluate similarity for a single column"""
        results = {}
        
        # Basic statistics
        results[f"{column_prefix}_mean_diff"] = abs(np.mean(real_col) - np.mean(synthetic_col))
        results[f"{column_prefix}_std_diff"] = abs(np.std(real_col) - np.std(synthetic_col))
        results[f"{column_prefix}_median_diff"] = abs(np.median(real_col) - np.median(synthetic_col))
        
        # Distribution similarity tests
        # Kolmogorov-Smirnov test
        ks_statistic, ks_p_value = stats.ks_2samp(real_col, synthetic_col)
        results[f"{column_prefix}_ks_statistic"] = ks_statistic
        results[f"{column_prefix}_ks_p_value"] = ks_p_value
        
        # Anderson-Darling test (if enough samples)
        if len(real_col) > 8 and len(synthetic_col) > 8:
            try:
                ad_statistic, ad_critical_values, ad_significance_levels = stats.anderson_ks_2samp(real_col, synthetic_col)
                results[f"{column_prefix}_ad_statistic"] = ad_statistic
                results[f"{column_prefix}_ad_critical_values"] = ad_critical_values.tolist()
                results[f"{column_prefix}_ad_significance_levels"] = ad_significance_levels.tolist()
            except:
                results[f"{column_prefix}_ad_test"] = "failed"
        
        # Correlation with original data
        if len(real_col) == len(synthetic_col):
            correlation = np.corrcoef(real_col, synthetic_col)[0, 1]
            results[f"{column_prefix}_correlation"] = correlation
        
        # Wasserstein distance (Earth Mover's Distance)
        try:
            from scipy.stats import wasserstein_distance
            wasserstein_dist = wasserstein_distance(real_col, synthetic_col)
            results[f"{column_prefix}_wasserstein_distance"] = wasserstein_dist
        except ImportError:
            results[f"{column_prefix}_wasserstein_distance"] = "not_available"
        
        return results

    def _compute_overall_similarity(self, results: Dict[str, Any]) -> float:
        """Compute overall similarity score"""
        similarity_scores = []
        
        for key, value in results.items():
            if "p_value" in key:
                # Higher p-value means more similar distributions
                similarity_scores.append(value)
            elif "correlation" in key:
                # Higher correlation means more similar
                similarity_scores.append(abs(value))
            elif "distance" in key or "statistic" in key:
                # Lower distance/statistic means more similar
                similarity_scores.append(1.0 / (1.0 + value))
            elif "_diff" in key:
                # Lower difference means more similar
                similarity_scores.append(1.0 / (1.0 + value))
        
        if similarity_scores:
            return np.mean(similarity_scores)
        else:
            return 0.0
```

### Example 2: Machine Learning Evaluator

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from typing import Dict, Any, Tuple
from sdg_core_lib.evaluate.base_evaluator import BaseEvaluator

class MachineLearningEvaluator(BaseEvaluator):
    """Evaluator using machine learning performance metrics"""
    
    def __init__(self, real_data: Dataset, synthetic_data: Dataset,
                 test_size: float = 0.3, random_state: int = 42):
        super().__init__(real_data, synthetic_data)
        self.test_size = test_size
        self.random_state = random_state

    def compute(self) -> Dict[str, Any]:
        """Compute ML-based evaluation metrics"""
        results = {}
        
        # Get data arrays
        real_array = self._real_data.get_computing_data()
        synthetic_array = self._synth_data.get_computing_data()
        
        # Combine data for ML evaluation
        combined_data = np.vstack([real_array, synthetic_array])
        labels = np.array([0] * len(real_array) + [1] * len(synthetic_array))
        
        # Train classifier to distinguish real vs synthetic
        classification_results = self._evaluate_classification(combined_data, labels)
        results.update(classification_results)
        
        # Evaluate feature importance preservation
        feature_importance_results = self._evaluate_feature_importance(real_array, synthetic_array)
        results.update(feature_importance_results)
        
        # Evaluate predictive performance on downstream tasks
        downstream_results = self._evaluate_downstream_tasks(real_array, synthetic_array)
        results.update(downstream_results)
        
        # Add metrics to report
        for metric_name, metric_value in results.items():
            self.add_metric(metric_name, metric_value)
        
        return results

    def _evaluate_classification(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Evaluate how well ML can distinguish real vs synthetic data"""
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=self.test_size, random_state=self.random_state
        )
        
        # Train classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(classifier, data, labels, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Feature importance for distinguishing
        feature_importance = classifier.feature_importances_
        
        return {
            "distinguishability_accuracy": accuracy,
            "distinguishability_cv_mean": cv_mean,
            "distinguishability_cv_std": cv_std,
            "distinguishability_feature_importance_mean": np.mean(feature_importance),
            "distinguishability_feature_importance_std": np.std(feature_importance)
        }

    def _evaluate_feature_importance(self, real_data: np.ndarray, 
                                   synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate preservation of feature importance patterns"""
        # Train models on real and synthetic data separately
        real_model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        synth_model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        
        # Use last column as target (or create synthetic target)
        if real_data.shape[1] > 1:
            X_real, y_real = real_data[:, :-1], real_data[:, -1]
            X_synth, y_synth = synthetic_data[:, :-1], synthetic_data[:, -1]
        else:
            # Create synthetic target from data itself
            X_real = real_data.reshape(-1, 1)
            y_real = real_data
            X_synth = synthetic_data.reshape(-1, 1)
            y_synth = synthetic_data
        
        # Train models
        real_model.fit(X_real, y_real)
        synth_model.fit(X_synth, y_synth)
        
        # Compare feature importances
        real_importance = real_model.feature_importances_
        synth_importance = synth_model.feature_importances_
        
        # Calculate correlation between feature importances
        importance_correlation = np.corrcoef(real_importance, synth_importance)[0, 1]
        importance_mse = np.mean((real_importance - synth_importance) ** 2)
        
        return {
            "feature_importance_correlation": importance_correlation,
            "feature_importance_mse": importance_mse,
            "feature_importance_max_diff": np.max(np.abs(real_importance - synth_importance))
        }

    def _evaluate_downstream_tasks(self, real_data: np.ndarray, 
                                 synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate performance on downstream ML tasks"""
        results = {}
        
        if real_data.shape[1] < 2:
            return {"downstream_evaluation": "insufficient_features"}
        
        # Classification task (if we can create binary labels)
        classification_results = self._evaluate_classification_task(real_data, synthetic_data)
        results.update(classification_results)
        
        # Regression task
        regression_results = self._evaluate_regression_task(real_data, synthetic_data)
        results.update(regression_results)
        
        return results

    def _evaluate_classification_task(self, real_data: np.ndarray, 
                                   synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate classification performance"""
        from sklearn.model_selection import train_test_split
        
        # Create binary labels using median of first feature
        combined_data = np.vstack([real_data, synthetic_data])
        first_feature = combined_data[:, 0]
        labels = (first_feature > np.median(first_feature)).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            combined_data, labels, test_size=self.test_size, random_state=self.random_state
        )
        
        # Train classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Train separately on real and synthetic data
        real_labels = (real_data[:, 0] > np.median(real_data[:, 0])).astype(int)
        synth_labels = (synthetic_data[:, 0] > np.median(synthetic_data[:, 0])).astype(int)
        
        # Train on real data
        real_classifier = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        real_classifier.fit(real_data[:, 1:], real_labels)
        
        # Train on synthetic data
        synth_classifier = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        synth_classifier.fit(synthetic_data[:, 1:], synth_labels)
        
        # Test on held-out real data
        from sklearn.model_selection import train_test_split
        X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
            real_data[:, 1:], real_labels, test_size=self.test_size, random_state=self.random_state
        )
        
        real_accuracy = accuracy_score(y_real_test, real_classifier.predict(X_real_test))
        synth_accuracy = accuracy_score(y_real_test, synth_classifier.predict(X_real_test))
        
        return {
            "classification_combined_accuracy": accuracy,
            "classification_real_trained_accuracy": real_accuracy,
            "classification_synth_trained_accuracy": synth_accuracy,
            "classification_performance_gap": abs(real_accuracy - synth_accuracy)
        }

    def _evaluate_regression_task(self, real_data: np.ndarray, 
                                synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate regression performance"""
        from sklearn.model_selection import train_test_split
        
        # Use last feature as target
        X_real, y_real = real_data[:, :-1], real_data[:, -1]
        X_synth, y_synth = synthetic_data[:, :-1], synthetic_data[:, -1]
        
        # Train on real data
        real_regressor = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        real_regressor.fit(X_real, y_real)
        
        # Train on synthetic data
        synth_regressor = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        synth_regressor.fit(X_synth, y_synth)
        
        # Test on held-out real data
        X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
            X_real, y_real, test_size=self.test_size, random_state=self.random_state
        )
        
        # Evaluate
        real_pred = real_regressor.predict(X_real_test)
        synth_pred = synth_regressor.predict(X_real_test)
        
        real_mse = mean_squared_error(y_real_test, real_pred)
        synth_mse = mean_squared_error(y_real_test, synth_pred)
        
        real_r2 = r2_score(y_real_test, real_pred)
        synth_r2 = r2_score(y_real_test, synth_pred)
        
        return {
            "regression_real_mse": real_mse,
            "regression_synth_mse": synth_mse,
            "regression_mse_ratio": synth_mse / real_mse,
            "regression_real_r2": real_r2,
            "regression_synth_r2": synth_r2,
            "regression_r2_gap": abs(real_r2 - synth_r2)
        }
```

### Example 3: Privacy Evaluator

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from typing import Dict, Any, List, Tuple
from sdg_core_lib.evaluate.base_evaluator import BaseEvaluator

class PrivacyEvaluator(BaseEvaluator):
    """Evaluator for privacy preservation in synthetic data"""
    
    def __init__(self, real_data: Dataset, synthetic_data: Dataset,
                 k_neighbors: int = 5, threshold_percentile: float = 95):
        super().__init__(real_data, synthetic_data)
        self.k_neighbors = k_neighbors
        self.threshold_percentile = threshold_percentile

    def compute(self) -> Dict[str, Any]:
        """Compute privacy preservation metrics"""
        results = {}
        
        # Get data arrays
        real_array = self._real_data.get_computing_data()
        synthetic_array = self._synth_data.get_computing_data()
        
        # Distance-based privacy metrics
        distance_results = self._evaluate_distance_privacy(real_array, synthetic_array)
        results.update(distance_results)
        
        # Membership inference attack
        mia_results = self._evaluate_membership_inference(real_array, synthetic_array)
        results.update(mia_results)
        
        # Attribute inference risk
        attribute_results = self._evaluate_attribute_inference(real_array, synthetic_array)
        results.update(attribute_results)
        
        # Record linkage risk
        linkage_results = self._evaluate_record_linkage(real_array, synthetic_array)
        results.update(linkage_results)
        
        # Add metrics to report
        for metric_name, metric_value in results.items():
            self.add_metric(metric_name, metric_value)
        
        return results

    def _evaluate_distance_privacy(self, real_data: np.ndarray, 
                                 synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate distance-based privacy metrics"""
        # Compute nearest neighbor distances
        nn_real = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        nn_real.fit(real_data)
        
        nn_synth = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        nn_synth.fit(synthetic_data)
        
        # Find distances to nearest neighbors in same dataset
        real_distances, _ = nn_real.kneighbors(real_data)
        synth_distances, _ = nn_synth.kneighbors(synthetic_data)
        
        # Exclude self-distance (first neighbor)
        real_nn_distances = real_distances[:, 1:]  # Exclude self
        synth_nn_distances = synth_distances[:, 1:]  # Exclude self
        
        # Find distances to nearest neighbors in other dataset
        real_to_synth_distances, _ = nn_synth.kneighbors(real_data)
        synth_to_real_distances, _ = nn_real.kneighbors(synthetic_data)
        
        # Compute privacy metrics
        avg_real_nn_distance = np.mean(real_nn_distances)
        avg_synth_nn_distance = np.mean(synth_nn_distances)
        avg_real_to_synth_distance = np.mean(real_to_synth_distances)
        avg_synth_to_real_distance = np.mean(synth_to_real_distances)
        
        # Privacy ratio (higher is better)
        privacy_ratio = avg_real_to_synth_distance / avg_real_nn_distance
        
        # Distance overlap
        threshold_real = np.percentile(real_nn_distances.flatten(), self.threshold_percentile)
        threshold_synth = np.percentile(synth_nn_distances.flatten(), self.threshold_percentile)
        
        real_close_to_synth = np.mean(real_to_synth_distances[:, 0] < threshold_real)
        synth_close_to_real = np.mean(synth_to_real_distances[:, 0] < threshold_synth)
        
        return {
            "privacy_avg_real_nn_distance": avg_real_nn_distance,
            "privacy_avg_synth_nn_distance": avg_synth_nn_distance,
            "privacy_avg_real_to_synth_distance": avg_real_to_synth_distance,
            "privacy_avg_synth_to_real_distance": avg_synth_to_real_distance,
            "privacy_ratio": privacy_ratio,
            "privacy_real_close_to_synth_ratio": real_close_to_synth,
            "privacy_synth_close_to_real_ratio": synth_close_to_real
        }

    def _evaluate_membership_inference(self, real_data: np.ndarray, 
                                   synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate membership inference attack vulnerability"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Create training and test sets
        real_train, real_test = train_test_split(real_data, test_size=0.3, random_state=42)
        synth_train, synth_test = train_test_split(synthetic_data, test_size=0.3, random_state=42)
        
        # Train shadow models
        shadow_model1 = RandomForestClassifier(n_estimators=50, random_state=42)
        shadow_model2 = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Create labels for membership inference
        # Members (1): real_train, synth_train
        # Non-members (0): real_test, synth_test
        
        X_members = np.vstack([real_train, synth_train])
        y_members = np.array([1] * len(real_train) + [1] * len(synth_train))
        
        X_non_members = np.vstack([real_test, synth_test])
        y_non_members = np.array([0] * len(real_test) + [0] * len(synth_test))
        
        # Create target variable (using first feature for simplicity)
        if real_data.shape[1] > 1:
            target_members = X_members[:, 0]
            target_non_members = X_non_members[:, 0]
        else:
            target_members = X_members.flatten()
            target_non_members = X_non_members.flatten()
        
        # Train shadow models
        shadow_model1.fit(X_members, target_members)
        shadow_model2.fit(X_non_members, target_non_members)
        
        # Get predictions for attack model
        member_preds = shadow_model1.predict_proba(X_members)[:, 1]
        non_member_preds = shadow_model2.predict_proba(X_non_members)[:, 1]
        
        # Train attack model
        attack_X = np.vstack([member_preds, non_member_preds]).reshape(-1, 1)
        attack_y = np.hstack([y_members, y_non_members])
        
        attack_model = RandomForestClassifier(n_estimators=50, random_state=42)
        attack_model.fit(attack_X, attack_y)
        
        # Evaluate attack
        attack_accuracy = attack_model.score(attack_X, attack_y)
        
        # Random baseline accuracy
        random_baseline = max(len(y_members), len(y_non_members)) / len(attack_y)
        
        # Privacy risk (higher attack accuracy means lower privacy)
        privacy_risk = (attack_accuracy - random_baseline) / (1.0 - random_baseline)
        
        return {
            "membership_inference_attack_accuracy": attack_accuracy,
            "membership_inference_random_baseline": random_baseline,
            "membership_inference_privacy_risk": privacy_risk
        }

    def _evaluate_attribute_inference(self, real_data: np.ndarray, 
                                    synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate attribute inference risk"""
        if real_data.shape[1] < 2:
            return {"attribute_inference": "insufficient_features"}
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Use last feature as target attribute
        X_real = real_data[:, :-1]
        y_real = real_data[:, -1]
        
        X_synth = synthetic_data[:, :-1]
        y_synth = synthetic_data[:, -1]
        
        # Train model on synthetic data
        attr_model = RandomForestClassifier(n_estimators=50, random_state=42)
        attr_model.fit(X_synth, y_synth)
        
        # Test on real data
        real_pred = attr_model.predict(X_real)
        
        # Calculate accuracy (discretize continuous targets)
        if len(np.unique(y_real)) > 10:  # Continuous target
            # Convert to bins
            y_real_binned = pd.cut(y_real, bins=5, labels=False)
            real_pred_binned = pd.cut(real_pred, bins=5, labels=False)
            accuracy = accuracy_score(y_real_binned, real_pred_binned)
        else:  # Categorical target
            accuracy = accuracy_score(y_real, real_pred)
        
        # Compare with random baseline
        random_baseline = 1.0 / len(np.unique(y_real))
        
        # Inference risk
        inference_risk = (accuracy - random_baseline) / (1.0 - random_baseline)
        
        return {
            "attribute_inference_accuracy": accuracy,
            "attribute_inference_random_baseline": random_baseline,
            "attribute_inference_risk": inference_risk
        }

    def _evaluate_record_linkage(self, real_data: np.ndarray, 
                               synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate record linkage risk"""
        # Compute pairwise distances
        distances = euclidean_distances(real_data, synthetic_data)
        
        # Find minimum distances for each real record
        min_distances = np.min(distances, axis=1)
        
        # Find records that are very close to synthetic records
        threshold = np.percentile(min_distances, 5)  # 5th percentile
        close_matches = np.sum(min_distances < threshold)
        
        # Linkage risk
        linkage_risk = close_matches / len(real_data)
        
        # Average minimum distance
        avg_min_distance = np.mean(min_distances)
        
        # Maximum similarity (minimum distance)
        max_similarity = np.min(min_distances)
        
        return {
            "record_linkage_risk": linkage_risk,
            "record_linkage_avg_min_distance": avg_min_distance,
            "record_linkage_max_similarity": max_similarity,
            "record_linkage_close_matches": close_matches
        }
```

### Example 4: Domain-Specific Evaluator

```python
import numpy as np
from typing import Dict, Any, List
from sdg_core_lib.evaluate.base_evaluator import BaseEvaluator

class FinancialDataEvaluator(BaseEvaluator):
    """Evaluator specialized for financial data"""
    
    def __init__(self, real_data: Dataset, synthetic_data: Dataset):
        super().__init__(real_data, synthetic_data)

    def compute(self) -> Dict[str, Any]:
        """Compute financial data-specific metrics"""
        results = {}
        
        # Get data arrays
        real_array = self._real_data.get_computing_data()
        synthetic_array = self._synth_data.get_computing_data()
        
        # Financial metrics
        if real_array.shape[1] >= 1:
            # Assume first column is price/return series
            price_results = self._evaluate_price_series(
                real_array[:, 0], synthetic_array[:, 0]
            )
            results.update(price_results)
        
        if real_array.shape[1] >= 2:
            # Assume second column is volume
            volume_results = self._evaluate_volume_series(
                real_array[:, 1], synthetic_array[:, 1]
            )
            results.update(volume_results)
        
        if real_array.shape[1] >= 3:
            # Correlation structure
            correlation_results = self._evaluate_correlation_structure(
                real_array, synthetic_array
            )
            results.update(correlation_results)
        
        # Add metrics to report
        for metric_name, metric_value in results.items():
            self.add_metric(metric_name, metric_value)
        
        return results

    def _evaluate_price_series(self, real_prices: np.ndarray, 
                             synthetic_prices: np.ndarray) -> Dict[str, Any]:
        """Evaluate price series properties"""
        results = {}
        
        # Basic statistics
        results["price_mean_diff"] = abs(np.mean(real_prices) - np.mean(synthetic_prices))
        results["price_std_diff"] = abs(np.std(real_prices) - np.std(synthetic_prices))
        
        # Returns
        real_returns = np.diff(real_prices) / real_prices[:-1]
        synthetic_returns = np.diff(synthetic_prices) / synthetic_prices[:-1]
        
        results["return_mean_diff"] = abs(np.mean(real_returns) - np.mean(synthetic_returns))
        results["return_std_diff"] = abs(np.std(real_returns) - np.std(synthetic_returns))
        
        # Volatility clustering (GARCH-like behavior)
        real_volatility = np.abs(real_returns)
        synthetic_volatility = np.abs(synthetic_returns)
        
        # Autocorrelation of volatility
        real_vol_autocorr = self._calculate_autocorrelation(real_volatility, lags=5)
        synth_vol_autocorr = self._calculate_autocorrelation(synthetic_volatility, lags=5)
        
        results["volatility_autocorr_diff"] = np.mean(np.abs(real_vol_autocorr - synth_vol_autocorr))
        
        # Tail risk
        real_var_95 = np.percentile(real_returns, 5)
        synthetic_var_95 = np.percentile(synthetic_returns, 5)
        results["var_95_diff"] = abs(real_var_95 - synthetic_var_95)
        
        return results

    def _evaluate_volume_series(self, real_volume: np.ndarray, 
                               synthetic_volume: np.ndarray) -> Dict[str, Any]:
        """Evaluate volume series properties"""
        results = {}
        
        results["volume_mean_diff"] = abs(np.mean(real_volume) - np.mean(synthetic_volume))
        results["volume_std_diff"] = abs(np.std(real_volume) - np.std(synthetic_volume))
        
        # Volume autocorrelation
        real_vol_autocorr = self._calculate_autocorrelation(real_volume, lags=5)
        synth_vol_autocorr = self._calculate_autocorrelation(synthetic_volume, lags=5)
        
        results["volume_autocorr_diff"] = np.mean(np.abs(real_vol_autocorr - synth_vol_autocorr))
        
        return results

    def _evaluate_correlation_structure(self, real_data: np.ndarray, 
                                      synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate correlation structure preservation"""
        results = {}
        
        # Correlation matrices
        real_corr = np.corrcoef(real_data.T)
        synthetic_corr = np.corrcoef(synthetic_data.T)
        
        # Correlation matrix difference
        corr_diff = np.abs(real_corr - synthetic_corr)
        results["correlation_matrix_mean_diff"] = np.mean(corr_diff)
        results["correlation_matrix_max_diff"] = np.max(corr_diff)
        
        # Eigenvalue structure
        real_eigenvals = np.linalg.eigvals(real_corr)
        synthetic_eigenvals = np.linalg.eigvals(synthetic_corr)
        
        results["eigenvalue_mean_diff"] = np.mean(np.abs(real_eigenvals - synthetic_eigenvals))
        results["eigenvalue_max_diff"] = np.max(np.abs(real_eigenvals - synthetic_eigenvals))
        
        return results

    def _calculate_autocorrelation(self, series: np.ndarray, lags: int) -> np.ndarray:
        """Calculate autocorrelation for given lags"""
        autocorr = []
        series_centered = series - np.mean(series)
        
        for lag in range(1, lags + 1):
            if lag < len(series):
                corr = np.corrcoef(series_centered[:-lag], series_centered[lag:])[0, 1]
                autocorr.append(corr if not np.isnan(corr) else 0)
            else:
                autocorr.append(0)
        
        return np.array(autocorr)

class TimeSeriesEvaluator(BaseEvaluator):
    """Evaluator specialized for time series data"""
    
    def __init__(self, real_data: Dataset, synthetic_data: Dataset):
        super().__init__(real_data, synthetic_data)

    def compute(self) -> Dict[str, Any]:
        """Compute time series-specific metrics"""
        results = {}
        
        # Get data arrays
        real_array = self._real_data.get_computing_data()
        synthetic_array = self._synth_data.get_computing_data()
        
        # Temporal dependencies
        temporal_results = self._evaluate_temporal_dependencies(real_array, synthetic_array)
        results.update(temporal_results)
        
        # Seasonality
        seasonality_results = self._evaluate_seasonality(real_array, synthetic_array)
        results.update(seasonality_results)
        
        # Trend preservation
        trend_results = self._evaluate_trends(real_array, synthetic_array)
        results.update(trend_results)
        
        # Add metrics to report
        for metric_name, metric_value in results.items():
            self.add_metric(metric_name, metric_value)
        
        return results

    def _evaluate_temporal_dependencies(self, real_data: np.ndarray, 
                                      synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate temporal dependency preservation"""
        results = {}
        
        # Autocorrelation function comparison
        max_lags = min(20, len(real_data) // 4)
        
        real_autocorr = []
        synthetic_autocorr = []
        
        for col in range(real_data.shape[1]):
            real_col_autocorr = self._calculate_autocorrelation(
                real_data[:, col], max_lags
            )
            synth_col_autocorr = self._calculate_autocorrelation(
                synthetic_data[:, col], max_lags
            )
            
            real_autocorr.append(real_col_autocorr)
            synthetic_autocorr.append(synth_col_autocorr)
        
        # Compare autocorrelation functions
        autocorr_diff = np.mean([
            np.mean(np.abs(r - s)) 
            for r, s in zip(real_autocorr, synthetic_autocorr)
        ])
        
        results["autocorrelation_mean_diff"] = autocorr_diff
        
        return results

    def _evaluate_seasonality(self, real_data: np.ndarray, 
                             synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate seasonality preservation"""
        results = {}
        
        # Simple seasonality detection using FFT
        for col in range(real_data.shape[1]):
            real_fft = np.fft.fft(real_data[:, col])
            synthetic_fft = np.fft.fft(synthetic_data[:, col])
            
            # Compare dominant frequencies
            real_power = np.abs(real_fft) ** 2
            synthetic_power = np.abs(synthetic_fft) ** 2
            
            # Find top frequencies
            real_top_freqs = np.argsort(real_power)[-5:]
            synthetic_top_freqs = np.argsort(synthetic_power)[-5:]
            
            # Frequency overlap
            overlap = len(set(real_top_freqs) & set(synthetic_top_freqs)) / 5
            
            results[f"seasonality_freq_overlap_col_{col}"] = overlap
        
        return results

    def _evaluate_trends(self, real_data: np.ndarray, 
                        synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate trend preservation"""
        results = {}
        
        for col in range(real_data.shape[1]):
            # Linear trend
            x = np.arange(len(real_data))
            real_slope, _ = np.polyfit(x, real_data[:, col], 1)
            synthetic_slope, _ = np.polyfit(x, synthetic_data[:, col], 1)
            
            results[f"trend_slope_diff_col_{col}"] = abs(real_slope - synthetic_slope)
        
        return results

    def _calculate_autocorrelation(self, series: np.ndarray, max_lags: int) -> np.ndarray:
        """Calculate autocorrelation function"""
        n = len(series)
        autocorr = []
        series_centered = series - np.mean(series)
        
        for lag in range(1, max_lags + 1):
            if lag < n:
                corr = np.corrcoef(series_centered[:-lag], series_centered[lag:])[0, 1]
                autocorr.append(corr if not np.isnan(corr) else 0)
            else:
                autocorr.append(0)
        
        return np.array(autocorr)
```

## Evaluator Composition and Pipelines

Create composite evaluators that combine multiple evaluation approaches:

```python
class CompositeEvaluator(BaseEvaluator):
    """Composite evaluator that combines multiple evaluators"""
    
    def __init__(self, real_data: Dataset, synthetic_data: Dataset,
                 evaluators: List[BaseEvaluator]):
        super().__init__(real_data, synthetic_data)
        self.evaluators = evaluators

    def compute(self) -> Dict[str, Any]:
        """Compute all evaluation metrics"""
        results = {}
        
        for evaluator in self.evaluators:
            evaluator_results = evaluator.compute()
            
            # Prefix results with evaluator name
            evaluator_name = evaluator.__class__.__name__
            prefixed_results = {
                f"{evaluator_name}_{key}": value 
                for key, value in evaluator_results.items()
            }
            
            results.update(prefixed_results)
            
            # Also add to individual evaluator reports
            for metric_name, metric_value in evaluator_results.items():
                self.add_metric(f"{evaluator_name}_{metric_name}", metric_value)
        
        return results

class AdaptiveEvaluator(BaseEvaluator):
    """Evaluator that adapts based on data characteristics"""
    
    def __init__(self, real_data: Dataset, synthetic_data: Dataset):
        super().__init__(real_data, synthetic_data)
        self.selected_evaluators = []

    def compute(self) -> Dict[str, Any]:
        """Select and run appropriate evaluators"""
        # Analyze data characteristics
        data_characteristics = self._analyze_data()
        
        # Select evaluators based on characteristics
        self.selected_evaluators = self._select_evaluators(data_characteristics)
        
        # Run selected evaluators
        results = {}
        for evaluator in self.selected_evaluators:
            evaluator_results = evaluator.compute()
            results.update(evaluator_results)
            
            # Add to report
            for metric_name, metric_value in evaluator_results.items():
                self.add_metric(metric_name, metric_value)
        
        return results

    def _analyze_data(self) -> Dict[str, Any]:
        """Analyze data characteristics"""
        real_array = self._real_data.get_computing_data()
        
        return {
            "data_type": self._detect_data_type(real_array),
            "dimensionality": real_array.shape[1],
            "sample_size": real_array.shape[0],
            "has_temporal_structure": self._has_temporal_structure(real_array),
            "is_financial_data": self._is_financial_data(real_array)
        }

    def _select_evaluators(self, characteristics: Dict[str, Any]) -> List[BaseEvaluator]:
        """Select appropriate evaluators based on characteristics"""
        evaluators = []
        
        # Always include statistical similarity
        evaluators.append(StatisticalSimilarityEvaluator(self._real_data, self._synth_data))
        
        # Include ML evaluator if enough data
        if characteristics["sample_size"] > 100:
            evaluators.append(MachineLearningEvaluator(self._real_data, self._synth_data))
        
        # Include privacy evaluator for sensitive data
        if characteristics["is_financial_data"]:
            evaluators.append(PrivacyEvaluator(self._real_data, self._synth_data))
        
        # Include domain-specific evaluators
        if characteristics["is_financial_data"]:
            evaluators.append(FinancialDataEvaluator(self._real_data, self._synth_data))
        
        if characteristics["has_temporal_structure"]:
            evaluators.append(TimeSeriesEvaluator(self._real_data, self._synth_data))
        
        return evaluators
```

## Testing Custom Evaluators

```python
import pytest
import numpy as np
from your_module import (
    StatisticalSimilarityEvaluator,
    MachineLearningEvaluator,
    PrivacyEvaluator,
    FinancialDataEvaluator
)

class TestCustomEvaluators:
    def test_statistical_similarity_evaluator(self):
        """Test statistical similarity evaluator"""
        from your_module import Table
        
        # Create dummy data
        real_data = np.random.randn(100, 3)
        synthetic_data = np.random.randn(100, 3)
        
        # Create dataset objects (simplified)
        real_dataset = Table([])  # Would need proper initialization
        synth_dataset = Table([])
        
        # Set computing data
        real_dataset.get_computing_data = lambda: real_data
        synth_dataset.get_computing_data = lambda: synthetic_data
        
        evaluator = StatisticalSimilarityEvaluator(real_dataset, synth_dataset)
        results = evaluator.compute()
        
        # Check that metrics are computed
        assert "overall_similarity" in results
        assert "column_0_mean_diff" in results
        assert "column_0_ks_statistic" in results

    def test_machine_learning_evaluator(self):
        """Test machine learning evaluator"""
        # Similar setup as above
        real_data = np.random.randn(200, 5)
        synthetic_data = np.random.randn(200, 5)
        
        real_dataset = Table([])
        synth_dataset = Table([])
        
        real_dataset.get_computing_data = lambda: real_data
        synth_dataset.get_computing_data = lambda: synthetic_data
        
        evaluator = MachineLearningEvaluator(real_dataset, synth_dataset)
        results = evaluator.compute()
        
        # Check ML-specific metrics
        assert "distinguishability_accuracy" in results
        assert "feature_importance_correlation" in results
        assert "classification_combined_accuracy" in results

    def test_privacy_evaluator(self):
        """Test privacy evaluator"""
        real_data = np.random.randn(100, 4)
        synthetic_data = np.random.randn(100, 4)
        
        real_dataset = Table([])
        synth_dataset = Table([])
        
        real_dataset.get_computing_data = lambda: real_data
        synth_dataset.get_computing_data = lambda: synthetic_data
        
        evaluator = PrivacyEvaluator(real_dataset, synth_dataset)
        results = evaluator.compute()
        
        # Check privacy-specific metrics
        assert "privacy_ratio" in results
        assert "membership_inference_attack_accuracy" in results
        assert "record_linkage_risk" in results
```

## Best Practices

1. **Metric Selection**: Choose metrics appropriate for your data type and domain
2. **Interpretability**: Provide clear explanations of what each metric means
3. **Normalization**: Normalize metrics to comparable scales where possible
4. **Statistical Validity**: Ensure statistical tests are applied correctly
5. **Computational Efficiency**: Consider performance for large datasets
6. **Robustness**: Handle edge cases and invalid data gracefully
7. **Documentation**: Provide clear docstrings and parameter descriptions
8. **Testing**: Create comprehensive tests for all evaluation functionality
9. **Reproducibility**: Ensure consistent results across runs
10. **Domain Knowledge**: Incorporate domain-specific evaluation criteria

## Integration with Existing Systems

### Evaluator Registry

```python
class EvaluatorRegistry:
    """Registry for custom evaluators"""
    
    _evaluators = {}
    
    @classmethod
    def register(cls, name: str, evaluator_class: type):
        """Register an evaluator class"""
        cls._evaluators[name] = evaluator_class
    
    @classmethod
    def get_evaluator(cls, name: str) -> type:
        """Get an evaluator class by name"""
        if name not in cls._evaluators:
            raise ValueError(f"Unknown evaluator: {name}")
        return cls._evaluators[name]
    
    @classmethod
    def list_evaluators(cls) -> Dict[str, str]:
        """List all registered evaluators"""
        return {
            name: evaluator_class.__doc__ or "No description available"
            for name, evaluator_class in cls._evaluators.items()
        }

# Register custom evaluators
EvaluatorRegistry.register("statistical_similarity", StatisticalSimilarityEvaluator)
EvaluatorRegistry.register("machine_learning", MachineLearningEvaluator)
EvaluatorRegistry.register("privacy", PrivacyEvaluator)
EvaluatorRegistry.register("financial", FinancialDataEvaluator)
```

This guide provides the foundation for creating custom evaluators. Adapt the examples to your specific evaluation requirements and domain needs.
