import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from tensorflow import keras

class ExplainableAI:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
    def calculate_shap_values(self, X_sample):
        """Calculate SHAP values for model explanation"""
        explainer = shap.TreeExplainer(self.model.xgb_model)
        shap_values = explainer.shap_values(X_sample)
        
        return explainer, shap_values
    
    def plot_shap_summary(self, X_sample, shap_values, max_display=20):
        """Plot SHAP summary plot"""
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, 
                         feature_names=self.feature_names,
                         max_display=max_display,
                         show=False)
        plt.tight_layout()
        plt.savefig('static/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_shap_bar(self, shap_values):
        """Plot SHAP bar plot"""
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, plot_type="bar", 
                         feature_names=self.feature_names,
                         max_display=15,
                         show=False)
        plt.tight_layout()
        plt.savefig('static/shap_bar.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, X_sample, shap_values, instance_idx=0):
        """Plot force plot for individual prediction"""
        explainer = shap.TreeExplainer(self.model.xgb_model)
        
        # For multi-class, we need to plot for each class
        for class_idx in range(shap_values.shape[0]):
            shap.initjs()
            shap.force_plot(
                explainer.expected_value[class_idx],
                shap_values[class_idx][instance_idx, :],
                X_sample[instance_idx, :],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            plt.title(f'SHAP Force Plot - Class {class_idx}')
            plt.tight_layout()
            plt.savefig(f'static/shap_force_class_{class_idx}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_explanation_report(self, X_sample, y_pred, y_prob, instance_idx=0):
        """Generate detailed explanation report for a prediction"""
        explainer, shap_values = self.calculate_shap_values(X_sample)
        
        # Get top contributing features
        mean_shap = np.abs(shap_values).mean(axis=0)
        top_features_idx = np.argsort(mean_shap)[::-1][:5]
        top_features = [self.feature_names[i] for i in top_features_idx]
        
        report = {
            'predicted_class': int(y_pred[instance_idx]),
            'predicted_probability': float(max(y_prob[instance_idx])),
            'class_probabilities': {
                'Low Risk': float(y_prob[instance_idx][0]),
                'Medium Risk': float(y_prob[instance_idx][1]),
                'High Risk': float(y_prob[instance_idx][2])
            },
            'top_contributing_features': top_features,
            'feature_contributions': {}
        }
        
        # Get contributions for each feature
        for i, feature in enumerate(self.feature_names):
            if i < shap_values.shape[1]:
                contribution = float(shap_values[y_pred[instance_idx], instance_idx, i])
                report['feature_contributions'][feature] = contribution
        
        return report
    
    def visualize_behavioral_patterns(self, X_latent, y_true, y_pred):
        """Visualize behavioral patterns in latent space"""
        from sklearn.manifold import TSNE
        
        # Reduce dimensions for visualization
        tsne = TSNE(n_components=2, random_state=42)
        X_2d = tsne.fit_transform(X_latent)
        
        plt.figure(figsize=(12, 8))
        
        # Plot actual vs predicted
        plt.subplot(1, 2, 1)
        scatter1 = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter1)
        plt.title('Actual Risk Levels in Latent Space')
        
        plt.subplot(1, 2, 2)
        scatter2 = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter2)
        plt.title('Predicted Risk Levels in Latent Space')
        
        plt.tight_layout()
        plt.savefig('static/latent_space_visualization.png', dpi=300)
        plt.show()
        
    def drift_detection(self, X_behavioral_latent, window_size=50):
        """Detect behavioral drift over time"""
        # Calculate moving statistics
        n_samples = len(X_behavioral_latent)
        drifts = []
        
        for i in range(window_size, n_samples, window_size):
            window = X_behavioral_latent[i-window_size:i]
            prev_window = X_behavioral_latent[i-2*window_size:i-window_size]
            
            # Calculate mean difference (simple drift detection)
            mean_diff = np.abs(window.mean(axis=0) - prev_window.mean(axis=0)).mean()
            drifts.append(mean_diff)
        
        # Plot drift detection
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(drifts)), drifts, marker='o', linestyle='-')
        plt.axhline(y=np.mean(drifts) + 2*np.std(drifts), color='r', linestyle='--', label='Drift Threshold')
        plt.xlabel('Window Index')
        plt.ylabel('Behavioral Drift Score')
        plt.title('Behavioral Drift Detection Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('static/drift_detection.png', dpi=300)
        plt.show()
        
        return drifts

def run_explainability_analysis(model, data_dict):
    """Run complete explainability analysis"""
    # Prepare combined features for XGBoost explainability
    X_beh_test_latent = model.extract_features(data_dict['X_beh_test'])
    X_combined_test = np.concatenate([X_beh_test_latent, data_dict['X_demo_test']], axis=1)
    
    # Create feature names
    feature_names = (
        [f'Behavioral_Latent_{i}' for i in range(X_beh_test_latent.shape[1])] +
        data_dict['demographic_feature_names']
    )
    
    # Initialize explainer
    explainer = ExplainableAI(model, feature_names)
    
    # Sample data for SHAP (use smaller sample for speed)
    sample_size = min(100, len(X_combined_test))
    X_sample = X_combined_test[:sample_size]
    
    # Get predictions for sample
    predictions, probabilities = model.predict(
        data_dict['X_beh_test'][:sample_size],
        data_dict['X_demo_test'][:sample_size]
    )
    
    # Calculate SHAP values
    print("Calculating SHAP values...")
    shap_explainer, shap_values = explainer.calculate_shap_values(X_sample)
    
    # Generate visualizations
    print("Generating SHAP visualizations...")
    explainer.plot_shap_summary(X_sample, shap_values)
    explainer.plot_shap_bar(shap_values)
    
    # Generate individual explanation
    print("Generating individual explanation report...")
    report = explainer.generate_explanation_report(
        X_sample, predictions, probabilities, instance_idx=0
    )
    
    # Visualize behavioral patterns
    print("Visualizing behavioral patterns...")
    explainer.visualize_behavioral_patterns(
        X_beh_test_latent, 
        data_dict['y_test'][:len(X_beh_test_latent)], 
        predictions[:len(X_beh_test_latent)]
    )
    
    # Detect behavioral drift
    print("Detecting behavioral drift...")
    drifts = explainer.drift_detection(X_beh_test_latent)
    
    print("\nExplainability analysis completed!")
    return explainer, report