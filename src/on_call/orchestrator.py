from typing import Any, Dict, Optional
from src.on_call.modules.eda import EDA
from src.on_call.modules.cohort_identification import CohortIdentifier
from src.on_call.modules.metric_analyzer import MetricAnalyzer
from src.on_call.modules.distribution_analyzer import DistributionAnalyzer
from src.on_call.modules.error_analyzer import ErrorAnalyzer
from src.on_call.modules.feature_analyzer import FeatureAnalyzer
from src.on_call.modules.explainable_ai import ExplainableAIModule
from src.on_call.modules.model_behavior_analyzer import ModelBehaviorAnalyzer
from src.on_call.utils.config_manager import ConfigManager
from src.on_call.utils.logger import get_logger


class Orchestrator:
    """
    The main class that orchestrates the debugging flow. Provides a high-level
    interface to run the RCA pipeline in a structured manner.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None
    ):
        """
        - config_path: Path to a YAML/JSON config file with debugging parameters.
        - extra_params: Additional parameters that can override or extend the config.
        """
        self.logger = get_logger()
        self.config_manager = ConfigManager(config_path, extra_params)
        self.logger.info("Initializing Orchestrator with provided configuration.")

        # Instantiate modules (lazy loading could be used if desired)
        self.eda = EDA(self.config_manager)
        self.cohort_identifier = CohortIdentifier(self.config_manager)
        self.metric_analyzer = MetricAnalyzer(self.config_manager)
        self.distribution_analyzer = DistributionAnalyzer(self.config_manager)
        self.error_analyzer = ErrorAnalyzer(self.config_manager)
        self.feature_analyzer = FeatureAnalyzer(self.config_manager)
        self.explainable_ai = ExplainableAIModule(self.config_manager)
        self.model_behavior_analyzer = ModelBehaviorAnalyzer(self.config_manager)

    def run_full_debug_pipeline(self, data, predictions, labels=None):
        """
        Execute the full debugging pipeline in a recommended order.
        This is a simplified version of the flowchart logic, but you can adapt
        it to detect 'sudden drop' vs 'gradual decline', etc.
        """
        # 1. EDA
        self.logger.info("Starting EDA...")
        self.eda.perform_eda(data)

        # 2. Metric Analysis
        self.logger.info("Analyzing performance metrics...")
        metrics_summary = self.metric_analyzer.evaluate_metrics(
            data, predictions, labels
        )

        # 3. Cohort Identification
        self.logger.info("Identifying cohorts...")
        cohorts = self.cohort_identifier.find_high_impact_cohorts(
            data, predictions, labels
        )

        # 4. Distribution Analysis
        self.logger.info("Analyzing distributions...")
        distribution_report = self.distribution_analyzer.compare_distributions(data)

        # 5. Error Analysis (Confusion matrix, residuals, etc.)
        self.logger.info("Performing error analysis...")
        error_insights = self.error_analyzer.analyze_errors(data, predictions, labels)

        # 6. Feature Analysis
        self.logger.info("Analyzing feature importance...")
        feature_insights = self.feature_analyzer.analyze_features(data, predictions)

        # 7. Explainability (SHAP, LIME, etc.)
        self.logger.info("Generating explainability reports...")
        explanations = self.explainable_ai.explain_predictions(data, predictions)

        # 8. Model Behavior Analysis (Global vs. local anomalies)
        self.logger.info("Assessing overall model behavior...")
        behavior_report = self.model_behavior_analyzer.analyze_behavior(
            data, predictions, labels
        )

        # Combine or output results
        self.logger.info("RCA Pipeline Completed.")
        return {
            "metrics_summary": metrics_summary,
            "cohorts": cohorts,
            "distribution_report": distribution_report,
            "error_insights": error_insights,
            "feature_insights": feature_insights,
            "explanations": explanations,
            "behavior_report": behavior_report
        }

    def stepwise_debug(self, step_name: str, data, predictions, labels=None):
        """
        If a user wants to run just one step of the pipeline, they can call this function.
        For example, step_name can be 'eda', 'metric_analysis', etc.
        """
        if step_name == "eda":
            return self.eda.perform_eda(data)
        elif step_name == "metric_analysis":
            return self.metric_analyzer.evaluate_metrics(data, predictions, labels)
        # ... Add other steps as needed
        else:
            self.logger.warning(f"Unknown step name: {step_name}")
            return None
