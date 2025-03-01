import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from on_call.modules.time_series_reports import generate_time_windows_datetime


def clean_regression_json(data):
    if data.get('metrics'):
        for metric in data['metrics']:
            if metric.get('metric') == 'RegressionQualityMetric':
                if metric.get('result'):
                    metric['result'].pop('error_normality', None)
                    return metric


def clean_drift_json(data):
    if data.get('metrics'):
        for metric in data['metrics']:
            if metric.get('result'):
                metric['result'].pop('current', None)
                metric['result'].pop('reference', None)
    return data


class WindowData:
    def __init__(self, regression_json, drift_json):
        regression_dict = json.loads(regression_json)
        self.regression_json = json.dumps(clean_regression_json(regression_dict))

        drift_dict = json.loads(drift_json)
        self.drift_json = json.dumps(clean_drift_json(drift_dict))


def parse_window_data(window: WindowData) -> Tuple[Dict, Dict]:
    reg_data = json.loads(window.regression_json)
    drift_data = json.loads(window.drift_json)
    return reg_data, drift_data


def extract_drift_info(drift_data: Dict) -> Tuple[List[str], Dict]:
    drifted = []
    scores = {}

    for metric in drift_data.get('metrics', []):
        if metric.get('metric') == 'ColumnDriftMetric':
            result = metric.get('result', {})
            name = result.get('column_name')
            if result.get('drift_detected'):
                drifted.append(name)
            scores[name] = {
                'score': result.get('drift_score'),
                'detected': result.get('drift_detected')
            }

    return drifted, scores


def format_metrics(metrics: Dict) -> Dict:
    return {
        "r2_score": round(float(metrics.get('r2_score', 0)), 3),
        "rmse": round(float(metrics.get('rmse', 0)), 2),
        "mean_error": round(float(metrics.get('mean_error', 0)), 2),
        "error_std": round(float(metrics.get('error_std', 0)), 2)
    }


def format_error_stats(category: str, metrics: Dict) -> Dict:
    stats = metrics.get('underperformance', {}).get(category, {})
    return {
        "mean": round(float(stats.get('mean_error', 0)), 2),
        "std": round(float(stats.get('std_error', 0)), 2)
    }


def format_feature_stats(error_bias: Dict) -> Dict:
    return {
        feature: {
            "range": round(float(metrics.get('current_range', 0)), 2),
            "majority_value": round(float(metrics.get('current_majority', 0)), 3),
            "under_value": round(float(metrics.get('current_under', 0)), 3),
            "over_value": round(float(metrics.get('current_over', 0)), 3)
        }
        for feature, metrics in error_bias.items()
        if metrics.get('feature_type') == 'num'
    }


def create_window_report(
    window_id: int,
    time_range: Tuple[str, str],
    reg_data: Dict,
    drift_data: Dict
) -> Tuple[Dict, List[str]]:
    current = reg_data.get('result', {}).get('current', {})
    error_bias = reg_data.get('result', {}).get('error_bias', {})
    drifted_features, drift_scores = extract_drift_info(drift_data)

    return {
        "window_id": window_id,
        "time_range": {"start": time_range[0], "end": time_range[1]},
        "metrics": format_metrics(current),
        "error_distribution": {
            category: format_error_stats(category, current)
            for category in ['majority', 'underestimation', 'overestimation']
        },
        "drift_analysis": {
            "drifted_features": drifted_features,
            "feature_scores": drift_scores
        },
        "feature_performance": format_feature_stats(error_bias)
    }, drifted_features


def create_critical_window_entry(
    window_id: int,
    time_range: Tuple[str, str],
    drifted_features: List[str],
    r2_score: float
) -> Dict:
    return {
        "window_id": window_id,
        "time_range": {"start": time_range[0], "end": time_range[1]},
        "issues": {
            "drifted_features": drifted_features,
            "poor_performance": r2_score < 0.7
        }
    }


def determine_status(n_critical: int, total: int) -> str:
    if n_critical > total // 2:
        return "critical"
    elif n_critical > 0:
        return "warning"
    return "stable"


def analyze_monitoring_data_json(window_jsons: List[WindowData], current: pd.DataFrame) -> Dict:
    try:
        windows = generate_time_windows_datetime(df=current)

        report = {
            "metadata": {
                "total_windows": len(window_jsons),
                "analysis_timestamp": datetime.now().isoformat(),
                "model_type": "RandomForestRegressor"
            },
            "windows": [],
            "summary": {
                "critical_windows": {"count": 0, "details": []},
                "drifted_features": set()
            }
        }

        for idx, window in enumerate(window_jsons):
            time_range = (windows[idx][0], windows[idx][1]) if idx < len(windows) else ("unknown", "unknown")

            reg_data, drift_data = parse_window_data(window)
            window_report, drifted_features = create_window_report(
                idx + 1, time_range, reg_data, drift_data
            )

            is_critical = bool(
                drifted_features or
                window_report["metrics"]["r2_score"] < 0.7
            )

            if is_critical:
                report["summary"]["critical_windows"]["count"] += 1
                report["summary"]["critical_windows"]["details"].append(
                    create_critical_window_entry(
                        idx + 1,
                        time_range,
                        drifted_features,
                        window_report["metrics"]["r2_score"]
                    )
                )

            report["summary"]["drifted_features"].update(drifted_features)
            report["windows"].append(window_report)

        report["summary"]["drifted_features"] = list(report["summary"]["drifted_features"])
        report["summary"]["status"] = determine_status(
            report["summary"]["critical_windows"]["count"],
            len(window_jsons)
        )

        return report

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "metadata": {
                "total_windows": len(window_jsons),
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
