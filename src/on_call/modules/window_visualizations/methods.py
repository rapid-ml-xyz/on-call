import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def extract_report_metrics(reports):
    metrics_data = []

    for reg_report, drift_report in reports:
        reg_data = json.loads(reg_report.json())
        drift_data = json.loads(drift_report.json())
        for metric in reg_data.get('metrics', []):
            if metric.get('metric') == 'RegressionQualityMetric':
                result = metric.get('result', {}).get('current', {})
                metrics_data.append({
                    'r2': result.get('r2_score', 0),
                    'rmse': result.get('rmse', 0),
                    'mean_error': result.get('mean_error', 0),
                    'error_std': result.get('error_std', 0)
                })

    return metrics_data


def create_consolidated_performance_plot(reports, windows):
    metrics_data = extract_report_metrics(reports)
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Model Performance Metrics', 'Error Distribution'),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )

    time_points = [f"Window {i+1}" for i in range(len(metrics_data))]
    fig.add_trace(
        go.Scatter(x=time_points,
                   y=[m['r2'] for m in metrics_data],
                   name="R² Score",
                   mode='lines+markers',
                   line=dict(color='blue')),
        row=1, col=1, secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=time_points,
                   y=[m['rmse'] for m in metrics_data],
                   name="RMSE",
                   mode='lines+markers',
                   line=dict(color='red')),
        row=1, col=1, secondary_y=True
    )

    fig.add_trace(
        go.Box(y=[m['mean_error'] for m in metrics_data],
               name="Mean Error",
               boxpoints='all',
               jitter=0.3,
               pointpos=-1.8,
               marker_color='lightblue'),
        row=2, col=1
    )

    fig.add_trace(
        go.Box(y=[m['error_std'] for m in metrics_data],
               name="Error Std",
               boxpoints='all',
               jitter=0.3,
               pointpos=-1.8,
               marker_color='lightgreen'),
        row=2, col=1
    )

    fig.update_layout(
        title_text="Model Performance Analysis",
        height=900,
        showlegend=True,
        template='plotly_white',
        margin=dict(t=150),
    )

    fig.update_yaxes(title_text="R² Score", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="RMSE", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Error Metrics", row=2, col=1)

    for i, (start_time, end_time) in enumerate(windows):
        fig.add_annotation(
            x=f"Window {i+1}",
            y=1.25,
            text=f"{start_time.strftime('%Y-%m-%d %H:%M')} to{end_time.strftime('%Y-%m-%d %H:%M')}",
            showarrow=False,
            xanchor='center',
            yanchor='bottom',
            yref='paper',
            font=dict(size=10)
        )

    return fig


def create_drift_analysis_plot(reports):
    """Create drift analysis visualization."""
    fig = make_subplots(rows=1, cols=1)

    drift_scores = []
    features = set()

    for i, (_, drift_report) in enumerate(reports):
        drift_data = json.loads(drift_report.json())

        for metric in drift_data.get('metrics', []):
            if metric.get('metric') == 'ColumnDriftMetric':
                result = metric.get('result', {})
                feature = result.get('column_name', '')
                features.add(feature)
                drift_scores.append({
                    'window': f"Window {i+1}",
                    'feature': feature,
                    'score': 1 - result.get('drift_score', 0),
                    'detected': result.get('drift_detected', False)
                })

    windows = sorted(set(d['window'] for d in drift_scores))
    features = sorted(features)

    z_data = np.zeros((len(features), len(windows)))
    text_data = [['' for _ in windows] for _ in features]

    for score in drift_scores:
        i = features.index(score['feature'])
        j = windows.index(score['window'])
        z_data[i][j] = score['score']
        text_data[i][j] = f"Score: {score['score']:.3f}Drift: {'Yes' if score['detected'] else 'No'}"

    fig.add_trace(
        go.Heatmap(
            z=z_data,
            x=windows,
            y=features,
            text=text_data,
            hoverongaps=False,
            colorscale='RdYlBu_R',
            colorbar=dict(title='Drift Score')
        )
    )

    fig.update_layout(
        title_text="Target Drift Analysis",
        height=600,
        template='plotly_white',
        yaxis=dict(title='Targets'),
        xaxis=dict(title='Time Windows')
    )

    return fig
