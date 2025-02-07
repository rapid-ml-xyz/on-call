import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def extract_drift_metrics(window_analyses):
    drift_data = []
    for analysis in window_analyses:
        drift_json = json.loads(analysis.drift_report.json())
        window_label = f"Window {analysis.window_id}"

        for metric in drift_json.get('metrics', []):
            if metric.get('metric') == 'DataDriftTable':
                drift_by_columns = metric.get('result', {}).get('drift_by_columns', {})
                for feature, metrics in drift_by_columns.items():
                    drift_data.append({
                        'window': window_label,
                        'feature': feature,
                        'drift_score': 1 - metrics.get('drift_score', 0),
                        'drift_detected': metrics.get('drift_detected', False),
                        'feature_type': metrics.get('column_type', 'unknown')
                    })
    return drift_data


def extract_quality_metrics(window_analyses):
    quality_data = []
    for analysis in window_analyses:
        quality_json = json.loads(analysis.quality_report.json())
        window_label = f"Window {analysis.window_id}"

        for metric in quality_json.get('metrics', []):
            if metric.get('metric') == 'ColumnSummaryMetric':
                result = metric.get('result', {})
                feature = result.get('column_name')
                current = result.get('current_characteristics', {})
                reference = result.get('reference_characteristics', {})

                if all([feature, current, reference]):
                    quality_data.append({
                        'window': window_label,
                        'feature': feature,
                        'mean_diff': abs(current.get('mean', 0) - reference.get('mean', 0)),
                        'std_diff': abs(current.get('std', 0) - reference.get('std', 0)),
                        'unique_diff': current.get('unique_percentage', 0) - reference.get('unique_percentage', 0)
                    })
    return quality_data


def create_feature_analysis_plot(window_analyses):
    drift_data = extract_drift_metrics(window_analyses)
    quality_data = extract_quality_metrics(window_analyses)

    if not drift_data:
        return None

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Feature Drift Analysis', 'Feature Distribution Changes'),
        vertical_spacing=0.2,
        row_heights=[0.6, 0.4]
    )

    windows = sorted(set(d['window'] for d in drift_data))
    features = sorted(set(d['feature'] for d in drift_data))

    drift_z = np.zeros((len(features), len(windows)))
    drift_text = [['' for _ in windows] for _ in features]

    for d in drift_data:
        i = features.index(d['feature'])
        j = windows.index(d['window'])
        drift_z[i][j] = d['drift_score']
        drift_text[i][j] = (f"Score: {d['drift_score']:.3f}\n"
                            f"Drift Detected: {'Yes' if d['drift_detected'] else 'No'}\n"
                            f"Type: {d['feature_type']}")

    fig.add_trace(
        go.Heatmap(
            z=drift_z,
            x=windows,
            y=features,
            text=drift_text,
            hoverongaps=False,
            colorscale='RdYlBu_R',
            colorbar=dict(
                title='Drift Score',
                y=0.8,
                len=0.5,
                ticktext=['Low Drift', 'Medium Drift', 'High Drift'],
                tickvals=[0, 0.5, 1]
            )
        ),
        row=1, col=1
    )

    if quality_data:
        for feature in features:
            feature_data = [d for d in quality_data if d['feature'] == feature]
            if feature_data:
                fig.add_trace(
                    go.Bar(
                        x=windows,
                        y=[d['mean_diff'] for d in feature_data],
                        name=f"{feature} (Mean Diff)",
                        visible='legendonly'
                    ),
                    row=2, col=1
                )

    fig.update_layout(
        height=1000,
        showlegend=True,
        template='plotly_white',
        title_text="Critical Windows Feature Analysis",
        title_x=0.5,
        legend=dict(
            yanchor="top",
            y=0.45,
            xanchor="left",
            x=1.0
        ),
        margin=dict(t=100, b=100, r=200)
    )

    fig.update_yaxes(title_text="Features", row=1, col=1)
    fig.update_yaxes(title_text="Distribution Change", row=2, col=1)
    fig.update_xaxes(title_text="Time Windows", row=2, col=1)

    fig.add_annotation(
        text="Red indicates high drift, Blue indicates low drift",
        xref="paper",
        yref="paper",
        x=0,
        y=1.05,
        showarrow=False,
        font=dict(size=10)
    )

    return fig


def create_feature_summary_plot(window_analyses):
    drift_data = extract_drift_metrics(window_analyses)
    quality_data = extract_quality_metrics(window_analyses)

    if not drift_data:
        return None

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Feature Drift Scores', 'Feature Distribution Changes'),
        specs=[[{"type": "box"}, {"type": "box"}]],
        horizontal_spacing=0.15
    )

    features = sorted(set(d['feature'] for d in drift_data))

    drift_by_feature = {f: [] for f in features}
    for d in drift_data:
        drift_by_feature[d['feature']].append(d['drift_score'])

    drift_stats = {
        f: {
            'median': np.median(scores),
            'q1': np.percentile(scores, 25),
            'q3': np.percentile(scores, 75),
            'count': len(scores)
        }
        for f, scores in drift_by_feature.items()
    }

    fig.add_trace(
        go.Box(
            x=[f for f, scores in drift_by_feature.items() for _ in scores],
            y=[s for scores in drift_by_feature.values() for s in scores],
            name="Drift Score",
            boxpoints='outliers',
            marker=dict(color='rgba(255, 65, 54, 0.7)'),
            line=dict(color='rgba(255, 65, 54, 1)'),
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Median: %{customdata[0]:.3f}<br>" +
                "Q1: %{customdata[1]:.3f}<br>" +
                "Q3: %{customdata[2]:.3f}<br>" +
                "Count: %{customdata[3]}<br>" +
                "<extra></extra>"
            ),
            customdata=[[
                drift_stats[f]['median'],
                drift_stats[f]['q1'],
                drift_stats[f]['q3'],
                drift_stats[f]['count']
            ] for f, scores in drift_by_feature.items() for _ in scores]
        ),
        row=1, col=1
    )

    if quality_data:
        quality_by_feature = {f: [] for f in features}
        for q in quality_data:
            quality_by_feature[q['feature']].append(q['mean_diff'])

        quality_stats = {
            f: {
                'median': np.median(diffs),
                'q1': np.percentile(diffs, 25),
                'q3': np.percentile(diffs, 75),
                'count': len(diffs)
            }
            for f, diffs in quality_by_feature.items() if diffs
        }

        fig.add_trace(
            go.Box(
                x=[f for f, diffs in quality_by_feature.items() for _ in diffs],
                y=[d for diffs in quality_by_feature.values() for d in diffs],
                name="Mean Difference",
                boxpoints='outliers',
                marker=dict(color='rgba(44, 160, 44, 0.7)'),
                line=dict(color='rgba(44, 160, 44, 1)'),
                hovertemplate=(
                    "<b>%{x}</b><br>" +
                    "Median: %{customdata[0]:.3f}<br>" +
                    "Q1: %{customdata[1]:.3f}<br>" +
                    "Q3: %{customdata[2]:.3f}<br>" +
                    "Count: %{customdata[3]}<br>" +
                    "<extra></extra>"
                ),
                customdata=[[
                    quality_stats[f]['median'],
                    quality_stats[f]['q1'],
                    quality_stats[f]['q3'],
                    quality_stats[f]['count']
                ] for f, diffs in quality_by_feature.items() for _ in diffs]
            ),
            row=1, col=2
        )

    fig.update_layout(
        height=600,
        showlegend=True,
        template='plotly_white',
        title_text="Feature Distribution Summary",
        title_x=0.5,
        xaxis_tickangle=-45,
        margin=dict(t=100, b=100, l=50, r=50),
        plot_bgcolor='white',
        boxmode='group',
        font=dict(size=12)
    )

    for i in [1, 2]:
        fig.update_xaxes(
            title_text="Features",
            row=1, col=i,
            gridcolor='lightgrey',
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True
        )

    fig.update_yaxes(
        title_text="Drift Score",
        row=1, col=1,
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )

    fig.update_yaxes(
        title_text="Mean Difference",
        row=1, col=2,
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )

    return fig
