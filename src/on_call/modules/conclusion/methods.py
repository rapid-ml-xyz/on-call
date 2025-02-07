def get_consolidated_summary(time_series_summary_md, feature_summaries_md):
    context = """
    Create a concise executive summary that combines:
    1. The time series analysis of the bike sharing model performance
    2. The key patterns and insights from feature-level analyses across all time windows
    
    Focus on the most important findings and their relationships. Avoid repetition and highlight critical insights.
    """

    prompt = f"""
    Please provide a strategic summary combining these analyses. Format your response in markdown with these sections:

    # Model Performance Summary
    [Key findings about model behavior and performance]

    # Data Quality and Drift Insights
    [Key patterns in features and data quality]

    # Critical Findings and Recommendations
    [Most important takeaways and suggested actions]

    Here are the analyses to summarize:

    Time Series Analysis:
    {time_series_summary_md}

    Feature Analyses:
    {' '.join(feature_summaries_md)}
    
    Requirements:
    - Synthesize into one cohesive narrative with the sections above
    - Focus on critical findings and their relationships
    - Keep it concise but comprehensive
    - Ensure proper markdown formatting with headers and lists
    """

    return context, prompt
