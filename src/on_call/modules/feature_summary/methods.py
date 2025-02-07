def get_features_nlp_setup(window):
    context = """
    How does the feature distribution & quality changed compared to baseline?
    Look at the data drift report & data quality report.
    Focus on the columns that have changed, and try to answer "how" & "why."
    """

    prompt = f"""
    Please provide a clear, concise natural language summary of the following JSON data:
    {window.drift_report.json()}
    {window.quality_report.json()}
    Here's the timeframe: [{window.start_time}, {window.end_time}]
    Try using layman's English
    """

    return context, prompt
