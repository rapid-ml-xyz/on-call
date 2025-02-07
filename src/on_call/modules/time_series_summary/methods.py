import json
import openai
import os
import pandas as pd
from datetime import datetime


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)


def get_time_series_nlp_setup(analysis_json):
    context = """
    Dataset. We took a Kaggle dataset on Bike Sharing Demand. Our goal is to predict the volume of bike rentals on an hourly basis. To do that, we have some data about the season, weather, and day of the week.
    Model. We trained a random forest model using data for the four weeks from January. Let's imagine that in practice, we just started the data collection, and that was all the data available. The performance of the trained model looked acceptable, so we decided to give it a go and deploy.
    Feedback. We assume that we only learn the ground truth (the actual demand) at the end of each week.
    """

    json_str = json.dumps(analysis_json, indent=2, cls=CustomJSONEncoder)
    prompt = f"""
    Please provide a clear, concise natural language summary of the following JSON data:
    {json_str}
    Try using layman's English"""

    return context, prompt


def get_summary(context: str, prompt: str) -> str:
    api_key = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": context},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating summary: {str(e)}"
