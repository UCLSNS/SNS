create_page.py: Creating GUI elements using the Tkinter library in Python

keyword_detect.py
Detect keywords related to stock and weather predictions in a given sentence. The keywords are inputs of prediction functions.

Parameters:
- sentence (str): The input sentence to analyze.

Defaults: 
- If the sentence doesn't contain any numbers, the default prediction time is set to 1. 
- For daily weather 
detection: if the predicted feature is weather and whether it's maximum or minimum is not specified, both maximum and 
minimum values are returned.

Returns:
For stock predictions:
- target_columns (list): Detected target columns for stock predictions.
- prediction_time (int): Detected prediction time.
- prediction_type (str): Detected prediction type (daily or hourly).

For weather predictions:
- prediction_city (list): Detected city names for which weather predictions are available.
- prediction_city_not_available (list): Detected city names for which weather predictions are not available.
- prediction_time (int): Detected prediction time.
- prediction_type (str): Detected prediction type (daily or hourly).
- target_columns (list): Detected target columns for weather predictions.
"""

