Brief description of each python file:

1. create_page.py:
AIM: Creating GUI elements using the Tkinter library in Python

2. keyword_detect.py
AIM: Detect keywords related to stock and weather predictions in a given sentence. The keywords are inputs of prediction functions.

Parameters:
- sentence (str): The input sentence to analyze.

Defaults: 
- If the sentence doesn't contain any numbers, the default prediction time is set to 1.
- For daily weather detection, if the predicted feature is weather and whether its maximum or minimum is not specified, both maximum and minimum values are returned.
The following situation would return an error warning:
-For prediction time & type: the number should not be bigger than one.
-For daily prediction of stock:
To maintain accuracy, the number of target columns should not be bigger than 2.
Separate high & low, open & close, volume into different groups, if the target columns contain different groups, it will return an error.
  




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

