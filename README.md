# AnalyzeWindTunnelData
Python code to extract and process van Breugel lab wind tunnel behavior data.


Uses basic tools from pandas and numpy to read a csv of data into a DataFrame, then computes some variables of interest from the data and adds these to the DataFrame as new columns. Additionally parses the DataFrame into separate trajectories of equal length.

The data describe flight trajectories in 3D during a fictive olfactory navigation task. The functions in utils.py assist with handling circular (angular) variables.