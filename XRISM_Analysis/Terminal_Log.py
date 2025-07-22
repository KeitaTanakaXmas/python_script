import subprocess
import datetime
import sys
import os

def create_directory(dir_name:str='Log'):
    """
    Create a directory if it does not exist.
    """
    # Check if the directory exists
    if not os.path.exists(dir_name):
        # If the directory does not exist, create it
        print(f"Creating directory: {dir_name}")
        os.makedirs(dir_name)

def execute_shell_script(cmd, log_file):
    """
    Log the terminal output to the specified file.

    Args:
        log_file (str): The name of the log file.

    Returns:
        None
    """
    with open(log_file, 'a') as f:
        process = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        process.wait()

def time_now(print=True):
    """
    Get the current time and date, and print them in the specified format.

    Returns:
        tuple: A tuple containing the current time and date (time format: hour_min_sec, date format: year_month_day).
    """
    current_time = datetime.datetime.now().strftime("%H_%M_%S")
    current_date = datetime.datetime.now().strftime("%Y_%m_%d")
    if print==True:
        print("Current Time:", current_time)
        print("Current Date:", current_date)
    return current_time, current_date

