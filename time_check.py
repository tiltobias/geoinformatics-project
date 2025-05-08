import pandas as pd
from datetime import datetime

folder = "KinematicData/"
files = ["T1.csv", "T2.csv", "T3.csv", "T4.csv", "T5.csv", "T6.csv", "T7.csv", "T8.csv"]

# Create an empty DataFrame
df = pd.DataFrame(columns=["file", "is_sorted", "first_time", "last_time", "n_lines", "n_times_div_60", "diff_seconds", "time_matches_n_times"])

def time_analysis(file):
    times = []
    path = folder + file
    print("Processing file:", path)

    with open(path) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line.strip().split(",")
            if i == 0: 
                continue  # Skip header
            if i > 10000:
                break
            times.append(parts[0].split(" ")[1])

    first_time = times[0]
    last_time = times[-1]

    # Calculate time difference
    time_format = "%H:%M:%S"
    t1 = datetime.strptime(first_time, time_format)
    t2 = datetime.strptime(last_time, time_format)

    diff_seconds = (t2 - t1).total_seconds()
    if diff_seconds < 0:
        diff_seconds += 24 * 3600  # Handle day wrap-around
        

    # Create a dictionary with results
    result = {
        "file": file,
        "is_sorted": times == sorted(times),
        "first_time": first_time,
        "last_time": last_time,
        "n_lines": len(times),
        "n_times_div_60": len(times) / 60,
        "diff_seconds": diff_seconds,
        "time_matches_n_times": diff_seconds == float(len(times) - 1),
    }

    return result

if __name__ == "__main__":
    # Collect results
    results = []
    for file in files:
        result = time_analysis(file)
        results.append(result)

    # Create the DataFrame from results
    df = pd.DataFrame(results)

    print(df)
