# Oliver Schneider
# MSE 240 F2024 Assignment 3 data download script
# Nov 24, 2024
# Downloads the full dataset from the Kaggle then moves it to the data/ library
# You may want to delete the downloaded dataset or its folder afterwards
# However, note that this may interact with kagglehub if you use it


import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("jfreyberg/spotify-chart-data")

print("Path to dataset files:", path)
print ("Checking for ./data folder...", end="")
if os.path.exists("./data"):
    print("exists!")
else:
    print ("does not exist, creating ./data folder...", end="")
    os.mkdir("./data")
    print ("done!")
print (f"Moving dataset from {path} to ./data folder...", end="")
os.rename(path+"/charts.csv", "./data/charts.csv")
print("...done!")

