import pandas as pd

def main():
    weather = pd.read_csv("TUL.csv", index_col="DATE")

    weather.loc["2020-1-1:2020-1-5",:]
main()