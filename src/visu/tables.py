import pandas as pd
from tabulate import tabulate
import json

def pd_to_latex(path):
    """returns latex code to create table from dataframe in path"""
    df = pd.read_csv(path)
    latex_output = df.to_latex(index=False, float_format="%.4f")
    print(latex_output)


def print_nice_table(path, div=1e4):
    """print table from dataframe in path"""
    with open(path) as file:
        data = json.load(file)
    df = pd.DataFrame(data)# / div
    df.index = ["Test MSE", "Test NMSE"]
    table = tabulate(df, headers='keys', tablefmt='grid', showindex=True, floatfmt=".4f")
    print(table)