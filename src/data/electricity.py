import datetime
import torch
from src.data.process import set_random_data
import pandas as pd

def fetch_txt_data(path, years=None, hourly=True):
    """returns electricity.txt dataset as consumptions tensor and datetimes list"""
    consumptions = [] #conso des individus
    datetimes = [] #informations sur la date

    with open(path + "electricity.txt", 'r') as file:
        next(file)
        for line in file:
            parts = line.strip().split(';')
            dt = datetime.datetime.strptime(parts[0].strip('"'), "%Y-%m-%d %H:%M:%S")
            if years is None or dt.year in years: #filtre sur l'année
                datetimes.append(dt)
                consumptions.append([float(parts[k].replace(",", ".")) for k in range(1,len(parts))])

    consumptions = torch.tensor(consumptions, dtype=torch.float32).transpose(1,0).unsqueeze(1) #(N_individuals, 1, N_dates)

    if hourly: #pas de temps 15 min => pas horaire
        N_individuals, dim, N_dates = consumptions.shape
        consumptions = consumptions.view(N_individuals, dim, N_dates//4, 4).sum(dim=3)
        datetimes = datetimes[::4]

    return consumptions, datetimes

def fetch_csv_data(path):
    """returns electricity.csv dataset as consumptions tensor and datetimes list"""
    df = pd.read_csv(path + "electricity.csv")
    datetimes = [datetime.datetime.strptime(date.strip('"'), "%Y-%m-%d %H:%M:%S") for date in df.date]
    consumptions = df.drop(columns=["date","OT"]).values
    consumptions = torch.tensor(consumptions, dtype=torch.float32).transpose(1,0).unsqueeze(1) #(N_individuals, 1, N_dates)
    return consumptions, datetimes


def fetch_data(path, origin="csv", years=None, hourly=None):
    """fetches correct dataset"""
    if origin == "txt":
        consumptions, datetimes = fetch_txt_data(path, years, hourly)
    elif origin == "csv":
        consumptions, datetimes = fetch_csv_data(path)
    else:
        raise ValueError("Origin of dataset not recognized")
    return consumptions, datetimes


def create_dataset(path="dataset/", origin="csv", years=None, hourly=None):
    values, datetimes = fetch_data(path, origin, years, hourly)
    torch.save(values, path + "values.pt")
    torch.save(datetimes, path + "datetimes.pt")