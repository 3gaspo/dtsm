## Script to rebuild dataset from scratch if cfg.rebuild=True
## Sets a new random data as example

import hydra
import logging
import torch

from src.data.electricity import create_dataset, set_random_data
from src.data.dataloader import get_data_splits
from src.data.process import fetch_example_data
from src.training.utils import normalize
from src.visu.plots import plot_example, plot_stats

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    print("\n\n")
    logger.info("=====Running data script=====")

    #configs
    path = cfg.data.path
    rebuild = cfg.data.rebuild
    indiv_split = cfg.data.indiv_split
    date_split = cfg.data.date_split
    lag = cfg.model.lag
    horizon = cfg.model.horizon
    logger.info("Fetched configs")

    #data
    if rebuild:
        logger.info("Rebuilding dataset")
        create_dataset(path, origin = cfg.data.origin)
    set_random_data(path, lag, horizon)
    values = torch.load(path +"values.pt")
    datetimes = torch.load(path + "datetimes.pt")
    logger.info("Fetched data")
    logger.info(f"Values shape : {values.shape}")
    logger.info(f"Datetimes size : {len(datetimes)}")

    #splits
    data_splits = get_data_splits(values, datetimes, indiv_split, date_split)
    assert(len(data_splits)==4)

    data_splits = {"train":data_splits["split_1"],"valid": data_splits["split_2"],"test":data_splits["split_4"]}
    plot_stats(data_splits, "mean", path, "mean_distributions.pdf")
    plot_stats(data_splits, "max", path, "max_distributions.pdf")    
    
    #example
    x, c, y = fetch_example_data(path, "rand")
    x_normalized, mean, std =  normalize(x, return_stats=True)
    y_normalized = (y - mean)/std
    plot_example(x[0], y[0], path, f"example.pdf", "Example")        
    plot_example(x_normalized[0], y_normalized[0], path, f"normal_example.pdf", "Normlized Example")        
    logger.info('Saved plots')

    logger.info('End of script\n')

if __name__ == "__main__":
    run()


