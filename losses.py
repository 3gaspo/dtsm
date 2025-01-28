## Script to rebuild dataset from scratch if cfg.rebuild=True
## Sets a new random data as example

import hydra
import logging
import torch
import os

# from src.data.electricity import create_dataset, set_random_data
# from src.data.dataloader import get_data_splits
# from src.data.process import fetch_example_data
# from src.training.utils import normalize
# from src.visu.plots import plot_stats
from src.visu.plots import plot_multi_losses

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    print("\n\n")
    logger.info("=====Running data script=====")

    #configs
    output_dir = cfg.misc.outputdir
    n_evals = cfg.training.n_evals
    logger.info("Fetched configs")
 
    expe_names = [name for name in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, name))]

    losses_dict = {}
    for expe_name in expe_names:
        if expe_name in ["linear", "MLP", "patch_tst", "patch_tst_revin"]:
            train_losses, valid_losses = torch.load(output_dir + expe_name + "/" + "train_losses.pt"), torch.load(output_dir + expe_name + "/" + "valid_losses.pt")
            #losses_dict[expe_name] = (train_losses, valid_losses)
            losses_dict[expe_name] = (valid_losses, None)
    plot_multi_losses(losses_dict, output_dir, "losses.pdf", f"Losses of {expe_name}", n_evals=n_evals)

    logger.info('End of script\n')

if __name__ == "__main__":
    run()


