## Script to test prediction of patch_tst on examples

import hydra
import logging
import torch
from src.data.process import fetch_example_data
from src.training.utils import normalize
from src.models.patchtst.patch_tst import PatchTST
from src.training.pipeline import eval_model
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch.nn as nn


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    print("\n\n")
    logger.info("=====Running main script=====")
    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    #configs
    model_name = cfg.model.name
    output_dir = cfg.misc.outputdir
    if output_dir is None:
        output_dir = hydra_dir + "/"
    save_name = model_name
    model_path = cfg.model.modelpath
    logger.info("Fetched configs")

    #model
    save_name = model_name + "_revin"
    model = torch.load(model_path)
    logger.info(f'Loaded model: {save_name}')
    
    save_dir = output_dir + save_name + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir + "examples/"):
        os.makedirs(save_dir + "examples/")

    #data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dico = fetch_example_data("datasets/examples/", ["motif", "big_motif", "anomalie"])
    X, C, Y = [], [], []
    for data_name, data_tuple in dico.items():
        x, c, y = data_tuple[0].unsqueeze(0), data_tuple[1].unsqueeze(0), data_tuple[2].unsqueeze(0)
        X.append(x)
        C.append(c)
        Y.append(y)
    X, C, Y = torch.cat(X, dim=0).to(device), torch.cat(C, dim=0).to(device), torch.cat(Y, dim=0).to(device)
    
    #pred
    model.to(device)
    X_normalized, mean, std =  normalize(X, return_stats=True)
    print("X shape : ",X.shape)
    print("C shape : ",C.shape)
    pred = model(X,C)
    pred_normalized = (pred - mean)/std
    print("pred shape : ", pred.shape)
    print("y shape : ", Y.shape)

    #eval
    loss_all = nn.MSELoss(reduction="none")
    loss_reduc = nn.MSELoss()
    mse_all = loss_all(pred, Y)
    mse_reduc = loss_reduc(pred, Y)
    mse_all_reduc = mse_all.mean()
    print("MSE all : ", mse_all.shape)
    print("MSE reduc: ", mse_reduc.shape, mse_reduc)
    print("MSE all reduc : ", mse_all_reduc.shape, mse_all_reduc)

    logger.info('End of script\n')

if __name__ == "__main__":
    run()


