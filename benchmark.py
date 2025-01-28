import hydra
import logging
import torch
from src.data.dataloader import get_data_loaders
from src.data.process import fetch_example_data
from src.training.pipeline import train_model, eval_model
from src.training.utils import save_results, normalize
from src.models.network import MLP
from src.models.patchtst.patch_tst import PatchTST
from src.models.naive import persistence, repeat, lookback, linear
from src.visu.plots import plot_losses, plot_pred, plot_errors, plot_horizon_errors
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    print("\n\n")
    logger.info("=====Running main script=====")
    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    #configs
    path = cfg.data.path
    indiv_split, date_split = cfg.data.indiv_split, cfg.data.date_split
    lag, horizon = cfg.model.lag, cfg.model.horizon
    model_name = cfg.model.name
    steps, batch_size = cfg.training.steps, cfg.training.bs
    lr, schedule = cfg.training.lr, cfg.training.schedule
    loss_name = cfg.training.loss
    do_print = cfg.training.print
    valid_steps, test_steps = cfg.training.valid_steps, cfg.training.test_steps
    n_prints, n_evals = cfg.training.n_prints, cfg.training.n_evals
    seed = cfg.misc.seed
    output_dir = cfg.misc.outputdir
    if output_dir is None:
        output_dir = hydra_dir + "/"
    save_name = model_name + loss_name
    if not os.path.exists(output_dir + "config.yaml"):
        OmegaConf.save(cfg, output_dir + "config.yaml")
    logger.info("Fetched configs")

    #data
    values, datetimes = torch.load(path +"values.pt"), torch.load(path + "datetimes.pt")
    logger.info("Fetched data")
    
    dataloaders = get_data_loaders(values, datetimes, steps, indiv_split, date_split, None, lag, horizon, batch_size, seed, valid_steps, test_steps)
    train_loader, valid_loader, test_loader = dataloaders["train"], dataloaders["valid"], dataloaders["test"]
    logger.info("Built dataloaders")
    
    #sizes
    X, c, y = next(iter(train_loader))
    dim, context = X.shape[1], c.shape[1]

    #model
    normal = True
    if model_name == "persistence":
        model = persistence(horizon)
    elif model_name == "repeat":
        model = repeat(horizon)
    elif model_name == "lookback":
        model = lookback(cfg.model.lookback_idx, horizon)
    elif model_name == "MLP":
        revin = cfg.model.revin
        if revin==1:
            save_name = save_name + "_revin"
            normal = False
        model = MLP(lag, cfg.model.hidden, horizon, dim, context, revin)
    elif model_name == "linear":
        model = linear(lag, horizon, dim=X.shape[1])
    elif model_name == "patch_tst":
        revin = cfg.model.revin
        if revin==1:
            save_name = save_name + "_revin"
            normal = False
        model = PatchTST(lag, horizon, revin)
    else:
        raise ValueError(f"Model name not recognized : {model_name}")
    logger.info(f'Loaded model: {save_name}')
    
    save_dir = output_dir + save_name + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir + "examples/"):
        os.makedirs(save_dir + "examples/")

    #training
    normal = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name in ["MLP", "linear","patch_tst"]:
        logger.info("Starting training...")
        logger.info(f"batch_size={batch_size}, learning_rate={lr}, steps={steps}")
        model, train_losses, valid_losses = train_model(model, train_loader, valid_loader, device, lr, do_print, normal, schedule, n_evals, n_prints, loss_name)
        torch.save(model.state_dict(), save_dir + "model.pt")
        plot_losses(train_losses, valid_losses, save_dir, "train_losses.pdf", f"Losses of {save_name}")
        plot_losses(valid_losses, None, save_dir, "losses.pdf", f"Losses of {save_name}")
        torch.save(train_losses, save_dir + "train_losses.pt")
        torch.save(valid_losses, save_dir + "valid_losses.pt")
        logger.info("End of training")
    else:
        logger.info("No training needed")

    #eval
    test_losses, normalized_test_losses = eval_model(model, test_loader, device, normal=normal, return_all=True) #(bs * steps, dim, horizon)
    test_mse, test_nmse = test_losses.mean().item(), normalized_test_losses.mean().item()
    torch.save(test_losses, save_dir + "test_losses.pt")
    torch.save(normalized_test_losses, save_dir + "normalized_losses.pt")
    logger.info(f"Test MSE : {test_mse:.2f}, Test NMSE : {test_nmse:.5f}")
    save_results(test_mse, output_dir, "results.json", save_name, "Test MSE")
    save_results(test_nmse, output_dir, "results.json", save_name, "Test NMSE")

    #errors
    plot_errors(test_losses[:, 0, :].mean(axis=1).cpu().numpy(), save_dir, "test_mse.pdf", f"Test MSE of {save_name} : {test_mse}")
    plot_errors(normalized_test_losses[:, 0, :].mean(axis=1).cpu().numpy(), save_dir, "test_nme.pdf", f"Test NMSE of {save_name} : {test_nmse}")
    plot_horizon_errors(test_losses[:, 0, :].mean(axis=0).cpu().numpy(), save_dir, "horizon_mse.pdf", f"Test MSE of {save_name} : {test_nmse}")
    plot_horizon_errors(normalized_test_losses[:, 0, :].mean(axis=0).cpu().numpy(), save_dir, "horizon_nmse.pdf", f"Test NMSE of {save_name} : {test_nmse}")
    
    #example
    dico = fetch_example_data("datasets/examples/", ["motif", "big_motif", "anomalie"])
    for data_name, data_tuple in dico.items():
        x, c, y = data_tuple[0].unsqueeze(0).to(device), data_tuple[1].unsqueeze(0).to(device), data_tuple[2].unsqueeze(0).to(device)
        x_normalized, mean, std =  normalize(x, return_stats=True)
        if normal:
            pred_normalized = model(x_normalized,c)
            pred = pred_normalized*std + mean
        else:
            pred = model(x,c)
            pred_normalized = (pred - mean)/std
        y_normalized = (y - mean)/std
        plot_pred(x[0,0].cpu().detach().numpy(), y[0,0].cpu().detach().numpy(), pred[0,0].cpu().detach().numpy(), save_dir + "examples/", f"{data_name}_predictions.pdf", f"Example {data_name} prediction for {save_name}")        
        plot_pred(x_normalized[0,0].cpu().detach().numpy(), y_normalized[0,0].cpu().detach().numpy(), pred_normalized[0,0].cpu().detach().numpy(), save_dir + "examples/", f"{data_name}_normal_predictions.pdf", f"Example {data_name} normalized prediction for {save_name}")        
    logger.info('Saved plots')

    logger.info('End of script\n')

if __name__ == "__main__":
    run()


