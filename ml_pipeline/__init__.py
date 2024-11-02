import utils, models, data, train
from data import MedakaDataset
from models import AutoEncoderSigmoid, AutoEncoderRelu, AutoEncoderConv, AutoEncoderResNet, AutoEncoderVGGNet
import os
import torch
import wandb
from torchinfo import summary
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import argparse

# Import the necessary packages
import yaml

def main(run, config):
    # Set the environment variables
    os.environ['WANDB_DATA_DIR'] = '/hps/nobackup/birney/users/esther/wandb/artifacts/staging/'
    os.environ['WANDB_ARTIFACT_DIR'] = '/hps/nobackup/birney/users/esther/wandb/artifacts/'
    os.environ['WANDB_CACHE_DIR'] = '/hps/nobackup/birney/users/esther/wandb/.cache/'
    os.environ['WANDB_TIMEOUT'] = '120'

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch.version.cuda)

    utils.set_seeds()

    # Load the data
    dataset = MedakaDataset(data_csv=config['data_csv'], direction_csv=config['direction_csv'], src_dir=config['data_dir'], transform=data.transform(resize_shape=(224, 224)), config=config)
    
    train_len = int(len(dataset) * config['train_split'])
    val_len = len(dataset) - train_len

    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    # Run model pipeline
    with run:
      run_name = run.name
      # make the model, data, and optimization problem
      model, train_loader, val_loader, criterion, optimizer = make(train_dataset, val_dataset, config)

      # and use them to train the model
      torch.cuda.empty_cache()
      train.train(model=model, train_loader=train_loader, val_loader=val_loader, val_dataset=val_dataset, criterion=criterion, optimizer=optimizer, run_name=run_name, config=config, device=device)

    run.finish()

    return model, run_name


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def make(train_dataset, val_dataset, config):
    # Make the data
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # Make the model
    num_input = 224*224 # 50176
    num_hidden_0 = 7168
    num_hidden_1 = 1024
    num_hidden_2 = 512
    num_hidden_3 = 256
    num_hidden_4 = 128

    if config['model'] == 'vanilla-ae-sigmoid':
        model = AutoEncoderSigmoid(num_input, num_hidden_0, num_hidden_1, num_hidden_2, num_hidden_3, num_hidden_4)
        print(summary(model, input_size=(1, 224*224)))
    elif config['model'] == 'vanilla-ae-relu':
        model = AutoEncoderRelu(num_input, num_hidden_0, num_hidden_1, num_hidden_2, num_hidden_3, num_hidden_4)
        print(summary(model, input_size=(1, 224*224)))
    elif config['model'] == 'convnet-ae':
        model = AutoEncoderConv(input_dim=(3, 224, 224), latent_dim=128)
        print(summary(model, input_size=(32, 3, 224, 224)))
    elif config['model'] == 'resnet-ae':
        model = AutoEncoderResNet(input_dim=(3, 224, 224), latent_dim=128)
        print(summary(model, input_size=(32, 3, 224, 224)))
    elif config['model'] == 'vggnet-ae':
        model = AutoEncoderVGGNet(input_dim=(3, 224, 224), latent_dim=128)
        model.encoder_features.requires_grad_(False)
        model.encoder_adaptive_pool.requires_grad_(False)
        print(summary(model, input_size=(32, 3, 224, 224)))
    else:
        return NotImplementedError(f"{config['model']} is not implemented.")

    

    # Make the loss and optimizer
    if config['criterion'] == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        return NotImplementedError(f"{config['criterion']} is not implemented.")
    
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
        model.parameters(), lr=config['learning_rate'])
    else:
        return NotImplementedError(f"{config['optimizer']} is not implemented.")
    
    return model, train_loader, val_loader, criterion, optimizer

if __name__ == "__main__":
    # Load the configuration file
    parser = argparse.ArgumentParser(description="Load configuration for the model.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for the model.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs for the model.")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for the model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    # parser.add_argument("--sweep", type=lambda x: (str(x).lower() == 'true'), required=True, help="Sweep ID for the model.")
    # parser.add_argument("--sweep-config", type=str, required=False, help="Sweep yaml for the model.")
    args = parser.parse_args()

    config = load_config(args.config)
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['learning_rate'] = args.learning_rate
    # sweep = args.sweep
    # sweep_config = load_config(args.sweep_config)

    print(f"Config: {config}")
    # print(f"Sweep: {sweep == True}")

    # tell wandb to get started
    # Initialize wandb
    wandb.login()

    # if args.sweep:
    #     run = wandb.init(config={"parameters": {"manual_key": 1}}, project="vanilla-ae-pytorch-medaka")
    #     sweep_id = wandb.sweep(sweep=config, project="vanilla-ae-pytorch-medaka")
    #     wandb.agent(sweep_id, function=main(run=run, config=config), count=4)
    # else:


    run = wandb.init(project=config['project_name'], config=config)
    # run = wandb.init(entity="ey267-university-of-cambridge",
    #                  project=config['project_name'],
    #                  id="earthy-sweep-3", resume="must")
    main(run=run, config=config)    