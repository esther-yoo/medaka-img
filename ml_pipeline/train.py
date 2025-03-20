import wandb
import torch
from tqdm import tqdm
from torchvision.transforms import v2 as transforms
from utils import save_checkpoint
from datetime import datetime
from data import MinMaxScaling

def train(model, train_loader, val_loader, val_dataset, criterion, optimizer, run_name, config, device):
    # Define a random rotation transform
    random_trans = transforms.Compose([
        transforms.RandomRotation(config['train_img_rotation'])
    ])

    # Define a dict that will store the epoch as the key and the val_loss as the value
    val_loss_dict = {}
    
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Create wandb table
    columns = ["epoch", "id", "image", "reconstructed"]
    test_table = wandb.Table(columns=columns)

    # Run training and track with wandb
    example_ct = 0  # number of examples seen
    batch_ct = 0

    for epoch in tqdm(range(config['epochs'])):
        model.train()
        for image, image_name, image_mask in train_loader:
        # image_mask = torch.zeros(224, 224)
        # for image_raw, image_name, image in train_loader:
            ### TODO: Add random rotation to both the image and mask
            image = random_trans(image)
            if config['model'] in ["vanilla-ae-sigmoid", "vanilla-ae-relu"]:
                image = torch.reshape(image,(-1, 224*224))
            train_loss = train_batch(image, image_mask, model, optimizer, criterion, device, config)
            example_ct +=  len(image) # the batch size
            batch_ct += 1

        train_log(train_loss, example_ct, epoch)

        model.eval()
        model = model.to(device)
        with torch.no_grad():
            val_loss = 0.0
            for image, image_name, image_mask in val_loader:
            # for image_raw, image_name, image in val_loader:
                if config['model'] in ["vanilla-ae-sigmoid", "vanilla-ae-relu"]:
                    image = torch.reshape(image,(-1, 224*224))
                image = image.to(device)
                output = model(image)
                if (config['model'] in ["convnet-vae"]): 
                    output, mean, logvar = model(image)
                    loss = criterion(output, image, mean, logvar)
                else:
                    if config['loss_with_mask'] == True:
                        output_mask = (MinMaxScaling()(output)[:,:,0] < 0.19).float()
                        # image_mask = (MinMaxScaling()(image)[:,:,0] > 0.15).float()
                        loss = ((1 - config['mask_coef'])*criterion(output, image)) + (config['mask_coef']*criterion(output_mask, image_mask))
                    elif config['loss_with_mask'] == False:
                        loss = criterion(output, image)
                val_loss += loss.item()

            val_loss /= len(val_loader)
            wandb.log({"val_loss": val_loss}, step=epoch)
            print(f"Validation loss: {val_loss:.5f}")
            
        # Every 20th epoch (or last epoch), save model checkpoint
        if ((epoch % config['ckpt_freq'] == 0) or (epoch == config['epochs']-1)):
            val_loss_dict[epoch] = val_loss
            date_time = datetime.now().strftime("%Y-%m-%d-%H%M")
            ckpt_name = f"ckpt-model-{config['model']}-run-{run_name}-epoch-{epoch}-time-{date_time}" 

            save_checkpoint(model, optimizer, epoch, filename=f"{config['ckpt_dir']}/checkpoints/{config['project_name']}/{run_name}/{ckpt_name}.pt")

        # Every 10th epoch (or last epoch), output image reconstruction
        if (epoch % config['reconstruction_table_freq'] == 0) or (epoch == config['epochs']-1):
            print("Logging table at epoch", epoch)
            show_img, id, show_mask = val_dataset[0]
            # raw_img, id, show_img = val_dataset[0]
            if config['model'] in ["vanilla-ae-sigmoid", "vanilla-ae-relu"]:
                recon_img = model(torch.reshape(show_img, (-1, 224*224)).to(device)).reshape(224, 224).cpu().detach().to(torch.float32)
            elif config['model'] in ["convnet-vae"]:
                show_img = show_img.unsqueeze(0)
                mean, logvar = model.get_latent(show_img.to(device))
                recon_img = model.decode(mean, logvar).cpu().detach().to(torch.float32)
                # recon_img, _, _ = model(show_img.to(device))
                # recon_img = recon_img.cpu().detach().to(torch.float32)
            else:
                show_img = show_img.unsqueeze(0)
                recon_img = model(show_img.to(device)).cpu().detach().to(torch.float32)
            test_table.add_data(epoch, id, wandb.Image(show_img), wandb.Image(recon_img))
            test_table = wandb.Table(columns=columns, data=test_table.data)
            wandb.log({"Image reconstruction performance": test_table})

    # Only log model checkpoint as artifact if the val_loss is the lowest in val_loss_dict
    # lowest_epoch, lowest_val_loss = min(val_loss_dict.items(), key=lambda item: item[1])

    # ckpt_name = f"ckpt-model-{config['model']}-run-{run_name}-epoch-{lowest_epoch}"

    # wandb.log_artifact(
    #     artifact_or_path=f"{config['ckpt_dir']}/checkpoints/{config['project_name']}/{run_name}/{ckpt_name}*.pt", 
    #     type="model-checkpoint",
    #     aliases=[f"model={config['model']}", f"architecture={config['architecture']}", f"epoch={lowest_epoch}", f"run={run_name}"])

def train_batch(image, image_mask, model, optimizer, criterion, device, config):
    image = image.to(device)
    model = model.to(device)
    
    # Forward pass
    # if the model is a vae, calculate custom KL divergence loss
    if (config['model'] in ["convnet-vae"]): 
        output, mean, logvar = model(image)
        loss = criterion(output, image, mean, logvar)
    else:
        output = model(image)
        if config['loss_with_mask'] == True:
            output_mask = (MinMaxScaling()(output)[:,:,0] < 0.19).float()
            # image_mask = (MinMaxScaling()(image)[:,:,0] > 0.15).float()
            loss = ((1 - config['mask_coef'])*criterion(output, image)) + (config['mask_coef']*criterion(output_mask, image_mask))
        elif config['loss_with_mask'] == False:
            loss = criterion(output, image)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    print("Loss: ", loss.item())
    return loss.item()

def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "train loss": loss}, step=epoch)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.5f}")