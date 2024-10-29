import wandb
import torch
from tqdm import tqdm
from torchvision.transforms import v2 as transforms
from utils import save_checkpoint
from datetime import datetime

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
        for image, image_name in train_loader:
            image = random_trans(image)
            if config['model'] in ["vanilla-ae", "vanilla-ae-relu"]:
                image = torch.reshape(image,(-1, 224*224))
            train_loss = train_batch(image, model, optimizer, criterion, device)
            example_ct +=  len(image) # the batch size
            batch_ct += 1

        train_log(train_loss, example_ct, epoch)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for image, image_name in val_loader:
                if config['model'] in ["vanilla-ae", "vanilla-ae-relu"]:
                    image = torch.reshape(image,(-1, 224*224))
                image = image.to(device)
                output = model(image)
                loss = criterion(output, image)
                val_loss += loss.item()

            val_loss /= len(val_loader)
            wandb.log({"val_loss": val_loss}, step=epoch)
            print(f"Validation loss: {val_loss:.5f}")
            
        # Every 20th epoch (or last epoch), save model checkpoint
        if ((epoch % config['ckpt_freq'] == 0) or (epoch == config['epochs']-1)):
            if epoch == 0:
                old_val_loss = val_loss
            else:
                date_time = datetime.now().strftime("%Y-%m-%d-%H%M")
                ckpt_name = f"ckpt-model-{config['model']}-run-{run_name}-epoch-{epoch}-time-{date_time}" 

                save_checkpoint(model, optimizer, epoch, filename=f"{config['ckpt_dir']}/checkpoints/vanilla-ae-pytorch-medaka/{run_name}/{ckpt_name}.pt")

                # Only log model checkpoint as artifact if new val_loss is less than the previous val_loss
                if val_loss < old_val_loss:
                    wandb.log_artifact(
                        artifact_or_path=f"{config['ckpt_dir']}/checkpoints/vanilla-ae-pytorch-medaka/{run_name}/{ckpt_name}.pt", 
                        type="model-checkpoint",
                        aliases=[f"model={config['model']}", f"architecture={config['architecture']}", f"epoch={epoch}", f"run={run_name}"])
                
                old_val_loss = val_loss


        # Every 50th epoch (or last epoch), output image reconstruction
        if (epoch % config['reconstruction_table_freq'] == 0) or (epoch == config['epochs']-1):
            print("Logging table at epoch", epoch)
            show_img, id = val_dataset[0]
            show_img = show_img.unsqueeze(0)
            recon_img = model(show_img.to(device)).cpu().detach().to(torch.float32)
            test_table.add_data(epoch, id, wandb.Image(show_img), wandb.Image(recon_img))
            test_table = wandb.Table(columns=columns, data=test_table.data)
            wandb.log({"Image reconstruction performance": test_table})


def train_batch(image, model, optimizer, criterion, device):
    image = image.to(device)
    
    # Forward pass
    output = model(image)
    loss = criterion(output, image)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss

def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "train loss": loss}, step=epoch)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.5f}")