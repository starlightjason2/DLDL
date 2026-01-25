import os
local_rank = os.environ.get("PMI_LOCAL_RANK")
os.environ["CUDA_VISIBLE_DEVICES"] = local_rank
from DLDL import ipDataset, ipCNN, loss, split
from datetime import timedelta
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=10)
    )
    torch.cuda.set_device(0)  # Assign a GPU to each process
    # Each process sees only one GPU, so use ID 0


def setup_file(rank, world_size, rendezvous_file):
    dist.init_process_group(
        backend='nccl',
        init_method=f'file://{rendezvous_file}',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=10)
    )
    torch.cuda.set_device(rank)  # Assign a GPU to each process


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, data_path, labels_path, prog_dir, max_length, jobID,\
        lr = 0.01, num_epochs = 100, log_interval = 20, classification = True):
    setup(rank, world_size)

    # Make sure each process has a different seed if you are using any randomness
    torch.manual_seed(42 + rank)

    dataset = ipDataset(data_path, labels_path, classification)
    train, dev, _ = split(dataset)

    # Use DistributedSampler
    train_sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train, batch_size=128, sampler=train_sampler, pin_memory=True)
    dev_loader = DataLoader(dev, batch_size=128, shuffle=False, pin_memory=True)

    model = ipCNN(max_length = max_length, classification = classification).cuda()
    model = DDP(model, device_ids=[0])

    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce_loss = torch.nn.BCEWithLogitsLoss()  # For binary classification
    if not classification:
        mse_loss = torch.nn.MSELoss()  # For regression

    logs = []
    if rank == 0:
        writer = SummaryWriter(prog_dir+jobID)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            if not classification:
                classification_targets, time_targets = target[:, 0], target[:, 1]

            optimizer.zero_grad()
            output = model(data)
            if not classification:
                classification_output, time_output = output[:, 0], output[:, 1]
                loss_classification = bce_loss(classification_output, classification_targets)
                loss_time = mse_loss(time_output, time_targets)

                L = loss(output, target)
            else:
                L = bce_loss(output, target)

            L.backward()
            optimizer.step()
            total_train_loss += loss.item()

            if batch_idx % log_interval == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, "+\
                      f"[{batch_idx * len(data)}/{len(train_loader.dataset)}] "+\
                      f"Loss {L.item()}")
                if rank == 0:
                    writer.add_scalar('Training Loss', L.item(),\
                                      epoch*len(train_loader.dataset) + batch_idx)
                    if not classification:
                        writer.add_scalar('Training Classification Loss',\
                            loss_classification.item(),\
                            epoch * len(train_loader.dataset) + batch_idx)
                        writer.add_scalar('Training Time Loss', loss_time.item(),\
                            epoch * len(train_loader.dataset) + batch_idx)


        # Save model - ensure only one orocess does this or save in each process with unique filenames
        if rank == 0:
            # Validation loop
            model.eval()
            total_val_loss = 0
            if not classification:
                all_classification_targets, all_classification_predictions = [], []
                all_time_targets, all_time_predictions = [], []

            with torch.no_grad():
                for data, targets in dev_loader:
                    data, targets = data.cuda(), targets.cuda()
                    output = model(data)
                    if not classification:
                        classification_targets, time_targets = targets[:, 0], targets[:, 1]
                        classification_output, time_output = output[:, 0], output[:, 1]
                        val_loss_classification = bce_loss(classification_output, classification_targets)
                        val_loss_time = mse_loss(time_output, time_targets)
                        all_time_targets.extend(time_targets.cpu().numpy())
                        all_time_predictions.extend(time_output.cpu().numpy())
                    else:
                        classification_output = output
                        classification_targets = targets
                    
                    classification_predictions = torch.sigmoid(classification_output) > 0.5
                    all_classification_targets.extend(classification_targets.cpu().numpy())
                    all_classification_predictions.extend(classification_predictions.cpu().numpy())

                    val_total_loss = loss(output, targets)
                    total_val_loss += val_total_loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(dev_loader)

            # Log to DataFrame
            logs.append({
                "epoch": epoch,
                "training_loss": avg_train_loss,
                "validation_loss": avg_val_loss,
                'Validation Accuracy': accuracy_score(all_classification_targets,\
                    all_classification_predictions),
                'Validation Precision': precision_score(all_classification_targets,\
                    all_classification_predictions),
                'Validation Recall': recall_score(all_classification_targets,\
                    all_classification_predictions),
                'Validation F1 Score': f1_score(all_classification_targets,\
                    all_classification_predictions)
            })

            writer.add_scalar('Validation Loss', avg_val_loss, epoch)
            writer.add_scalar('Validation Accuracy',\
                    accuracy_score(all_classification_targets,\
                    all_classification_predictions), epoch)
            writer.add_scalar('Validation Precision',\
                    precision_score(all_classification_targets,\
                    all_classification_predictions), epoch)
            writer.add_scalar('Validation Recall',\
                    recall_score(all_classification_targets,\
                    all_classification_predictions), epoch)
            writer.add_scalar('Validation F1 Score',\
                    f1_score(all_classification_targets,\
                    all_classification_predictions), epoch)
            if not classification:
                writer.add_scalar('Validation Time MSE',\
                    mse_loss(torch.tensor(all_time_predictions),\
                    torch.tensor(all_time_targets)).item(), epoch)

        if epoch % 5 == 0 and rank == 0:
            torch.save(model.state_dict(), f"{prog_dir}{jobID}_params_epoch{epoch}.pt")

    if rank == 0:
        writer.close()
        # Convert logs to DataFrame and save to CSV
        df_logs = pd.DataFrame(logs)
        df_logs.to_csv(prog_dir+jobID+"_training_log.csv", index=False)

    cleanup()


if __name__ == "__main__":
    data_path = '/eagle/fusiondl_aesp/jrodriguez/processed_data/processed_dataset_meanvar-whole.pt'
    labels_path = '/eagle/fusiondl_aesp/jrodriguez/processed_data/processed_labels_scaled_labels.pt'
    max_length = np.loadtxt('/eagle/fusiondl_aesp/jrodriguez/processed_data/max_length.txt')\
                    .astype(int)
    prog_dir = '/eagle/fusiondl_aesp/jrodriguez/train_progress/'

    rank = int(os.getenv('PMI_RANK', '0'))
    world_size = int(os.getenv('PMI_SIZE', '1'))  # Default to 1 if not set
    print("GPUs Available:", torch.cuda.device_count())
    print("Rank:", rank)
    train(rank, world_size, data_path, labels_path, prog_dir, max_length,\
            jobID = "DLDL_test_lr0005", lr = 0.0005, num_epochs = 250, log_interval = 50)