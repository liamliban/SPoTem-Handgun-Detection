import torch
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(train_loader, val_loader, combined_model, criterion, optimizer, device, num_epochs):
    train_losses = []  # To store training losses for each epoch
    val_losses = []    # To store validation losses for each epoch

    for epoch in range(num_epochs):
        combined_model.train()
        total_train_loss = 0

        for batch in train_loader:
            data_name, gun_data, pose_data, motion_data, target_labels = batch

            gun_data = gun_data.to(device)
            pose_data = pose_data.to(device)
            motion_data = motion_data.to(device)
            target_labels = target_labels.to(device)

            optimizer.zero_grad()

            combined_output = combined_model(gun_data, pose_data, motion_data)

            loss = criterion(combined_output, target_labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)
        train_losses.append(average_train_loss)

        # Validation loop
        combined_model.eval()  # Set the model to evaluation mode
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                data_name, gun_data, pose_data, motion_data, target_labels = batch

                gun_data = gun_data.to(device)
                pose_data = pose_data.to(device)
                motion_data = motion_data.to(device)
                target_labels = target_labels.to(device)

                combined_output = combined_model(gun_data, pose_data, motion_data)

                val_loss = criterion(combined_output, target_labels)
                total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / len(val_loader)
        val_losses.append(average_val_loss)

        combined_model.train()  # Set the model back to training mode

    return train_losses, val_losses