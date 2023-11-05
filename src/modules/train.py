import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_model(train_loader, val_loader, combined_model, criterion, optimizer, device, num_epochs):
    train_losses = []  # To store training losses for each epoch
    val_losses = []    # To store validation losses for each epoch

    print("")
    print("Training Started: ")

    for epoch in range(num_epochs):
        start_time = time.time()
        combined_model.train()
        total_train_loss = 0
        correct = 0
        total = 0
        train_predictions = []
        train_targets = []

        for batch in train_loader:
            data_name, gun_data, pose_data, motion_data, target_labels = batch

            gun_data = gun_data.to(device)
            pose_data = pose_data.to(device)
            motion_data = motion_data.to(device)
            target_labels = target_labels.to(device)

            optimizer.zero_grad()

            combined_output = combined_model(gun_data, pose_data, motion_data)

            _, predicted = torch.max(combined_output, 1)  # Get the class with the highest probability
            total += target_labels.size(0)  # Accumulate the total number of examples
            correct += (predicted == target_labels).sum().item()  # Count correct predictions

            loss = criterion(combined_output, target_labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(target_labels.cpu().numpy())

        train_accuracy = 100 * correct / total
        train_precision = precision_score(train_targets, train_predictions, average='weighted')
        train_recall = recall_score(train_targets, train_predictions, average='weighted')
        train_f1_score = f1_score(train_targets, train_predictions, average='weighted')
        train_confusion_matrix = confusion_matrix(train_targets, train_predictions)

        average_train_loss = total_train_loss / len(train_loader)
        train_losses.append(average_train_loss)

        # Validation loop
        combined_model.eval()  # Set the model to evaluation mode
        total_val_loss = 0
        correct = 0
        total = 0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                data_name, gun_data, pose_data, motion_data, target_labels = batch

                gun_data = gun_data.to(device)
                pose_data = pose_data.to(device)
                motion_data = motion_data.to(device)
                target_labels = target_labels.to(device)

                combined_output = combined_model(gun_data, pose_data, motion_data)

                _, predicted = torch.max(combined_output, 1)  # Get the class with the highest probability
                total += target_labels.size(0)  # Accumulate the total number of examples
                correct += (predicted == target_labels).sum().item()  # Count correct predictions

                val_loss = criterion(combined_output, target_labels)
                total_val_loss += val_loss.item()
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(target_labels.cpu().numpy())

        val_accuracy = 100 * correct / total
        val_precision = precision_score(val_targets, val_predictions, average='weighted')
        val_recall = recall_score(val_targets, val_predictions, average='weighted')
        val_f1_score = f1_score(val_targets, val_predictions, average='weighted')
        val_confusion_matrix = confusion_matrix(val_targets, val_predictions)

        end_time = time.time()  # Record the end time for the epoch
        epoch_time = end_time - start_time  # Calculate the time taken for the epoch
        
        average_val_loss = total_val_loss / len(val_loader)
        val_losses.append(average_val_loss)

        combined_model.train()  # Set the model back to training mode

        print("Epoch [{}/{}], Training Accuracy: {:.2f}%, Training Loss: {:.4f}, Validation Accuracy: {:.2f}%, Validation Loss: {:.4f}, Time: {:.2f} seconds".format(epoch + 1, num_epochs, train_accuracy, average_train_loss, val_accuracy, average_val_loss, epoch_time))
        print("Training Precision: {:.4f}, Training Recall: {:.4f}, Training F1 Score: {:.4f}".format(train_precision, train_recall, train_f1_score))
        print("Validation Precision: {:.4f}, Validation Recall: {:.4f}, Validation F1 Score: {:.4f}".format(val_precision, val_recall, val_f1_score))
        print()

    return train_losses, val_losses