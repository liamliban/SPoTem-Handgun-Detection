import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import os
from datetime import datetime
import torch
import random
import numpy as np

# Set a random seed for reproducibility
torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)
os.environ['PYTHONHASHSEED'] = str(12)
torch.cuda.manual_seed_all(12)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

torch.backends.cudnn.deterministic=True

def train_model(user_input, train_loader, val_loader, combined_model, criterion, optimizer, device, num_epochs, excel_filename):
    train_losses = []  # To store training losses for each epoch
    val_losses = []    # To store validation losses for each epoch

    # Get the current date and time
    current_datetime = datetime.now().strftime('%Y-%m-d %H:%M:%S')
    # Get the run number based on the existing Excel file
    run_number = 1
    if os.path.exists(excel_filename):
        existing_df = pd.read_excel(excel_filename)
        run_number = len(existing_df) + 1
    
    willSave = input("Do you want to save the model? [y/n]: ").strip().lower() == 'y'

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
            data_name, gun_data, pose_data, target_labels = batch

            gun_data = gun_data.to(device)
            pose_data = pose_data.to(device)
            target_labels = target_labels.to(device)

            optimizer.zero_grad()

            combined_output = combined_model(gun_data, pose_data)  # New line for CombinedWithNoMotion

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
                data_name, gun_data, pose_data, target_labels = batch

                gun_data = gun_data.to(device)
                pose_data = pose_data.to(device)
                target_labels = target_labels.to(device)
                
                combined_output = combined_model(gun_data, pose_data)
                
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

        # Save Model
        if willSave:
            model_checkpoint = {
                'epoch': epoch,
                'model_state_dict': combined_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            model_type = 'GPM' if user_input == '1' else 'GP'
            model_folder = f'logs/models/{model_type}/{run_number}/'
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            model_path = f'{model_folder}model_epoch_{epoch}.pt'
            torch.save(model_checkpoint, model_path)

    

    # Save the results to Excel with run number and date
    write_results_to_excel(excel_filename, run_number, current_datetime, user_input, num_epochs, train_accuracy, train_precision, train_recall, train_f1_score, val_accuracy, val_precision, val_recall, val_f1_score, train_losses, val_losses)

    return train_losses, val_losses

# Function to write the results to Excel
def write_results_to_excel(filename, run_number, current_datetime, user_input, num_epochs, train_accuracy, train_precision, train_recall, train_f1, val_accuracy, val_precision, val_recall, val_f1, train_losses, val_losses):
    if not os.path.exists(filename):
        df = pd.DataFrame(columns=["Run #", "Date", "Model", "Epochs", "Train Accuracy", "Train Loss", "Train Precision", "Train Recall", "Train F1", "Val Accuracy", "Val Loss", "Val Precision", "Val Recall", "Val F1"])
    else:
        df = pd.read_excel(filename)
    model = ''
    if user_input == '1':
        model = 'GPM'
    elif user_input == '2':
        model = 'GP'

    # Format the accuracy with 2 decimal points and others with at most 4 decimal points
    train_accuracy_str = "{:.2f}%".format(train_accuracy)
    train_precision_str = "{:.4f}".format(train_precision)
    train_recall_str = "{:.4f}".format(train_recall)
    train_f1_str = "{:.4f}".format(train_f1)
    val_accuracy_str = "{:.2f}%".format(val_accuracy)
    val_precision_str = "{:.4f}".format(val_precision)
    val_recall_str = "{:.4f}".format(val_recall)
    val_f1_str = "{:.4f}".format(val_f1)
    train_loss_str = "{:.4f}".format(train_losses[-1]) 
    val_loss_str = "{:.4f}".format(val_losses[-1])

    new_row = pd.Series({"Run #": run_number,
                         "Date": current_datetime,
                         "Model": model,
                         "Epochs": num_epochs,
                         "Train Accuracy": train_accuracy_str,
                         "Train Loss": train_loss_str,
                         "Train Precision": train_precision_str,
                         "Train Recall": train_recall_str,
                         "Train F1": train_f1_str,
                         "Val Accuracy": val_accuracy_str,
                         "Val Loss": val_loss_str,
                         "Val Precision": val_precision_str,
                         "Val Recall": val_recall_str,
                         "Val F1": val_f1_str})

    df = df.append(new_row, ignore_index=True)
    
    # Create logs folder if missing
    if not os.path.exists('logs/'):
        os.makedirs("logs/")

    df.to_excel(filename, index=False)

    print ("Results saved to: ", filename)

