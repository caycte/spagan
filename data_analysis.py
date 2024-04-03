import os
import pandas as pd
import matplotlib.pyplot as plt


def get_exp_id(path_folder):
    # List all files in the directory
    files = os.listdir(path_folder)

    # Filter out files that match the pattern 'n.png'
    numbers = []
    for file in files:
        if file.endswith(".png") and file.replace('.png', '').isdigit():
            numbers.append(int(file.replace('.png', '')))

    # Find the maximum number if there are any numbers
    if numbers:
        return max(numbers)
    else:
        return 0

def plot_training_loss_and_accuracy(acc_train_list, acc_val_list, loss_train_list, loss_val_list, model, dataset):
    epochs = range(1, len(acc_train_list) + 1)

    # paths for saving results
    plot_dir = f"results/plots/{dataset}/{model}/"
    dataframe_dir = f"results/dataframes/{dataset}/{model}/"
    


    # Create directory if it doesn't exist
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(dataframe_dir):
        os.makedirs(dataframe_dir)

    results = pd.DataFrame({'acc_train':acc_train_list,
                           'acc_val':acc_val_list,
                           'loss_train':loss_train_list,
                           'loss_val':loss_val_list})
    
    exp_id = get_exp_id(plot_dir)
    plot_path = f"{plot_dir}{exp_id}.png"
    dataframe_path = f"{dataframe_dir}{exp_id}.csv"


    
    # saving the dataframe
    results.to_csv(dataframe_path, index=False)

    # Plotting code
    plt.figure(figsize=(12, 5))

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_train_list, 'b-', label='Training Loss')
    plt.plot(epochs, loss_val_list, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_train_list, 'b-', label='Training Accuracy')
    plt.plot(epochs, acc_val_list, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved as {plot_path}")
    print(f"Dataframe saved as {dataframe_path}")