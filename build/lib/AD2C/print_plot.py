# plot_results.py

import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def plot_all_trajectories(base_folder):
    """
    Loads data from all experiment runs and plots them on a single 3D chart.
    """
    sns.set_theme(style="whitegrid")
    
    run_folders = sorted([f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))])
    
    if not run_folders:
        print("No run folders found. Please run the training script first.")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for folder_name in run_folders:
        folder_path = os.path.join(base_folder, folder_name)
        data_path = os.path.join(folder_path, "trajectory_data.pkl")
        
        if not os.path.exists(data_path):
            print(f"Warning: Data not found for {folder_name}. Skipping...")
            continue

        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        label = folder_name.replace("snd_", "SND=")
        
        # Plot the actual diversity trajectory for this run
        ax.plot(data['episodes'], data['actual_snd'], data['returns'], marker='o', linestyle='-', label=f'Actual Trajectory ({label})')

        # Plot the target diversity trajectory for this run
        # This will be a flat line in the SND dimension as the target is constant
        ax.plot(data['episodes'], data['target_snd'], data['returns'], linestyle='--', c='r', label=f'Target Trajectory ({label})')


    ax.set_xlabel('Episode Number')
    ax.set_ylabel('SND (Behavioral Distance)')
    ax.set_zlabel('Mean Return')
    ax.set_title('Combined 3D Trajectory for All Experiments')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.05))
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    # You MUST change this path to the correct location of your 'exploration_runs' folder
    # This folder is created by Hydra inside the 'outputs' directory
    base_folder_to_plot = "outputs/2025-08-20/22-34-57/exploration_runs"
    
    if not os.path.exists(base_folder_to_plot):
        print(f"Error: The folder '{base_folder_to_plot}' does not exist.")
        print("Please check the `outputs` directory for the correct path after running `main.py`.")
    else:
        plot_all_trajectories(base_folder_to_plot)