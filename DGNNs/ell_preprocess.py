import pandas as pd
import os
import tarfile

# Define paths to the original files and new files
data_path = "./data/elliptic_bitcoin_dataset"
out_path = "./data/elliptic_temporal"
features_file = os.path.join(data_path, "elliptic_txs_features.csv")
classes_file = os.path.join(data_path, "elliptic_txs_classes.csv")
edgelist_file = os.path.join(data_path, "elliptic_txs_edgelist.csv")

# Output files
features_out_file = os.path.join(out_path, "elliptic_bitcoin_dataset_cont/elliptic_txs_features.csv")
classes_out_file = os.path.join(out_path, "elliptic_bitcoin_dataset_cont/elliptic_txs_classes.csv")
edgelist_out_file = os.path.join(out_path, "elliptic_bitcoin_dataset_cont/elliptic_txs_edgelist.csv")
nodetime_file = os.path.join(out_path, "elliptic_bitcoin_dataset_cont/elliptic_txs_nodetime.csv")
tar_file = os.path.join(out_path, "elliptic_bitcoin_dataset_cont.tar.gz")

# Create output directory if it doesn't exist
os.makedirs(os.path.join(out_path, "elliptic_bitcoin_dataset_cont"), exist_ok=True)

# Load data directly without tar archive
print("Loading data files...")

# Step 1: Process `elliptic_txs_features.csv`
print("Processing features file...")
features_df = pd.read_csv(features_file, header=None)

# Map original node IDs to contiguous IDs
orig_to_cont_map = {orig_id: cont_id for cont_id, orig_id in enumerate(features_df[0])}

# Modify `elliptic_txs_features.csv`
features_df[0] = features_df[0].map(orig_to_cont_map).astype(float)
features_df[1] = features_df[1].astype(float)
features_df.to_csv(features_out_file, index=False, header=False)
print(f"Modified features file saved to {features_out_file}")

# Step 2: Process `elliptic_txs_classes.csv`
print("Processing classes file...")
classes_df = pd.read_csv(classes_file)
classes_df["txId"] = classes_df["txId"].map(orig_to_cont_map).astype(float)
classes_df["class"] = classes_df["class"].replace({"unknown": -1.0, "1": 1.0, "2": 0.0})  # note: unknown data are not used
classes_df.to_csv(classes_out_file, index=False)
print(f"Modified classes file saved to {classes_out_file}")

# Step 3: Create `elliptic_txs_nodetime.csv`
print("Creating node time file...")
nodetime_df = features_df[[0, 1]].copy()
nodetime_df.columns = ["txId", "timestep"]
nodetime_df["timestep"] -= 1  # Convert to zero-based indexing
nodetime_df.to_csv(nodetime_file, index=False)
print(f"Node time file saved to {nodetime_file}")

# Step 4: Process `elliptic_txs_edgelist.csv`
print("Processing edge list file...")
edgelist_df = pd.read_csv(edgelist_file)
edgelist_df["txId1"] = edgelist_df["txId1"].map(orig_to_cont_map)
edgelist_df["txId2"] = edgelist_df["txId2"].map(orig_to_cont_map)
edgelist_df = edgelist_df.dropna()  # Drop rows with unmapped IDs

# Add time stamp information
nodetime_map = nodetime_df.set_index("txId")["timestep"].to_dict()
edgelist_df["timestep"] = edgelist_df["txId1"].map(nodetime_map)
edgelist_df.to_csv(edgelist_out_file, index=False)
print(f"Timed edge list file saved to {edgelist_out_file}")

# Create tar.gz archive
print("Creating tar.gz archive...")
with tarfile.open(tar_file, "w:gz") as tar:
    tar.add(os.path.join(out_path, "elliptic_bitcoin_dataset_cont"), arcname="elliptic_bitcoin_dataset_cont")
print(f"Compressed dataset saved to {tar_file}")

print("All preprocessing steps completed!")