import os
import pickle

project_name = "m4443"
scratch_path = os.environ.get("PSCRATCH")
OGWSC_run  = "O1"
data_path = os.path.join(scratch_path, f"{OGWSC_run}_training_data/")
files = [os.path.join(data_path, x) for x in os.listdir(data_path)]
for file in files:
    with open(file, "rb") as f:
        specs = pickle.load(f)
        assert isinstance(specs, list)
    if len(specs) == 0:
        os.remove(file)
        print(f"removed {file}")
    else:
        print(f"file {file}: {len(specs)} specs")