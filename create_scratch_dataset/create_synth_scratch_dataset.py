import os
import pickle
from itertools import groupby


def move_files_to_scratch(sig_func_name, H_files, L_files):
    files_written = 0
    H_buffer = []
    L_buffer = []
    while H_files and L_files:
        if not H_buffer:
            h_file = H_files.pop(0)
            print(h_file)
            with open(h_file, "rb") as f:
                specs = pickle.load(f)
            if not isinstance(specs, list) or not specs:
                continue
            H_buffer += specs
        if not L_buffer:
            l_file = L_files.pop(0)
            print(l_file)
            with open(l_file, "rb") as f:
                specs = pickle.load(f)
            if not isinstance(specs, list) or not specs:
                continue
            L_buffer += specs
        output_file_name = f"synth-out-{sig_func_name}-{len(H_files)}-{len(H_buffer)}-{len(L_files)}-{len(L_buffer)}.pkl"
        with open(os.path.join(output_path, output_file_name), "wb") as f:
            pickle.dump({"H": H_buffer.pop(0), "L": L_buffer.pop(0)}, f)
            files_written += 1
    return files_written


if __name__ == "__main__":
    project_name = "m4443"
    GWOSC_run = "O1"
    scratch_path = os.environ['PSCRATCH']
    data_path = f"/global/cfs/cdirs/{project_name}/synth_data/{GWOSC_run}_synth/"
    output_path = os.path.join(scratch_path, f"{GWOSC_run}_synth_data/")
    files_written = 0
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    H_files = [os.path.join(data_path, x) for x in os.listdir(data_path) if x.startswith("H")]
    L_files = [os.path.join(data_path, x) for x in os.listdir(data_path) if x.startswith("L")]

    # split files lists based on sig_func name (file name suffix)
    H_files_by_sig = groupby(H_files, key=lambda x: x.split("_")[-1].replace(".pkl", ""))
    L_files_by_sig = groupby(L_files, key=lambda x: x.split("_")[-1].replace(".pkl", ""))
    H_files_by_sig = {k: list(v) for k, v in H_files_by_sig}
    L_files_by_sig = {k: list(v) for k, v in L_files_by_sig}

    assert H_files_by_sig.keys() == L_files_by_sig.keys()
    for k in H_files_by_sig.keys():
        H_sig_files = H_files_by_sig[k]
        L_sig_files = L_files_by_sig[k]
        files_written += move_files_to_scratch(k, H_sig_files, L_sig_files)

    with open("./specs_counts.txt", "w") as f:
        f.writelines([
            f"files written to {output_path}: {files_written}"
            ])
