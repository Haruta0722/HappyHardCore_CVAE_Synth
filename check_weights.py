import h5py
fpath = "checkpoints_cvae/cvae_062.weights.h5"
with h5py.File(fpath, "r") as f:
    keys = []
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            keys.append(name)
    f.visititems(walk)
for k in keys:
    print(k)
print("total keys:", len(keys))