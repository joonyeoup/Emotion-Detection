import h5py

path = "emotion_model.h5"
with h5py.File(path, "r") as f:
    for key, val in f.attrs.items():
        print(f"{key!r}: {val!r}")
