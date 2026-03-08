import h5py

path = r"data\\h5_out\\0_1EHZ_1_A_chain_id_A.h5"
with h5py.File(path, "r") as f:
    print("keys:", list(f.keys()))
    for k in f.keys():
        d = f[k]
        print(k, d.shape, d.dtype)
    seq = f["seq"][()]
    print("seq type", type(seq), "len", len(seq))
    print("seq preview", seq[:60])
