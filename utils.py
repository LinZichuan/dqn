import numpy as np

def save_npy(obj, path):
    np.save(path, obj)
    print("[*] save memory to %s" % path)

def load_npy(path):
    obj = np.load(path)
    print("[*] load memory from %s" % path)
    return obj
