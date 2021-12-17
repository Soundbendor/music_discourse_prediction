import cudf

class EmoVad():
    def __init__(self, path: str):
        self.wlist = cudf.read_csv(path, names=['Word','Valence','Arousal','Dominance'], skiprows=1,  sep='\t')