from torch.utils.data import Dataset


class TemporalGraphDataset(Dataset):
    def __init__(self, temporal_graphs):
        """
        temporal_graphs: list of DynamicGraphTemporalSignal objects
        """
        self.temporal_graphs = temporal_graphs

    def __len__(self):
        return len(self.temporal_graphs)

    def __getitem__(self, idx):
        return self.temporal_graphs[idx]
