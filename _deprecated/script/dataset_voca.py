class VOCARKitVAEDataset(Dataset):
    """Abstract class of VOCA-ARKit dataset for VAE"""

    def __init__(self, blendshape_coeffs_dir: str):
        """Constructor of the class

        Parameters
        ----------
        blendshape_coeffs_dir : str
            Directory of the blendshape coefficients
        """
        self.blendshape_coeffs_dir = blendshape_coeffs_dir

        self.blendshape_coeffs_paths = [
            os.path.join(self.blendshape_coeffs_dir, f)
            for f in sorted(os.listdir(self.blendshape_coeffs_dir))
        ]

        bl_seq_list = []
        for path in self.blendshape_coeffs_paths:
            bl_seq = load_blendshape_coeffs(path)
            bl_seq_list.append(bl_seq)

        self.blendshapes = torch.cat(bl_seq_list, dim=0)

        self.length = self.blendshapes.shape[0]

    def __len__(self) -> int:
        """Return the size of the dataset

        Returns
        -------
        int
            Size of the dataset
        """
        return self.length

    def __getitem__(self, index: int) -> torch.FloatTensor:
        """Return the item of the given index

        Parameters
        ----------
        index : int
            Index of the item

        Returns
        -------
        torch.FloatTensor
            "blendshape_coeffs": (1, num_blendshapes),
        """
        return self.blendshapes[index].unsqueeze(0)
