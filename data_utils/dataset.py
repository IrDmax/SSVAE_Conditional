from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
import torch
import scanpy as sc
import numpy as np
import scipy.sparse as sp

class SingleCellDataset(Dataset):
    def __init__(self, datapath, use_covariates=False, random_state=42):
        self.datapath = datapath
        self.random_state = random_state
        (
            self.rna_matrix, 
            self.adt_matrix,
            self.batch_idx,
            self.num_batches,
            self.label_idx,
            self.num_classes,
            self.labels,
            self.covariates_cat,
            self.covariates_cont
        ) = self._load_data(use_covariates)   
   
    def __len__(self):
        return self.rna_matrix.shape[0]

    def __getitem__(self, idx):
        
        # Load RNA data
        rna = self.rna_matrix[idx]
        if sp.issparse(rna):
            rna = rna.toarray().squeeze()
        else:
            rna = np.asarray(rna).squeeze()

        # Load ADT data
        adt = self.adt_matrix[idx]
        if sp.issparse(adt):
            adt = adt.toarray().squeeze()
        else:
            adt = np.asarray(adt).squeeze()

        sample = {
            "rna": torch.from_numpy(rna).float(),   # (batch_size, rna_dim)
            "adt": torch.from_numpy(adt).float(),   # (batch_size, adt_dim)
            "batch": self.batch_idx[idx]            # (batch_size, ) long tensor
        }

        # Optional labels
        sample["label"] = self.label_idx[idx]  # (batch_size, ) long tensor

        # Optional categorical covariates
        if self.covariates_cat is not None:
            sample["covariates_cat"] = {}
            for key, tensor in self.covariates_cat.items():
                sample["covariates_cat"][key] = tensor[idx]

        # Optional continuous covariates
        if self.covariates_cont is not None:
            sample["covariates_cont"] = {}
            for key, tensor in self.covariates_cont.items():
                sample["covariates_cont"][key] = tensor[idx]

        return sample


    def _load_data(self, use_covariates):

        adata = sc.read_h5ad(self.datapath)
        
        # Encode batch
        _, batch_idx, num_batches = self._encode_categorical(adata, "Batch")

        # Labels - encode as indices for embedding
        if "WCTcoursecelltype" in adata.obs.columns:
            labels, label_idx, num_classes = self._encode_categorical(adata, "WCTcoursecelltype")
        else:
            labels = None
            label_idx = None
            num_classes = 0
        print(f"  Loaded dataset with {adata.n_obs} cells, {adata.n_vars} genes, {num_batches} batches, {num_classes} classes.")   
        # Optional covariates
        if use_covariates:
            covariates_cat = {} # dictionary of categorical covs
            covariates_cont = {} # dictionary of continuous covs
            
            if "Age_group" in adata.obs.columns:
                _, age_group_idx, _ = self._encode_categorical(adata, "Age_group")
                covariates_cat["age_group_idx"] = age_group_idx

            if "Inflammation_score" in adata.obs.columns:
                covariates_cont["inflammation_score"] = self._encode_continuous(
                    adata, "Inflammation_score"
                )

            if "Immunosenescence_score" in adata.obs.columns:
                covariates_cont["immunosenescence_score"] = self._encode_continuous(
                    adata, "Immunosenescence_score"
                )

        else:
            covariates_cat = None
            covariates_cont = None

        # Extract data matrices
        rna_matrix = adata.layers["log1p_norm"]
        adt_matrix = adata.obsm['ADT_clr']
        
        return rna_matrix, adt_matrix, batch_idx, num_batches, label_idx, num_classes, labels, covariates_cat, covariates_cont
    
    @staticmethod
    def _encode_continuous(adata, column_name):
        """
        Extract continuous obs column as float tensor, with NaNs for missing values.
        """
        # Extract column and convert to float numpy array
        col = adata.obs[column_name].astype(float).to_numpy()

        # Compute mean/std over observed values only (exclude NaNs)
        mask = ~np.isnan(col)
        if mask.sum() == 0:
            raise ValueError(f"Column '{column_name}' has no valid numeric values.")

        mean = col[mask].mean()
        std = col[mask].std()
        if std == 0:
            std = 1.0  # avoid divide-by-zero

        # Z-score normalization
        col_z = (col - mean) / std  # NaN stays NaN automatically

        # Convert to tensor (float32), preserving NaNs
        return torch.tensor(col_z, dtype=torch.float32)

    @staticmethod
    def _encode_categorical(adata, column_name):
        """
        Encode a categorical obs column for SSVAE:
        - valid categories → 0..C-1
        - missing labels → -1
        """
        raw = adata.obs[column_name]

        # Detect missing labels
        mask_missing = raw.isna()

        # Extract valid category strings
        labeled_values = raw[~mask_missing].astype(str).values
        unique_values = np.unique(labeled_values)
        num_categories = len(unique_values)

        # Create mapping for real categories
        value_to_idx = {val: i for i, val in enumerate(unique_values)}

        # Initialize all labels as -1
        indices = np.full(len(raw), -1, dtype=int)

        # Encode only the non-missing values
        indices[~mask_missing] = np.array([value_to_idx[v] for v in labeled_values])

        return raw, torch.tensor(indices, dtype=torch.long), num_categories


    def stratified_split(self, subset_size=None, val_size=0.1, test_size=0.1, min_samples_per_class=10):
        """
        Optionally stratified subsample the dataset, then stratified train/val/test split.

        Parameters
        ----------
        subset_size : int or None
            - If None or 0 → use the full dataset.
            - Otherwise → take a stratified subset of this size first.
        val_size : float
            Fraction of (sub)set to allocate for validation.
        test_size : float
            Fraction of (sub)set to allocate for test.
        min_samples_per_class : int
            Minimum number of samples per class after subsampling. Classes with fewer
            samples will be filtered out. Default is 10 (safe for stratified split).

        Returns
        -------
        train_dataset, val_dataset, test_dataset
        """
        
        if self.labels is None:
            raise ValueError("Cannot perform stratified split without labels. Dataset must have 'WCTcoursecelltype' column.")

        labels = np.array(self.labels)
        full_indices = np.arange(len(self))

        # ---------------------------------------------------
        # 1. SUBSAMPLING STEP (OPTIONAL)
        # ---------------------------------------------------
        if subset_size is None or subset_size == 0 or subset_size >= len(self):
            # ☑ Use full dataset
            subset_idx = full_indices
            subset_labels = labels
        else:
            # ☑ Take stratified subset
            subset_fraction = subset_size / len(self)

            sss_sub = StratifiedShuffleSplit(
                n_splits=1,
                train_size=subset_fraction,
                random_state=self.random_state
            )

            subset_idx, _ = next(sss_sub.split(np.zeros(len(self)), labels))
            subset_labels = labels[subset_idx]
            
            # ---------------------------------------------------
            # FILTER OUT CLASSES WITH TOO FEW SAMPLES
            # ---------------------------------------------------
            unique_labels, label_counts = np.unique(subset_labels, return_counts=True)
            valid_classes = unique_labels[label_counts >= min_samples_per_class]
            
            # Keep only samples from valid classes
            valid_mask = np.isin(subset_labels, valid_classes)
            subset_idx = subset_idx[valid_mask]
            subset_labels = subset_labels[valid_mask]
            
            # Report filtering
            n_filtered = len(unique_labels) - len(valid_classes)
            if n_filtered > 0:
                print(f"Filtered out {n_filtered} classes with < {min_samples_per_class} samples:")
                print(f"Remaining: {len(valid_classes)}/{len(unique_labels)} classes")

        # ---------------------------------------------------
        # 2. TRAIN / VAL / TEST SPLIT WITHIN (SUB)SET
        # ---------------------------------------------------
        vt_fraction = val_size + test_size
        train_fraction = 1.0 - vt_fraction
        assert train_fraction > 0, "val_size + test_size must be < 1"

        # Split (subset) → train + temp
        sss1 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=vt_fraction,
            random_state=self.random_state
        )

        train_rel_idx, temp_rel_idx = next(sss1.split(np.zeros(len(subset_labels)), subset_labels))

        # Split temp → val + test
        test_fraction_rel = test_size / (val_size + test_size)

        sss2 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_fraction_rel,
            random_state=self.random_state
        )

        val_rel_idx, test_rel_idx = next(
            sss2.split(np.zeros(len(temp_rel_idx)), subset_labels[temp_rel_idx])
        )

        # Map into global indices
        train_idx = subset_idx[train_rel_idx]
        val_idx   = subset_idx[temp_rel_idx[val_rel_idx]]
        test_idx  = subset_idx[temp_rel_idx[test_rel_idx]]

        # ---------------------------------------------------
        # 3. BUILD FINAL DATASETS
        # ---------------------------------------------------
        train_dataset = Subset(self, train_idx)
        val_dataset   = Subset(self, val_idx)
        test_dataset  = Subset(self, test_idx)

        return train_dataset, val_dataset, test_dataset
