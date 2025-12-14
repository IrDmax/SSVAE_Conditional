import torch
import scanpy as sc
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from collections import defaultdict
from models.ssvae_wrapper import FrozenSSVAEEncoder


class LongitudinalDataset(Dataset):
    """
    Dataset for encoding patient-timepoint sequences using a pretrained SSVAE.
    Works with HC + T0 now, and automatically supports T7/T23/T30 later.
    """

    def __init__(self, datapath, ssvae_model, device='cpu'):
        self.datapath = datapath
        self.device = device

        # Wrap SSVAE model with frozen encoder
        self.encoder = FrozenSSVAEEncoder(ssvae_model).to(device)

        # Extract sequences
        (self.patient_sequences,
         self.patient_ids,
         self.timepoint_order) = self._extract_sequences()

    # ------------------------------------------------------
    # STATIC: Batch mapping
    # ------------------------------------------------------
    @staticmethod
    def _get_batch_mapping(adata):
        unique_batches = sorted(adata.obs['Batch'].unique())
        return {batch: idx for idx, batch in enumerate(unique_batches)}

    # ------------------------------------------------------
    # MAIN EXTRACTION FUNCTION
    # ------------------------------------------------------
    def _extract_sequences(self):
        print(f"Loading temporal data from {self.datapath}...")
        adata = sc.read_h5ad(self.datapath)


        # print(adata.obs['Timepoint'].unique())
        # print(adata.obs['sample_name'].unique()[:20])
        # print(adata.obs['sample_id'].unique()[:20])


        # cols_to_test = [
        #     "Subject", "Donor", "Sample", "sample", "sample_id",
        #     "sample_name", "sample_alt_id", "sample_label",
        #     "hash.ID", "HTO_maxID", "INT_ID"
        # ]

        # for col in cols_to_test:
        #     if col in adata.obs.columns:
        #         print(f"\n=== {col} ===")
        #         print(adata.obs[col].unique()[:20])



        # --------------------------------------------
        # Parse patient + timepoint from sample_name
        # --------------------------------------------
        if "sample_name" not in adata.obs.columns:
            raise ValueError("sample_name column required for parsing patient/timepoint")

        parsed_patients = []
        parsed_timepoints = []

        for name in adata.obs["sample_name"]:
            parts = name.split("_")
            patient = "_".join(parts[:-1])     # e.g., HGR0000079
            tp = parts[-1]                     # e.g., T0 or HC
            parsed_patients.append(patient)
            parsed_timepoints.append(tp)

        adata.obs["parsed_patient"] = parsed_patients
        adata.obs["parsed_timepoint"] = parsed_timepoints

        # --------------------------------------------
        # Determine timepoint order
        # --------------------------------------------
        def tp_key(tp):
            if tp == "HC":  # always first
                return (0, -1)
            if tp.startswith("T"):
                try:
                    return (1, int(tp[1:]))  # T0 → 0
                except:
                    return (1, 9999)
            return (2, 9999)

        timepoint_order = sorted(adata.obs["parsed_timepoint"].unique(), key=tp_key)
        print("Using timepoints:", timepoint_order)

        # --------------------------------------------
        # Group data by patient × timepoint
        # --------------------------------------------
        patient_timepoint_data = defaultdict(dict)
        batch_map = self._get_batch_mapping(adata)

        for tp in timepoint_order:
            tp_cells = adata[adata.obs["parsed_timepoint"] == tp]

            for patient in tp_cells.obs["parsed_patient"].unique():
                p_cells = tp_cells[tp_cells.obs["parsed_patient"] == patient]

                # RNA
                rna = p_cells.layers["log1p_norm"]
                if sp.issparse(rna):
                    rna = rna.toarray()
                rna_mean = torch.tensor(rna.mean(axis=0), dtype=torch.float32)

                # ADT
                adt = p_cells.obsm["ADT_clr"]
                if sp.issparse(adt):
                    adt = adt.toarray()
                adt_mean = torch.tensor(adt.mean(axis=0), dtype=torch.float32)

                # Batch → index
                batch_val = p_cells.obs["Batch"].iloc[0]
                batch_idx = torch.tensor(batch_map[batch_val], dtype=torch.long)

                patient_timepoint_data[patient][tp] = {
                    "rna": rna_mean,
                    "adt": adt_mean,
                    "batch": batch_idx
                }

        # --------------------------------------------
        # Encode latents per patient
        # --------------------------------------------
        patient_sequences = {}

        with torch.no_grad():
            for patient, tp_dict in patient_timepoint_data.items():

                latent_list = []

                for tp in timepoint_order:
                    if tp not in tp_dict:
                        continue

                    data = tp_dict[tp]

                    z = self.encoder(
                        data["rna"].unsqueeze(0).to(self.device),
                        data["adt"].unsqueeze(0).to(self.device),
                        data["batch"].unsqueeze(0).to(self.device),
                        covariate_indices=None
                    )

                    z = z.squeeze(0).cpu()
                    latent_list.append(z)

                if len(latent_list) > 0:
                    patient_sequences[patient] = torch.stack(latent_list)

        patient_ids = list(patient_sequences.keys())
        print(f"Extracted sequences for {len(patient_ids)} patients")

        # print("\nDEBUG:")
        # print("Patients with extracted timepoints:")
        # for pid, tps in patient_timepoint_data.items():
        #     print(pid, list(tps.keys()))

        # print("\nPatients that passed the complete-timepoint filter:")
        # print([pid for pid, seq in patient_sequences.items()])

        # print(f"\nTotal patients kept: {len(patient_sequences)}")

        return patient_sequences, patient_ids, timepoint_order

    # ------------------------------------------------------
    # STANDARD DATASET METHODS
    # ------------------------------------------------------
    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        return self.patient_sequences[pid]

    def get_sequence_length(self):
        return len(self.timepoint_order)

    def get_latent_dim(self):
        first = self.patient_ids[0]
        return self.patient_sequences[first].shape[1]
