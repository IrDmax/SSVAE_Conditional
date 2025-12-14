import torch
from models.ssvae_conditioned import SSVAE_Conditioned
from visualizations.ssvae_visualize import (
    plot_training_loss_and_accuracy,
    plot_training_curves,
    plot_kl_distribution,
    plot_prior_posterior_alignment,
    plot_latent_space,
    plot_prediction_confidence,
    plot_reconstruction_quality,
    plot_kl_vs_recon
)

def load_ssvae_conditioned_model(CHECKPOINT_PATH, DEVICE):
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # RNA dim: from encoder_rna first layer input - batch_emb_dim
    rna_encoder_weight = checkpoint['encoder_rna.backbone.0.net.0.weight']
    rna_dim = rna_encoder_weight.shape[1] - checkpoint['batch_emb.weight'].shape[1]
    
    # ADT dim: from encoder_adt first layer input - batch_emb_dim
    adt_encoder_weight = checkpoint['encoder_adt.backbone.0.net.0.weight']
    adt_dim = adt_encoder_weight.shape[1] - checkpoint['batch_emb.weight'].shape[1]
    
    # Latent dim: from encoder_rna mu_layer output
    latent_dim = checkpoint['encoder_rna.mu_layer.weight'].shape[0]
    
    # Number of batches: from batch_emb num_embeddings
    num_batches = checkpoint['batch_emb.weight'].shape[0]
    
    # Number of classes: from label_emb num_embeddings
    n_classes = checkpoint['label_emb.weight'].shape[0]
    
    # Number of covariates: from covariate embeddings
    num_covariates_cat = {}
    for key in checkpoint.keys():
        if key.startswith('covariate_cat_embeddings.') and key.endswith('.weight'):
            cov_name = key.split('.')[1]
            num_values = checkpoint[key].shape[0]
            num_covariates_cat[cov_name] = num_values
    
    # Continuous covariates: from covariate MLPs
    num_covariates_cont = []
    for key in checkpoint.keys():
        if key.startswith('covariate_cont_mlps.') and key.endswith('.0.weight'):
            cov_name = key.split('.')[1]
            num_covariates_cont.append(cov_name)
    
    print(f"Saved architecture:")
    print(f"  RNA dim: {rna_dim}, ADT dim: {adt_dim}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Classes: {n_classes}, Batches: {num_batches}")
    print(f"  Categorical covariates: {num_covariates_cat}")
    print(f"  Continuous covariates: {num_covariates_cont}")
    
    # Initialize model with inferred parameters
    model = SSVAE_Conditioned(
        rna_dim=rna_dim,
        adt_dim=adt_dim,
        latent_dim=latent_dim,
        num_batches=num_batches,
        n_classes=n_classes,
        num_covariates_cat=num_covariates_cat,
        num_covariates_cont=num_covariates_cont
    ).to(DEVICE)
    
    # Load model weights
    model.load_state_dict(checkpoint)
    model.eval()


# -------------------------------
# VISUALIZATION UTILS
# -------------------------------

def plot_ssvae_diagnostics(history, hyperparams, model, val_loader, DEVICE):
    
    # Prepare history dictionary for plot_training_curves
    plot_history = build_plot_history(history)
    
    # 0. Loss and accuracy over epochs
    plot_training_loss_and_accuracy(history, hyperparams=hyperparams)

    # 1. Training curves overview
    print("\n 1. Training Curves Overview")
    plot_training_curves(plot_history, hyperparams=hyperparams)
    
   
    model.eval()

    all_mu_q = []
    all_logvar_q = []
    all_mu_p = []
    all_logvar_p = []
    all_z = []
    all_logits = []
    all_labels = []
    all_rna_true = []
    all_rna_recon = []
    all_adt_true = []
    all_adt_recon = []

    with torch.no_grad():
        for val_batch in val_loader:
            rna_val = val_batch['rna'].to(DEVICE)
            adt_val = val_batch['adt'].to(DEVICE)
            label_val = val_batch['label'].to(DEVICE)
            batch_val = val_batch['batch'].to(DEVICE)

            covs_cat = val_batch.get("covariates_cat", None)
            if covs_cat is not None:
                covs_cat = {k: v.to(DEVICE) for k, v in covs_cat.items()}
            
            covs_cont = val_batch.get("covariates_cont", None)
            if covs_cont is not None:
                covs_cont = {k: v.to(DEVICE) for k, v in covs_cont.items()}

            outputs = model(
                rna_val,
                adt_val,
                batch_val,
                labels=label_val,
                covariates_cat=covs_cat,
                covariates_cont=covs_cont
            )

            # posterior stats
            all_mu_q.append(outputs["mu_fused"])
            all_logvar_q.append(outputs["logvar_fused"])
            all_z.append(outputs["z_fused"])
            all_logits.append(outputs["label_logits"])
            all_labels.append(label_val)

            # store recon for reconstruction-quality plots
            all_rna_true.append(rna_val)
            all_rna_recon.append(outputs["rna_recon"])
            all_adt_true.append(adt_val)
            all_adt_recon.append(outputs["adt_recon"])

            # prior stats
            if model.prior_net is not None:
                # Use the embeddings already computed in forward pass
                label_emb = outputs["label_emb"]
                covariate_emb = outputs["covariate_emb"]
                
                mu_p, logvar_p = model.prior_net(label_emb, covariate_emb)
            else:
                mu_p = torch.zeros_like(outputs["mu_fused"])
                logvar_p = torch.zeros_like(outputs["logvar_fused"])

            all_mu_p.append(mu_p)
            all_logvar_p.append(logvar_p)

    # concatenate across all batches
    mu_q = torch.cat(all_mu_q, dim=0)
    logvar_q = torch.cat(all_logvar_q, dim=0)
    mu_p = torch.cat(all_mu_p, dim=0)
    logvar_p = torch.cat(all_logvar_p, dim=0)
    z_full = torch.cat(all_z, dim=0)
    logits_full = torch.cat(all_logits, dim=0)
    labels_full = torch.cat(all_labels, dim=0)

    rna_true_full = torch.cat(all_rna_true, dim=0)
    rna_recon_full = torch.cat(all_rna_recon, dim=0)
    adt_true_full = torch.cat(all_adt_true, dim=0)
    adt_recon_full = torch.cat(all_adt_recon, dim=0)

    # Compute per-sample KL
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    kl_values = 0.5 * (
        logvar_p - logvar_q +
        (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0
    ).sum(dim=-1)

    # ------------------------------------------------------------------
    # 2. KL Distribution plot
    # ------------------------------------------------------------------
    print("\n 2. KL Distribution (Collapse Detection)")
    plot_kl_distribution(outputs["kl_values"], epoch=len(history["train_loss"]), hyperparams=hyperparams)

    # ------------------------------------------------------------------
    # 3. Prior–Posterior alignment plot
    # ------------------------------------------------------------------
    print("\n 3. Prior–Posterior Alignment")
    plot_prior_posterior_alignment(
        mu_q, mu_p,
        logvar_q, logvar_p,
        epoch=len(history["train_loss"]),
        hyperparams=hyperparams
    )

    # ------------------------------------------------------------------
    # 4. Latent Space Visualization (UMAP)
    # ------------------------------------------------------------------
    print("\n 4. Latent Space Structure (UMAP)")
    plot_latent_space(outputs["z_fused"], labels=label_val, method='umap', hyperparams=hyperparams)

    # ------------------------------------------------------------------
    # 5. Prediction Confidence
    # ------------------------------------------------------------------
    print("\n 5. Prediction Confidence Distribution")
    plot_prediction_confidence(outputs['label_logits'], epoch=len(history["train_loss"]), hyperparams=hyperparams)

    # ------------------------------------------------------------------
    # 6. Reconstruction Quality
    # ------------------------------------------------------------------
    print("\n 6. Reconstruction Quality")
    plot_reconstruction_quality(
        rna_val, outputs['rna_recon'],
        adt_val, outputs['adt_recon'],
        epoch=len(history["train_loss"]),
        hyperparams=hyperparams
    )

    # ------------------------------------------------------------------
    # 7. KL vs Reconstruction
    # ------------------------------------------------------------------
    print("\n 7. KL vs Reconstruction Loss")
    recon_combined = (
        (rna_val - outputs['rna_recon']).pow(2).sum(1) +
        (adt_val - outputs['adt_recon']).pow(2).sum(1)
    )
    plot_kl_vs_recon(outputs["kl_values"], recon_combined, epoch=len(history["train_loss"]), hyperparams=hyperparams)



def build_plot_history(history):
    """
    Convert trainer history (list of dicts) into plotting history.
    """
    ph = {
        "recon_rna": [],
        "recon_adt": [],
        "kl": [],
        "class_ce": [],
        "class_entropy": [],
        "cov_cat_ce": [],
        "cov_cat_entropy": []
    }
    
    for d in history["train_loss_dict"]:
        ph["recon_rna"].append(d["recon"])   # RNA+ADT recon combined — OK for now
        ph["recon_adt"].append(0.0)          # Until you split recon modalities
        ph["kl"].append(d["kl"])
        ph["class_ce"].append(d["class_ce"])
        ph["class_entropy"].append(d["class_entropy"])
        ph["cov_cat_ce"].append(d["cov_cat_ce"])
        ph["cov_cat_entropy"].append(d["cov_cat_entropy"])
        ph["recon_rna"].append(d["recon_rna"])
        ph["recon_adt"].append(d["recon_adt"])
    
    return ph

def collect_posterior_prior(model, loader, device):
    """
    Collect μ_q, μ_p, σ²_q, σ²_p, and z for the entire validation set.
    Returns tensors concatenated across all batches.
    """
    model.eval()

    all_mu_q = []
    all_logvar_q = []
    all_mu_p = []
    all_logvar_p = []
    all_z = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            rna = batch['rna'].to(device)
            adt = batch['adt'].to(device)
            batch_idx = batch['batch'].to(device)
            labels = batch['label'].to(device)

            covs_cat = batch.get("covariates_cat", None)
            if covs_cat:
                covs_cat = {k: v.to(device) for k, v in covs_cat.items()}
            
            covs_cont = batch.get("covariates_cont", None)
            if covs_cont:
                covs_cont = {k: v.to(device) for k, v in covs_cont.items()}

            out = model(
                rna, adt, batch_idx,
                labels=labels,
                covariates_cat=covs_cat,
                covariates_cont=covs_cont
            )

            # posterior
            all_mu_q.append(out["mu_fused"])
            all_logvar_q.append(out["logvar_fused"])
            all_z.append(out["z_fused"])
            all_labels.append(labels)

            # prior p(z | y, covariates)
            if model.prior_net is not None:
                # Use the embeddings already computed in forward pass
                label_emb = out["label_emb"]
                covariate_emb = out["covariate_emb"]
                
                mu_p, logvar_p = model.prior_net(label_emb, covariate_emb)
            else:
                mu_p = torch.zeros_like(out["mu_fused"])
                logvar_p = torch.zeros_like(out["logvar_fused"])

            all_mu_p.append(mu_p)
            all_logvar_p.append(logvar_p)

    # Concatenate everything
    return {
        "mu_q": torch.cat(all_mu_q, dim=0),
        "logvar_q": torch.cat(all_logvar_q, dim=0),
        "mu_p": torch.cat(all_mu_p, dim=0),
        "logvar_p": torch.cat(all_logvar_p, dim=0),
        "z": torch.cat(all_z, dim=0),
        "labels": torch.cat(all_labels, dim=0)
    }