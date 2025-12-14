"""
Visualization tools for Semi-Supervised VAE (SSVAE) diagnostics.

This module provides comprehensive plotting utilities to monitor:
- Training dynamics (loss curves, KL divergence)
- Posterior collapse detection
- Prior-posterior alignment
- Latent space structure
- Classification confidence
- Reconstruction quality
- Entropy trends
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import umap
from sklearn.decomposition import PCA


def plot_training_loss_and_accuracy(history, hyperparams=None):
    """
    Plot training and validation loss and accuracy curves using the unified
    history format produced by the new trainer.

    Expected keys:
        - 'train_loss'
        - 'val_loss'
        - 'train_acc'
        - 'val_acc'
    
    Args:
        hyperparams (dict, optional): Dictionary containing hyperparameters to display
    """

    # ---------------------------------------------------------
    # Sanity check: ensure required keys exist
    # ---------------------------------------------------------
    required_keys = ["train_loss", "val_loss"]
    for k in required_keys:
        if k not in history:
            raise KeyError(f"history dictionary missing required key: {k}")

    has_accuracy = ("train_acc" in history) and ("val_acc" in history)

    # ---------------------------------------------------------
    # LOSS PLOT
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.plot(epochs, history["train_loss"], "b-o",
             label="Training Loss", linewidth=2, markersize=4)
    plt.plot(epochs, history["val_loss"], "r-s",
             label="Validation Loss", linewidth=2, markersize=4)

    # Best validation loss
    best_val_idx = int(np.argmin(history["val_loss"]))
    best_val_epoch = best_val_idx + 1

    plt.axvline(best_val_epoch, color="g", linestyle="--", alpha=0.7,
                label=f"Best Val Loss (Epoch {best_val_epoch})")
    plt.scatter([best_val_epoch],
                [history["val_loss"][best_val_idx]],
                color="g", s=150, zorder=5,
                edgecolors="darkgreen", linewidths=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    
    # Add hyperparameters text box
    if hyperparams:
        hp_text = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
        plt.text(0.02, 0.98, hp_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # ACCURACY PLOT (optional)
    # ---------------------------------------------------------
    if has_accuracy:
        plt.figure(figsize=(7, 5))

        epochs = range(1, len(history["train_acc"]) + 1)

        plt.plot(epochs, history["train_acc"], "b-o",
                 label="Training Accuracy", linewidth=2, markersize=4)
        plt.plot(epochs, history["val_acc"], "r-s",
                 label="Validation Accuracy", linewidth=2, markersize=4)

        # Best validation accuracy
        best_acc_idx = int(np.argmax(history["val_acc"]))
        best_acc_epoch = best_acc_idx + 1

        plt.axvline(best_acc_epoch, color="purple", linestyle="--", alpha=0.7,
                    label=f"Best Val Acc (Epoch {best_acc_epoch})")
        plt.scatter([best_acc_epoch],
                    [history["val_acc"][best_acc_idx]],
                    color="purple", s=150, zorder=5,
                    edgecolors="indigo", linewidths=2)

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        
        # Add hyperparameters text box
        if hyperparams:
            hp_text = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
            plt.text(0.02, 0.98, hp_text, transform=plt.gca().transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()


def plot_training_curves(history, hyperparams=None):
    """
    Plot comprehensive training curves including reconstruction, KL, classification, and covariate losses.
    
    Args:
        history (dict): Dictionary containing loss histories with keys:
            - 'recon_rna': RNA reconstruction loss
            - 'recon_adt': ADT reconstruction loss
            - 'kl': KL divergence
            - 'class_ce': Classification cross-entropy
            - 'class_entropy': Classification entropy
            - 'cov_cat_ce': Covariate cross-entropy
            - 'cov_entropy': Covariate entropy
    """
    plt.figure(figsize=(14, 10))
    
    # Reconstruction + KL
    plt.subplot(2, 2, 1)
    if 'recon_rna' in history:
        plt.plot(history['recon_rna'], label="RNA Recon")
    if 'recon_adt' in history:
        plt.plot(history['recon_adt'], label="ADT Recon")
    if 'kl' in history:
        plt.plot(history['kl'], label="KL")
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Classification Terms
    plt.subplot(2, 2, 2)
    if 'class_ce' in history:
        plt.plot(history['class_ce'], label="CE")
    if 'class_entropy' in history:
        plt.plot(history['class_entropy'], label="Entropy")
    plt.title("Classification Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Covariate Terms
    plt.subplot(2, 2, 3)
    if 'cov_cat_ce' in history:
        plt.plot(history['cov_cat_ce'], label="Cov Cat CE")
    if 'cov_cat_entropy' in history:
        plt.plot(history['cov_cat_entropy'], label="Cov Cat Entropy")
    plt.title("Covariate Cat Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # KL-only plot for collapse detection
    plt.subplot(2, 2, 4)
    if 'kl' in history:
        plt.plot(history['kl'], label="KL")
        plt.axhline(0.1, color="red", linestyle="--", label="collapse threshold")
    plt.title("KL Divergence Monitoring")
    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add hyperparameters text box to the figure
    if hyperparams:
        hp_text = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
        plt.gcf().text(0.5, 0.01, hp_text, ha='center', fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)
    plt.show()


def plot_kl_distribution(kl_values, epoch, hyperparams=None):
    """
    Plot distribution of KL divergence values to detect posterior collapse or explosion.
    
    Args:
        kl_values (torch.Tensor or np.ndarray): Per-sample KL values
        epoch (int): Current epoch number
        hyperparams (dict, optional): Dictionary containing hyperparameters to display
    
    Interpretation:
        - Most values near 0 → posterior collapse
        - Wide spread → healthy training
        - Very large values → KL explosion
    """
    if isinstance(kl_values, torch.Tensor):
        kl_values = kl_values.detach().cpu().numpy()
    
    plt.figure(figsize=(8, 5))
    plt.hist(kl_values, bins=40, alpha=0.8, edgecolor='black')
    plt.title(f"KL Distribution at Epoch {epoch}")
    plt.xlabel("KL(q||p)")
    plt.ylabel("Count")
    plt.axvline(0.01, color="red", linestyle="--", linewidth=2, label="collapse region")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add hyperparameters text box
    if hyperparams:
        hp_text = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
        plt.text(0.98, 0.98, hp_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def plot_prior_posterior_alignment(mu_q, mu_p, logvar_q, logvar_p, epoch, hyperparams=None):
    """
    Visualize alignment between encoder posterior and learned prior.
    
    Args:
        mu_q (torch.Tensor): Mean from encoder (posterior)
        mu_p (torch.Tensor): Mean from prior network
        logvar_q (torch.Tensor): Log-variance from encoder
        logvar_p (torch.Tensor): Log-variance from prior network
        epoch (int): Current epoch number
        hyperparams (dict, optional): Dictionary containing hyperparameters to display
    
    Interpretation:
        - Points near diagonal → good alignment
        - Scattered points → prior not learning correctly
    """
    if isinstance(mu_q, torch.Tensor):
        mu_q = mu_q.detach().cpu().numpy()
    if isinstance(mu_p, torch.Tensor):
        mu_p = mu_p.detach().cpu().numpy()
    if isinstance(logvar_q, torch.Tensor):
        logvar_q = logvar_q.detach().cpu().numpy()
    if isinstance(logvar_p, torch.Tensor):
        logvar_p = logvar_p.detach().cpu().numpy()
    
    plt.figure(figsize=(14, 5))

    # Compare means
    plt.subplot(1, 2, 1)
    plt.scatter(mu_q.flatten(), mu_p.flatten(), s=4, alpha=0.5)
    plt.xlabel("μ_q (encoder)")
    plt.ylabel("μ_p (prior)")
    plt.title(f"Mean Alignment at Epoch {epoch}")
    
    # Add diagonal reference line
    lims = [
        np.min([plt.xlim()[0], plt.ylim()[0]]),
        np.max([plt.xlim()[1], plt.ylim()[1]])
    ]
    plt.plot(lims, lims, 'r--', alpha=0.5, label='perfect alignment')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Compare variances
    plt.subplot(1, 2, 2)
    var_q = np.exp(logvar_q)
    var_p = np.exp(logvar_p)
    plt.scatter(var_q.flatten(), var_p.flatten(), s=4, alpha=0.5)
    plt.xlabel("σ²_q (encoder)")
    plt.ylabel("σ²_p (prior)")
    plt.title(f"Variance Alignment at Epoch {epoch}")
    
    # Add diagonal reference line
    lims = [
        np.min([plt.xlim()[0], plt.ylim()[0]]),
        np.max([plt.xlim()[1], plt.ylim()[1]])
    ]
    plt.plot(lims, lims, 'r--', alpha=0.5, label='perfect alignment')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add hyperparameters text box to the figure
    if hyperparams:
        hp_text = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
        plt.gcf().text(0.5, 0.02, hp_text, ha='center', fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()


def plot_latent_space(z, labels=None, title="Latent Space", method='umap', hyperparams=None, **kwargs):
    """
    Visualize latent space using UMAP or PCA.
    
    Args:
        z (torch.Tensor or np.ndarray): Latent representations
        labels (np.ndarray, optional): Class labels for coloring
        title (str): Plot title
        method (str): 'umap' or 'pca'
        hyperparams (dict, optional): Dictionary containing hyperparameters to display
        **kwargs: Additional arguments for UMAP or PCA
    
    Interpretation:
        - Well-separated clusters → good representation learning
        - Collapsed to single point → posterior collapse
        - Mixed clusters → model needs more training
    """
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    
    # Dimensionality reduction
    if method.lower() == 'umap':
        n_neighbors = kwargs.get('n_neighbors', 20)
        min_dist = kwargs.get('min_dist', 0.2)
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    else:
        reducer = PCA(n_components=2)
    
    z2d = reducer.fit_transform(z)

    plt.figure(figsize=(8, 7))
    if labels is None:
        plt.scatter(z2d[:, 0], z2d[:, 1], s=4, alpha=0.7)
    else:
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        scatter = plt.scatter(z2d[:, 0], z2d[:, 1], s=4, alpha=0.7, c=labels, cmap="tab20")
        plt.colorbar(scatter, label="Class")

    plt.title(title)
    plt.xlabel(f"{method.upper()} 1")
    plt.ylabel(f"{method.upper()} 2")
    plt.grid(True, alpha=0.3)
    
    # Add hyperparameters text box
    if hyperparams:
        hp_text = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
        plt.text(0.02, 0.02, hp_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def plot_prediction_confidence(logits, epoch, hyperparams=None):
    """
    Plot distribution of prediction confidence for unlabeled samples.
    
    Args:
        logits (torch.Tensor): Logits from classifier
        epoch (int): Current epoch number
        hyperparams (dict, optional): Dictionary containing hyperparameters to display
    
    Interpretation:
        - All predictions >0.95 → too confident (bad for SSL)
        - Wide spread → good uncertainty quantification
        - Very low → undertrained model
    """
    if isinstance(logits, torch.Tensor):
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    else:
        probs = logits
    
    max_conf = probs.max(axis=1)

    plt.figure(figsize=(8, 5))
    plt.hist(max_conf, bins=30, alpha=0.8, edgecolor='black')
    plt.axvline(0.95, color="red", linestyle="--", linewidth=2, label='overconfidence threshold')
    plt.title(f"Prediction Confidence at Epoch {epoch}")
    plt.xlabel("Max Softmax Probability")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add hyperparameters text box
    if hyperparams:
        hp_text = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
        plt.text(0.02, 0.98, hp_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def plot_reconstruction_quality(true_rna, pred_rna, true_adt, pred_adt, epoch, n_genes=50, sample_idx=None, hyperparams=None):
    """
    Visualize reconstruction quality for RNA and ADT modalities.
    
    Args:
        true_rna (torch.Tensor or np.ndarray): True RNA expression
        pred_rna (torch.Tensor or np.ndarray): Predicted RNA expression
        true_adt (torch.Tensor or np.ndarray): True ADT expression
        pred_adt (torch.Tensor or np.ndarray): Predicted ADT expression
        epoch (int): Current epoch number
        n_genes (int): Number of genes to plot for RNA
        sample_idx (int, optional): Specific sample index to plot
        hyperparams (dict, optional): Dictionary containing hyperparameters to display
    """
    if isinstance(true_rna, torch.Tensor):
        true_rna = true_rna.detach().cpu().numpy()
    if isinstance(pred_rna, torch.Tensor):
        pred_rna = pred_rna.detach().cpu().numpy()
    if isinstance(true_adt, torch.Tensor):
        true_adt = true_adt.detach().cpu().numpy()
    if isinstance(pred_adt, torch.Tensor):
        pred_adt = pred_adt.detach().cpu().numpy()
    
    if sample_idx is None:
        idx = np.random.choice(len(true_rna), size=1)[0]
    else:
        idx = sample_idx

    plt.figure(figsize=(14, 5))

    # RNA
    plt.subplot(1, 2, 1)
    gene_indices = range(min(n_genes, true_rna.shape[1]))
    plt.plot(gene_indices, true_rna[idx, :n_genes], 'o-', label="true", alpha=0.7)
    plt.plot(gene_indices, pred_rna[idx, :n_genes], 's-', label="recon", alpha=0.7)
    plt.title(f"RNA Reconstruction (Epoch {epoch}, Sample {idx})")
    plt.xlabel("Gene Index")
    plt.ylabel("Expression")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ADT
    plt.subplot(1, 2, 2)
    protein_indices = range(true_adt.shape[1])
    plt.plot(protein_indices, true_adt[idx], 'o-', label="true", alpha=0.7)
    plt.plot(protein_indices, pred_adt[idx], 's-', label="recon", alpha=0.7)
    plt.title(f"ADT Reconstruction (Epoch {epoch}, Sample {idx})")
    plt.xlabel("Protein Index")
    plt.ylabel("Expression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add hyperparameters text box to the figure
    if hyperparams:
        hp_text = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
        plt.gcf().text(0.5, 0.02, hp_text, ha='center', fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()


def plot_entropy_trends(class_entropy, cov_entropy):
    """
    Monitor entropy trends for classification and covariate prediction.
    
    Args:
        class_entropy (list): History of classification entropy values
        cov_entropy (list): History of covariate entropy values
    
    Interpretation:
        - Entropy too low → overconfidence collapse
        - Entropy too high → model not learning constraints
        - Around 0.5 → healthy balance
    """
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(class_entropy, label="Class Entropy", linewidth=2)
    plt.axhline(0.5, color='orange', linestyle="--", linewidth=2, label='reference')
    plt.title("Class Entropy Trend")
    plt.xlabel("Epoch")
    plt.ylabel("Entropy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(cov_entropy, label="Covariate Entropy", linewidth=2)
    plt.axhline(0.5, color='orange', linestyle="--", linewidth=2, label='reference')
    plt.title("Covariate Entropy Trend")
    plt.xlabel("Epoch")
    plt.ylabel("Entropy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# def plot_latent_drift(z_list, method='pca'):
#     """
#     Track stability of latent space across training epochs.
    
#     Args:
#         z_list (list): List of z tensors captured at several epochs
#         method (str): 'pca' or 'umap'
    
#     Interpretation:
#         - Overlapping distributions → stable training
#         - Drifting apart → unstable or still learning structure
#     """
#     reducer = PCA(n_components=2) if method.lower() == 'pca' else umap.UMAP()

#     plt.figure(figsize=(9, 7))

#     for epoch, z in enumerate(z_list):
#         if isinstance(z, torch.Tensor):
#             z = z.detach().cpu().numpy()
#         z2d = reducer.fit_transform(z)
#         plt.scatter(z2d[:, 0], z2d[:, 1], s=3, alpha=0.3, label=f"epoch {epoch}")

#     plt.title("Latent Drift Across Training")
#     plt.xlabel(f"{method.upper()} 1")
#     plt.ylabel(f"{method.upper()} 2")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()


def plot_kl_vs_recon(kl_vals, recon_vals, epoch, hyperparams=None):
    """
    Scatter plot of KL vs reconstruction loss to detect collapse.
    
    Args:
        kl_vals (torch.Tensor or np.ndarray): KL divergence values
        recon_vals (torch.Tensor or np.ndarray): Reconstruction loss values
        epoch (int): Current epoch number
        hyperparams (dict, optional): Dictionary containing hyperparameters to display
    
    Interpretation:
        - All points near x=0 → posterior collapse
        - KL very large without recon improvement → KL explosion
        - Scattered distribution → healthy balance
    """
    if isinstance(kl_vals, torch.Tensor):
        kl_vals = kl_vals.detach().cpu().numpy()
    if isinstance(recon_vals, torch.Tensor):
        recon_vals = recon_vals.detach().cpu().numpy()
    
    plt.figure(figsize=(7, 7))
    plt.scatter(kl_vals, recon_vals, s=6, alpha=0.5)
    plt.xlabel("KL Divergence")
    plt.ylabel("Reconstruction Loss")
    plt.title(f"KL vs Reconstruction (Epoch {epoch})")
    plt.axvline(0.05, color="red", linestyle="--", linewidth=2, label='collapse threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add hyperparameters text box
    if hyperparams:
        hp_text = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
        plt.text(0.98, 0.02, hp_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
