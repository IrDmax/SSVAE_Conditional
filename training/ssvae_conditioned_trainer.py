import torch
from torch.optim import AdamW
from models.ssvae_conditioned import compute_loss

# Adaptive weighting scheduler for ADT reconstruction loss
class ADTWeightScheduler:
    def __init__(self, alpha=0.05, init_lambda=1.0, min_lambda=0.01, max_lambda=100.0):
        self.alpha = alpha
        self.lambda_adt = init_lambda
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

    def update(self, rna_pf, adt_pf):
        if adt_pf < 1e-9:    # avoid divide by zero
            return self.lambda_adt
        
        target = rna_pf / adt_pf

        # EMA smoothing
        self.lambda_adt = (
            (1 - self.alpha) * self.lambda_adt +
            self.alpha * target
        )

        # clamp to avoid instability
        self.lambda_adt = float(
            max(self.min_lambda, min(self.lambda_adt, self.max_lambda))
        )

        return self.lambda_adt

def print_hyperparams(
    num_epochs,
    lr,
    weight_decay,
    label_fraction,
    batch_size,
    subset_size,
    model=None
):
    print("TRAINING HYPERPARAMETERS:")

    print(f"- Num epochs:              {num_epochs}")
    print(f"- Learning rate:           {lr}")
    print(f"- Weight decay:            {weight_decay}")
    print(f"- Label fraction:          {label_fraction}")
    print(f"- Batch size:              {batch_size}")
    print(f"- Subset size:             {subset_size}")


    print(f"  - Number of categorical covariates:  {len(model.num_covariates_cat) if hasattr(model, 'num_covariates_cat') else 'N/A'}")
    print(f"  - Number of continuous covariates:   {len(model.num_covariates_cont) if hasattr(model, 'num_covariates_cont') else 'N/A'}")

    print("="*70 + "\n")

# beta_max=0.3 (was 1.0)  # try 0.3, if still too strong drop to 0.2
def get_beta(epoch, beta_start=0.01, beta_max=2.0, warmup_epochs=50):
    t = min(epoch / warmup_epochs, 1.0)         # goes 0 → 1
    return beta_start + t * (beta_max - beta_start)


def train_ssvae_conditioned(
    model,
    train_loader,
    val_loader,
    num_epochs,
    device,
    lr=1e-4,
    weight_decay=1e-2,
    label_fraction=1.0,
    checkpoint_path="best_model.pt",
    patience=10
):
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ADTWeightScheduler(alpha=0.05, init_lambda=1.0)

    # -------------------------------
    # PRINT HYPERPARAMETERS
    # -------------------------------
    print_hyperparams(
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        label_fraction=label_fraction,
        batch_size=train_loader.batch_size,
        subset_size=len(train_loader.dataset),
        model=model
    )

    # with open("training_hparams.txt", "w") as f:
    #     for k, v in hparams.items():
    #         f.write(f"{k}: {v}\n")

    lambda_adt = 1.0
    best_val_acc = -1
    patience_counter = 0

    history = {
        "train_loss": [], "val_loss": [],
        "train_loss_dict": [], "val_loss_dict": [],
        "train_acc": [], "val_acc": [],
        "train_labeled": [], "train_unlabeled": [],
        "train_cov_cat_ce": [], "train_cov_cat_entropy": [],
        "val_cov_cat_ce": [], "val_cov_cat_entropy": []
    }


    # def sigmoid(x):
    #     return 1 / (1 + math.exp(-x))

    for epoch in range(1, num_epochs + 1):

        # ---- KL warmup schedule ----
        # beta = 0.01 + 0.99 * sigmoid(0.1 * (epoch - 20))
        # or simply: beta = min(1.0, epoch / 50)   # linear warmup example

        beta = get_beta(epoch)
 
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            beta=beta,
            lambda_adt=lambda_adt,
            label_fraction=label_fraction
        )

        val_metrics = validate_epoch(
            model, val_loader, device, 
            lambda_adt=lambda_adt
        )

        # Update λ_adt based on average per-feature scale
        rna_pf_val = train_metrics["rna_pf"]
        adt_pf_val = train_metrics["adt_pf"]
        new_lambda = scheduler.update(rna_pf_val, adt_pf_val)
        # new_lambda = 1.0 # temporary fix to disable adaptive weighting

        # Logging
        if(epoch % 1 == 0):
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss {train_metrics['loss']:.4f} | Val Loss {val_metrics['loss']:.4f} | "
                f"Train Acc {train_metrics['acc']:.4f} | Val Acc {val_metrics['acc']:.4f} | "
                f"Labeled {train_metrics['labeled']} | Unlabeled {train_metrics['unlabeled']} | "
                f"rna_pf={rna_pf_val:.4f} adt_pf={adt_pf_val:.4f} | "
                f"λ_adt: {lambda_adt:.3f} → {new_lambda:.3f}"
            )
        lambda_adt = new_lambda

        # Save history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_loss_dict"].append(train_metrics["loss_dict"])
        history["val_loss_dict"].append(val_metrics["loss_dict"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["train_labeled"].append(train_metrics["labeled"])
        history["train_unlabeled"].append(train_metrics["unlabeled"])
        history["train_cov_cat_ce"].append(train_metrics["loss_dict"]["cov_cat_ce"])
        history["train_cov_cat_entropy"].append(train_metrics["loss_dict"]["cov_cat_entropy"])
        history["val_cov_cat_ce"].append(val_metrics["loss_dict"]["cov_cat_ce"])
        history["val_cov_cat_entropy"].append(val_metrics["loss_dict"]["cov_cat_entropy"])

        # Early stopping: Max validation accuracy
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  - Saved new best model (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training finished. Best validation accuracy = {best_val_acc:.4f}")
    return history



def train_epoch(model, loader, optimizer, device, beta, lambda_adt, label_fraction=1.0):
    model.train()

    total_loss = 0
    correct = 0
    total = 0
    labeled_count = 0
    unlabeled_count = 0

    # Accumulate loss components
    component_sums = {}
    batch_count = 0

    print(f"  Starting train_epoch, processing {len(loader)} batches...", flush=True)
    for batch_num, batch in enumerate(loader):
        if batch_num == 0:
            print(f"    [Batch 0] Loading data to device...", flush=True)
        
        rna = batch['rna'].to(device)
        adt = batch['adt'].to(device)
        batch_idx = batch['batch'].to(device)

        labels = batch.get('label', None)
        if labels is not None:
            labels = labels.to(device)

        covariates_cat = batch.get('covariates_cat', None)
        if covariates_cat is not None:
            covariates_cat = {k: v.to(device) for k, v in covariates_cat.items()}

        covariates_cont = batch.get('covariates_cont', None)
        if covariates_cont is not None:
            covariates_cont = {k: v.to(device) for k, v in covariates_cont.items()}

        # -------- semi-supervised masking --------
        if labels is not None and label_fraction < 1.0:
            mask = torch.rand_like(labels.float()) < label_fraction
            masked_labels = labels.clone()
            masked_labels[~mask] = -1

            labeled_count += mask.sum().item()
            unlabeled_count += (~mask).sum().item()

            labels_used = masked_labels
        else:
            labels_used = labels
            if labels is not None:
                labeled_count += labels.numel()

        # -------- compute loss --------
        loss, loss_dict = compute_loss(
            model, rna, adt, batch_idx,
            labels=labels_used,
            covariates_cat=covariates_cat,
            covariates_cont=covariates_cont,
            beta=beta,
            lambda_adt=lambda_adt             
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1
        
        # Progress indicator every 10 batches
        if batch_num % 20 == 0:
            print(f"    Batch {batch_num}/{len(loader)}, Loss: {loss.item():.4f}", flush=True)

        # Accumulate individual components (detach to avoid keeping computation graph)
        for k, v in loss_dict.items():
            component_sums[k] = component_sums.get(k, 0.0) + (v.item() if torch.is_tensor(v) else v)

        # -------- compute accuracy using TRUE labels (not masked) --------
        if labels is not None:
            preds = model(rna, adt, batch_idx, labels, covariates_cat, covariates_cont)['label_logits'].argmax(1)
            mask = (labels >= 0)
            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()

    acc = correct / total if total > 0 else 0.0

    # Average component losses
    avg_components = {k: v / batch_count for k, v in component_sums.items()}

    rna_pf = component_sums.get("rna_pf", 0.0) / batch_count
    adt_pf = component_sums.get("adt_pf", 0.0) / batch_count

    return {
        "loss": total_loss / batch_count,
        "loss_dict": avg_components,
        "acc": acc,
        "labeled": labeled_count,
        "unlabeled": unlabeled_count,
        "rna_pf": rna_pf,
        "adt_pf": adt_pf,
    }

def validate_epoch(model, loader, device, lambda_adt=None):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    # accumulate loss components
    component_sums = {}
    batch_count = 0

    with torch.no_grad():
        for batch in loader:
            rna = batch['rna'].to(device)
            adt = batch['adt'].to(device)
            batch_idx = batch['batch'].to(device)

            labels = batch.get('label', None)
            if labels is not None:
                labels = labels.to(device)

            covariates_cat = batch.get('covariates_cat', None)
            if covariates_cat is not None:
                covariates_cat = {k: v.to(device) for k, v in covariates_cat.items()}

            covariates_cont = batch.get('covariates_cont', None)
            if covariates_cont is not None:
                covariates_cont = {k: v.to(device) for k, v in covariates_cont.items()}

            loss, loss_dict = compute_loss(
                model, rna, adt, batch_idx,
                labels=labels,
                covariates_cat=covariates_cat,
                covariates_cont=covariates_cont,
                lambda_adt=lambda_adt
            )

            total_loss += loss.item()
            batch_count += 1

            # accumulate component losses
            for k, v in loss_dict.items():
                component_sums[k] = component_sums.get(k, 0.0) + v

            # accuracy
            if labels is not None:
                preds = model(rna, adt, batch_idx, labels, covariates_cat, covariates_cont)["label_logits"].argmax(1)
                mask = (labels >= 0)
                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()

    acc = correct / total if total > 0 else 0.0

    # average component losses
    avg_components = {k: v / batch_count for k, v in component_sums.items()}

    return {
        "loss": total_loss / batch_count,
        "acc": acc,
        "loss_dict": avg_components 
    }
