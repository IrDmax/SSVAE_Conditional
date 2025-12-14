
# MULTI-MODAL SS VAE CONDITIONED on biological covariates and age group
# ======================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# MLP Block
# ----------------------------------------------------------------------
class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)



class Encoder(nn.Module):
    """
    Architecture:
        - Input: [x_m, batch_emb, label_emb] concatenated
        - Output: (mu, logvar) for the latent distribution q(z | x, batch, label)
    """
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.1):
        """
        Args:
            input_dim: dimension of input = modality_dim + batch_emb_dim + label_emb_dim
            hidden_dims: list of hidden layer dimensions [512, 256, ...]
            latent_dim: dimension of latent space z
            dropout: dropout 
        """
        super().__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims must be a non-empty list")
        
        # Build encoder backbone
        dims = [input_dim] + hidden_dims
        layers = [MLPBlock(dims[i], dims[i+1], dropout) for i in range(len(dims)-1)]
        self.backbone = nn.Sequential(*layers)

        # Separate heads for mean and log-variance
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, input_dim) - concatenated [modality_data, batch_emb, label_emb]

        Returns:
            mu: (B, latent_dim) 
            logvar: (B, latent_dim) 
        """
        h = self.backbone(x)

        mu = self.mu_layer(h)

        # Stabilize variance 
        raw_logvar = self.logvar_layer(h)
        logvar = torch.clamp(raw_logvar, min=-4, max=4)

        return mu, logvar

  
class FusionNet(nn.Module):
    """
    Product of Experts (PoE) fusion for Gaussian posteriors.
    Numerically stable version working in log-precision space.
    
    q(z|x1,x2) ∝ q(z|x1) * q(z|x2)
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, mu1, logvar1, mu2, logvar2):
        # Convert to precision (more stable than variance)
        precision1 = torch.exp(-logvar1)  # 1/var1
        precision2 = torch.exp(-logvar2)  # 1/var2

        # Fused precision (sum of precisions)
        precision_fused = precision1 + precision2

        # Fused variance = 1/precision
        var_fused = 1.0 / (precision_fused + self.eps)  # eps added for numerical stability
        logvar_fused = torch.log(var_fused + self.eps)

        # Fused mean (precision-weighted)
        mu_fused = var_fused * (precision1 * mu1 + precision2 * mu2)

        return mu_fused, logvar_fused


class Decoder(nn.Module):
    """
    Decoder for SSVAE: p(x | z, batch_emb, label_emb, covariate_emb)
    
    Reconstructs observations (RNA or ADT) from:
        - z: latent representation
        - batch_emb: batch embedding
        - label_emb: cell type embedding
        - covariate_emb: biological covariate embeddings
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        """
        Args:
            input_dim: size of concatenated input [z + batch_emb + label_emb + cov_emb]
            hidden_dims: list of hidden layer dimensions
            output_dim: output dimension (rna_dim or adt_dim)
            dropout: dropout probability
        """
        super().__init__()

        # Build decoder backbone
        dims = [input_dim] + hidden_dims
        layers = [MLPBlock(dims[i], dims[i+1], dropout)
                  for i in range(len(dims)-1)]

        self.backbone = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        """
        Args:
            x: (B, input_dim) - concatenated [z, batch_emb, label_emb, covariate_emb]      
        Returns:
            recon: (B, output_dim) - reconstructed observations
        """
        h = self.backbone(x)
        return self.final(h)



class Classifier(nn.Module):
    """
    Semi-supervised classifier for SSVAE:
        - q(y|z): predicts cell type (label) from latent z

    Supports multiple hidden layers via hidden_dims list.
    """

    def __init__(self, latent_dim, n_classes, hidden_dims=None, dropout=0.1):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [latent_dim]  # Default: single hidden layer
        
        # Architecture: latent_dim → hidden[0] → hidden[1] → ... → n_classes
        layers = []
        dims = [latent_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(MLPBlock(dims[i], dims[i+1], dropout))
        
        layers.append(nn.Linear(dims[-1], n_classes))
        
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        """
        Args:
            z: (B, latent_dim) - latent representation
            
        Returns:
            logits: (B, n_classes) - unnormalized class scores
        """
        return self.net(z)


class PriorNetwork(nn.Module):
    """
    p(z | cell_type, covariates) = N(mu, sigma^2)
    
    Computes conditional prior over latent space given:
    - label_emb: (B, label_emb_dim) - pre-computed label embeddings
    - covariate_emb: (B, cov_emb_dim) - pre-computed covariate embeddings (cat + cont)
    
    For samples with zero embeddings (missing info), falls back to standard normal prior.
    """
    def __init__(self, label_emb_dim, cov_emb_dim, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.label_emb_dim = label_emb_dim
        self.cov_emb_dim = cov_emb_dim

        # Combine label and covariate embeddings
        input_dim = label_emb_dim + cov_emb_dim
        # self.shared = nn.Linear(input_dim, 128)
        self.shared = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.GELU(),
                    nn.Dropout(0.1), 
                    nn.Linear(64, 64),
                    nn.GELU(),
                    nn.Dropout(0.1)
        )

        self.mu_layer     = nn.Linear(64, z_dim)
        self.logvar_layer = nn.Linear(64, z_dim)

    def forward(self, label_emb, covariate_emb):
        """
        Args:
            label_emb: (B, label_emb_dim) - label embeddings (zero for unlabeled)
            covariate_emb: (B, cov_emb_dim) - covariate embeddings (zero for missing)

        Returns:
            prior_mu:     (B, z_dim)
            prior_logvar: (B, z_dim)
        """
        B = label_emb.size(0)
        device = label_emb.device

        # Default: standard normal prior 
        prior_mu     = torch.zeros(B, self.z_dim, device=device)
        prior_logvar = torch.zeros(B, self.z_dim, device=device)  # log(1) = 0

        # Check which samples have non-zero embeddings (valid info)
        # A sample is valid if either label or covariate embedding is non-zero
        label_valid = (label_emb.abs().sum(dim=-1) > 1e-6)
        cov_valid = (covariate_emb.abs().sum(dim=-1) > 1e-6)
        valid_mask = label_valid | cov_valid

        if valid_mask.any():
            # Concatenate label and covariate embeddings
            h = torch.cat([label_emb[valid_mask], covariate_emb[valid_mask]], dim=-1)

            h_shared = self.shared(h)   # (N_valid, 64)

            prior_mu_valid     = self.mu_layer(h_shared)      # (N_valid, z_dim)
            prior_logvar_valid = self.logvar_layer(h_shared)  # (N_valid, z_dim)

            # Eestrict variance range --> sharpen prior

            prior_logvar_valid = torch.clamp(prior_logvar_valid, min=-2.5, max=0.5)

            prior_mu[valid_mask]     = prior_mu_valid
            prior_logvar[valid_mask] = prior_logvar_valid

        return prior_mu, prior_logvar


# ============================================================
# 2. Class SSVAE conditioned on covariates
# ============================================================

class SSVAE_Conditioned(nn.Module):

    def __init__(
        self,
        rna_dim, 
        adt_dim,
        latent_dim=32,
        hidden_dims=None,
        num_batches=3,
        batch_emb_dim=8,
        dropout=0.1,
        n_classes=10,
        num_covariates_cat=None,    # dict: {covariate_name: num_unique_values (int)}
        num_covariates_cont=None,   # list: [covariate_names]
        covariate_emb_dim=4,    # embedding dimension for each covariate
        debug=False,            # Enable debug printing
    ):
        super().__init__()

        self.debug = debug

        self.rna_dim = rna_dim
        self.adt_dim = adt_dim
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.batch_emb_dim = batch_emb_dim
        self.covariate_emb_dim = covariate_emb_dim
        
        if hidden_dims is None:
            hidden_dims = [512, 256]


        # Batch embedding
        self.batch_emb = nn.Embedding(num_embeddings=num_batches,
                                 embedding_dim=batch_emb_dim
        )

        # Label embedding
        self.label_emb_dim = 16  
        self.label_emb = nn.Embedding(num_embeddings=n_classes,
                             embedding_dim=self.label_emb_dim
        )

       # Covariate definitions
        self.num_covariates_cat = num_covariates_cat or {}   # dict of {covariate_name: num_unique_values}
        self.num_covariates_cont = num_covariates_cont or []    # list of continuous covariate names

        self.covariate_emb_dim = covariate_emb_dim

         # Covariate embeddings
        self.covariate_cat_embeddings = nn.ModuleDict()
        for cov_name, num_values in self.num_covariates_cat.items():
            self.covariate_cat_embeddings[cov_name] = nn.Embedding(
                num_embeddings=num_values,
                embedding_dim=covariate_emb_dim
            )

        self.covariate_cont_mlps = nn.ModuleDict()
        for cov_name in self.num_covariates_cont:
            self.covariate_cont_mlps[cov_name] = nn.Sequential(
                nn.Linear(1, covariate_emb_dim),
                nn.ReLU(),
                nn.Linear(covariate_emb_dim, covariate_emb_dim)
            )

        # Total covariate embedding size
        self.total_cov_emb_dim = (
            len(self.num_covariates_cat) * covariate_emb_dim 
                + len(self.num_covariates_cont) * covariate_emb_dim
        )
        
        # Encoders for each modality (conditioned on batch + label)
        encoder_rna_dim = rna_dim + batch_emb_dim + self.label_emb_dim
        encoder_adt_dim = adt_dim + batch_emb_dim + self.label_emb_dim

        self.encoder_rna = Encoder(
                 encoder_rna_dim, hidden_dims, latent_dim, dropout
        )
        
        self.encoder_adt = Encoder(
            encoder_adt_dim, hidden_dims, latent_dim, dropout
        )
       
        # Fusion network
        self.fusion_net = FusionNet(eps=1e-8)
        
        # Classifier q(y|z) for cell type (q(y|z))
        self.classifier = Classifier(latent_dim, n_classes,
                         hidden_dims=None, dropout=dropout)
        
        # Classifier heads for categorical covariates q(c_i|z) 
        self.covariate_classifiers = nn.ModuleDict()
        for cov_name, num_values in self.num_covariates_cat.items():
            self.covariate_classifiers[cov_name] = Classifier(latent_dim, num_values)

        # Encode + sample z WITHOUT decoding --> for LSTM module
        # def encode_to_z(self, x):
        #     mu, logvar = self.encode(x)
        #     z = self.reparameterize(mu, logvar)
        #     return z, mu, logvar

        # Regression heads for continuous covariates
        self.covariate_cont_regressors = nn.ModuleDict()
        for cov_name in self.num_covariates_cont:
            self.covariate_cont_regressors[cov_name] = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, 1)        # predict scalar score
            )


        # Decoders (conditioned on z + batch + label_emb + covariates)
        decoder_input_dim = latent_dim + self.batch_emb_dim + self.label_emb_dim + self.total_cov_emb_dim
       
        self.decoder_rna = Decoder(
                 decoder_input_dim, hidden_dims[::-1], rna_dim, dropout
        )
        self.decoder_adt = Decoder(
                decoder_input_dim, hidden_dims[::-1], adt_dim, dropout
        ) 

        # Conditional prior network
        # Always create it - will use embedding dimensions instead of discrete counts
        self.prior_net = PriorNetwork(
            label_emb_dim=self.label_emb_dim,
            cov_emb_dim=self.total_cov_emb_dim,
            z_dim=latent_dim,
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    # Forward pass
    def forward(self, rna, adt, batch_idx, labels=None, covariates_cat=None, covariates_cont=None):
        """
        Args:
            rna: (B, rna_dim)
            adt: (B, adt_dim)
            batch_idx: (B,) - integer batch indices
            labels: (B,) - integer labels (None for unlabeled data)
            covariate_indices: dict of {covariate_name: (B,) tensor of indices}
        
        Returns:
            dict with reconstructions, latents, and logits
        """

        batch_size = rna.size(0)

        if self.debug:
            print(f"\n[Forward Pass] Batch size: {batch_size}")

        # 1. Get batch embedding 
        batch_emb = self.batch_emb(batch_idx)  # (B, batch_emb_dim)

        # 2. Get covariate embeddings
        #    - categorical: -1 → zero vector
        #    - continuous:  NaN → zero vector
      
        cat_embs = []
        cont_embs = []

        # Categorical covariates embeddings
        if covariates_cat is not None and len(self.num_covariates_cat) > 0:
            for cov_name in sorted(self.num_covariates_cat.keys()):
                if cov_name not in covariates_cat:
                    continue  # skip if this covariate is not present in the batch

                cov_values = covariates_cat[cov_name]            # (B,)
                D = self.covariate_cat_embeddings[cov_name].embedding_dim

                # Per-cell mask: valid indices are >= 0, missing are -1
                valid_mask = cov_values >= 0

                # Allocate full embedding matrix (zero for all cells initially)
                cov_emb = torch.zeros(batch_size, D, device=rna.device)

                # Fill only valid entries with embedding lookup
                if valid_mask.any():
                    cov_emb[valid_mask] = self.covariate_cat_embeddings[cov_name](
                        cov_values[valid_mask]
                    )

                # Missing entries remain zero vectors
                cat_embs.append(cov_emb)

        # Continuous covariates embeddings
        if covariates_cont is not None and len(self.num_covariates_cont) > 0:
            for cov_name in self.num_covariates_cont:
                if cov_name not in covariates_cont:
                    continue  # skip if this covariate is not present in the batch

                cont_values = covariates_cont[cov_name].to(rna.device)   # (B,) float
                D = self.covariate_emb_dim                           # MLP output dim

                # Mask of non-missing values (NaN = missing)
                valid_mask = ~torch.isnan(cont_values)

                # Allocate full embedding matrix (zero for all cells initially)
                cont_emb = torch.zeros(batch_size, D, device=rna.device)

                # Compute embeddings only for non-missing values
                if valid_mask.any():
                    cont_emb[valid_mask] = self.covariate_cont_mlps[cov_name](
                        cont_values[valid_mask].unsqueeze(-1)       # shape (N_valid, 1)
                    )

                # Missing entries remain zero vectors
                cont_embs.append(cont_emb)

        # Concatenate all covariate embeddings
        if cat_embs or cont_embs:
            covariate_emb = torch.cat(cat_embs + cont_embs, dim=-1)
        else:
            covariate_emb = torch.zeros(batch_size, self.total_cov_emb_dim, device=rna.device)

        ### 3. ENCODER  ###
        # LABEL-AGNOSTIC PATH (for classifier q(y|x))
        # NO LABEL INFO fed to encoder

        label_emb_cls = torch.zeros(batch_size, self.label_emb_dim, device=rna.device)

        rna_in_cls = torch.cat([rna, batch_emb, label_emb_cls], dim=-1)
        adt_in_cls = torch.cat([adt, batch_emb, label_emb_cls], dim=-1)

        # Encode each modality for classifier path
        mu_rna_cls, logvar_rna_cls = self.encoder_rna(rna_in_cls)
        mu_adt_cls, logvar_adt_cls = self.encoder_adt(adt_in_cls)

        mu_cls_fused, logvar_cls_fused = self.fusion_net(
            mu_rna_cls, logvar_rna_cls,
            mu_adt_cls, logvar_adt_cls
        )
        z_cls_fused = self.reparameterize(mu_cls_fused, logvar_cls_fused)

        # Classifier (q(y|x))
        label_logits = self.classifier(z_cls_fused)
        q_y = torch.softmax(label_logits, dim=-1)

   
        ### 3. LABEL-AWARE PATH (for generative model) ###
    
        label_emb_gen = torch.zeros(batch_size, self.label_emb_dim, device=rna.device)

        if labels is not None:
            labeled_mask = labels >= 0
            if labeled_mask.any():
                label_emb_gen[labeled_mask] = self.label_emb(labels[labeled_mask])

        rna_in_gen = torch.cat([rna, batch_emb, label_emb_gen], dim=-1)
        adt_in_gen = torch.cat([adt, batch_emb, label_emb_gen], dim=-1)

        # Encode each modality for generative path
        mu_rna_gen, logvar_rna_gen = self.encoder_rna(rna_in_gen)
        mu_adt_gen, logvar_adt_gen = self.encoder_adt(adt_in_gen)

        mu_gen_fused, logvar_gen_fused = self.fusion_net(
            mu_rna_gen, logvar_rna_gen,
            mu_adt_gen, logvar_adt_gen
        )
        z_gen_fused = self.reparameterize(mu_gen_fused, logvar_gen_fused)

 
        ### 4. DECODER LABEL INPUT ###
        # True labels where available, inferred labels otherwise
        label_emb_dec = torch.zeros(batch_size, self.label_emb_dim, device=rna.device)

        if labels is None:
            # all unlabeled → inferred labels
            label_emb_dec = q_y @ self.label_emb.weight
        else:
            labeled_mask = labels >= 0
            unlabeled_mask = labels < 0

            if labeled_mask.any():
                label_emb_dec[labeled_mask] = self.label_emb(labels[labeled_mask])

            if unlabeled_mask.any():
                label_emb_dec[unlabeled_mask] = (
                    q_y[unlabeled_mask] @ self.label_emb.weight
                )

        ### 5. DECODE (p(x|z,y)) ###
    
        decoder_input = torch.cat(
            [z_gen_fused, batch_emb, label_emb_dec, covariate_emb],
            dim=-1
        )

        rna_recon = self.decoder_rna(decoder_input)
        adt_recon = self.decoder_adt(decoder_input)

        ### 6. Covariate predictions from z
        covariate_cat_logits = {}
        covariate_cont_pred = {}

        for cov_name, num_classes in self.num_covariates_cat.items():
            covariate_cat_logits[cov_name] = self.covariate_classifiers[cov_name](z_gen_fused)

        for cov_name in self.num_covariates_cont:
            covariate_cont_pred[cov_name] = self.covariate_cont_regressors[cov_name](z_gen_fused)


        ### 7. Prior: p(z | y, covariates) for ALL classes (for proper marginalization)
        # Always compute prior for all classes in a vectorized way
        # Create label embeddings for all classes: (n_classes, label_emb_dim)

        class_indices = torch.arange(self.n_classes, device=rna.device)
        class_label_embs = self.label_emb(class_indices)  # (n_classes, label_emb_dim)

        # Expand to (batch_size, n_classes, label_emb_dim)
        class_label_embs = class_label_embs.unsqueeze(0).expand(batch_size, -1, -1)
        # Expand covariate_emb to (batch_size, n_classes, cov_emb_dim)
        covariate_emb_exp = covariate_emb.unsqueeze(1).expand(-1, self.n_classes, -1)

        # Flatten for batch processing
        flat_class_label_embs = class_label_embs.reshape(-1, class_label_embs.size(-1))
        flat_covariate_emb = covariate_emb_exp.reshape(-1, covariate_emb_exp.size(-1))

        # Compute prior for ALL classes for each sample in batch 
        # -> required for KL marginalization (for unlabeled data)
        mu_p_all, logvar_p_all = self.prior_net(flat_class_label_embs, flat_covariate_emb)
        mu_p_all_classes = mu_p_all.view(batch_size, self.n_classes, -1)
        logvar_p_all_classes = logvar_p_all.view(batch_size, self.n_classes, -1)
        
        # Compute the prior that corresponds to the actual labels (input for decoder -> uses soft labels)
        mu_p, logvar_p = self.prior_net(label_emb_dec, covariate_emb)

        ### 8. Diagnostics: KL divergence between q(z|x,y) and p(z|y,covariates) ###    
        with torch.no_grad():
            # Generative posterior q(z | x, y)
            var_q = torch.exp(logvar_gen_fused)

            # Conditional prior p(z | y, covariates)
            var_p = torch.exp(logvar_p)

            kl_per_dim = (
                logvar_p - logvar_gen_fused +
                (var_q + (mu_gen_fused - mu_p) ** 2) / var_p -
                1.0
            )

            kl_values = 0.5 * kl_per_dim.sum(dim=-1)   # shape: (B,)


        # return outputs
        outputs = {

            # Reconstructions
            'rna_recon': rna_recon,
            'adt_recon': adt_recon,

            # Latents
            # Generative latent q(z | x, y)
            'mu_gen_fused': mu_gen_fused,
            'logvar_gen_fused': logvar_gen_fused,
            'z_gen_fused': z_gen_fused,

            # Classifier latent q(z | x)
            #'z_cls': z_cls,

            # Classifier outputs
            'label_logits': label_logits,      # q(y | x)
            #'q_y': q_y,                        # softmax(label_logits)

            # Label embedding actually used by decoder / prior
            'label_emb_dec': label_emb_dec,

            # Covariate predictions (auxiliary heads)
            'covariate_cat_logits': covariate_cat_logits,
            'covariate_cont_pred': covariate_cont_pred,
            'covariate_emb': covariate_emb,

            # Prior parameters
            # Prior matching decoder conditioning
            'mu_p': mu_p,
            'logvar_p': logvar_p,

            # Prior for all classes (used for KL marginalization)
            'mu_p_all_classes': mu_p_all_classes,
            'logvar_p_all_classes': logvar_p_all_classes,

            # Diagnostics
            'kl_values': kl_values
        }

        if self.debug:
            outputs['z_cls_fused'] = z_cls_fused
            outputs['q_y'] = q_y

            outputs['mu_rna_cls'] = mu_rna_cls
            outputs['logvar_rna_cls'] = logvar_rna_cls
            outputs['mu_adt_cls'] = mu_adt_cls
            outputs['logvar_adt_cls'] = logvar_adt_cls
            
            outputs['mu_cls_fused'] = mu_cls_fused
            outputs['logvar_cls_fused'] = logvar_cls_fused

        return outputs


###### Semi-supervised ELBO with Cell Type and Covariate Prediction ######

def compute_loss(
    model, 
    rna, 
    adt, 
    batch_idx, 
    labels=None,                # (B,) cell type labels, -1 for unlabeled
    covariates_cat=None,        # dict: {name: (B,) long, -1 = missing}
    covariates_cont=None,       # dict: {name: (B,) float, NaN = missing}
    alpha_recon=1.0,
    beta=0.1,
    alpha_cov_cat=1.0,
    alpha_cov_cont=1.0,
    alpha_class=2.0,
    gamma=1.0,                  # entropy weight
    lambda_adt=None
):
    """
    Label handling:
      - Labeled samples (labels >= 0): Cross-entropy loss on predicted labels
      - Unlabeled samples (labels == -1 or None): Entropy regularization on predictions
    
    Covariate prediction from latent z:
      - Categorical covariates: Classification loss (cross-entropy) + entropy regularization for missing
      - Continuous covariates: Regression loss (MSE) for observed values only
    """

    # Forward pass
  
    outputs = model(
        rna,
        adt,
        batch_idx,
        labels=labels,
        covariates_cat=covariates_cat,
        covariates_cont=covariates_cont,
    )

    rna_recon = outputs["rna_recon"]
    adt_recon = outputs["adt_recon"]

    mu_q = outputs["mu_gen_fused"]
    logvar_q = outputs["logvar_gen_fused"]

    label_logits = outputs["label_logits"]
    cov_cat_logits = outputs["covariate_cat_logits"]
    cov_cont_pred = outputs["covariate_cont_pred"]

 
    # 1. Reconstruction loss per-sample sums
    recon_rna = F.mse_loss(rna_recon, rna, reduction="none").sum(dim=-1)
    recon_adt = F.mse_loss(adt_recon, adt, reduction="none").sum(dim=-1)

    # Set default lambda_adt if not provided
    if lambda_adt is None:
        lambda_adt = model.rna_dim / model.adt_dim

    recon_loss = recon_rna.mean() + lambda_adt * recon_adt.mean()

    
    # Compute per-feature losses for adaptive weighting
    rna_per_feature = recon_rna.mean() / model.rna_dim
    adt_per_feature = recon_adt.mean() / model.adt_dim

    # 2. KL divergence with proper marginalization
    mu_p_all_classes = outputs["mu_p_all_classes"]
    logvar_p_all_classes = outputs["logvar_p_all_classes"]
    label_logits = outputs["label_logits"]

    if labels is not None:
        labeled_mask = (labels >= 0)
        unlabeled_mask = (labels == -1)
    else:
        labeled_mask = torch.zeros(rna.size(0), device=rna.device, dtype=torch.bool)
        unlabeled_mask = torch.ones(rna.size(0), device=rna.device, dtype=torch.bool)

    # Compute per-sample KL for all samples
    kl_per_sample = torch.zeros(rna.size(0), device=rna.device)

    # For labeled cells: standard KL with true label prior
    if labeled_mask.any():
        # Index into mu_p_all_classes and logvar_p_all_classes using true labels
        # mu_p_all_classes[labeled_mask]: (N_labeled, n_classes, z_dim)
        # labels[labeled_mask]: (N_labeled,)
        # Need to gather along class dimension
        
        batch_indices = torch.arange(labeled_mask.sum(), device=rna.device)
        label_indices = labels[labeled_mask]
        
        # Shape: (N_labeled, z_dim)
        mu_p_labeled = mu_p_all_classes[labeled_mask][batch_indices, label_indices]
        logvar_p_labeled = logvar_p_all_classes[labeled_mask][batch_indices, label_indices]
        
        var_q_labeled = torch.exp(logvar_q[labeled_mask])
        var_p_labeled = torch.exp(logvar_p_labeled)
        
        kl_labeled = 0.5 * (
            logvar_p_labeled - logvar_q[labeled_mask] +
            (var_q_labeled + (mu_q[labeled_mask] - mu_p_labeled) ** 2) / var_p_labeled -
            1.0
        ).sum(dim=-1)  # (N_labeled,) - per-sample KL
        
        kl_per_sample[labeled_mask] = kl_labeled

    # For unlabeled cells: marginalize over q(y|z)
    if unlabeled_mask.any():
        # Get q(y|z) probabilities
        q_y = F.softmax(label_logits[unlabeled_mask], dim=-1)  # (N_unlabeled, n_classes)
        
        # Expand dimensions for broadcasting
        mu_q_unlabeled = mu_q[unlabeled_mask].unsqueeze(1)  # (N_unlabeled, 1, z_dim)
        logvar_q_unlabeled = logvar_q[unlabeled_mask].unsqueeze(1)  # (N_unlabeled, 1, z_dim)
        
        mu_p_unlabeled = mu_p_all_classes[unlabeled_mask]  # (N_unlabeled, n_classes, z_dim)
        logvar_p_unlabeled = logvar_p_all_classes[unlabeled_mask]  # (N_unlabeled, n_classes, z_dim)
        
        var_q_unlabeled = torch.exp(logvar_q_unlabeled)
        var_p_unlabeled = torch.exp(logvar_p_unlabeled)
        
        # Compute KL for each class: (N_unlabeled, n_classes, z_dim) -> (N_unlabeled, n_classes)
        kl_per_class = 0.5 * (
            logvar_p_unlabeled - logvar_q_unlabeled +
            (var_q_unlabeled + (mu_q_unlabeled - mu_p_unlabeled) ** 2) / var_p_unlabeled -
            1.0
        ).sum(dim=-1)  # (N_unlabeled, n_classes)
        
        # Marginalize: sum over classes weighted by q(y|z)
        kl_unlabeled = (q_y * kl_per_class).sum(dim=-1)  # (N_unlabeled,) - per-sample KL
        
        kl_per_sample[unlabeled_mask] = kl_unlabeled

    # Final KL loss: simple mean over all samples (no double normalization!)
    kl_loss = kl_per_sample.mean()


    # 3. Cell-type classification (semi-supervised)

    class_ce = torch.tensor(0.0, device=rna.device)
    class_entropy = torch.tensor(0.0, device=rna.device)

    if labels is not None:
        labeled_mask = (labels >= 0)
        unlabeled_mask = (labels == -1)
    else:
        labeled_mask = torch.zeros(rna.size(0), device=rna.device, dtype=torch.bool)
        unlabeled_mask = torch.ones(rna.size(0), device=rna.device, dtype=torch.bool)

    # Supervised classification loss (CE Loss) CE = −log p_y_true 
    if labeled_mask.any():
        class_ce = F.cross_entropy(
            label_logits[labeled_mask],
            labels[labeled_mask],
            reduction="mean"
        )

    # Unsupervised classification entropy regularization H = −∑_c p_c logp_c 
    if unlabeled_mask.any():
        probs = F.softmax(label_logits[unlabeled_mask], dim=-1)
        log_probs = F.log_softmax(label_logits[unlabeled_mask], dim=-1)
        class_entropy = -(probs * log_probs).sum(dim=-1).mean()


    # 4. Categorical covariates: CE + entropy for missing

    cov_cat_loss = torch.tensor(0.0, device=rna.device)
    cov_cat_entropy = torch.tensor(0.0, device=rna.device)

    if covariates_cat is not None:
        ce_vals = []
        ent_vals = []

        for name, cov_true in covariates_cat.items():
            # Skip if model doesn't predict this covariate
            if name not in cov_cat_logits:
                continue
            logits = cov_cat_logits[name]

            valid_mask = (cov_true >= 0)
            missing_mask = (cov_true == -1)

            # CE for observed covariates
            if valid_mask.any():
                ce_vals.append(
                    F.cross_entropy(
                        logits[valid_mask],
                        cov_true[valid_mask],
                        reduction="mean"
                    )
                )

            # Entropy for missing categories (semi-supervised covariate)
            if missing_mask.any():
                probs = F.softmax(logits[missing_mask], dim=-1)
                log_probs = F.log_softmax(logits[missing_mask], dim=-1)
                ent_vals.append(-(probs * log_probs).sum(dim=-1).mean())

        if ce_vals:
            cov_cat_loss = torch.stack(ce_vals).mean()
        if ent_vals:
            cov_cat_entropy = torch.stack(ent_vals).mean()


    # 5. Continuous covariates: regression (MSE loss for non-missing values)
 
    cov_cont_loss = torch.tensor(0.0, device=rna.device)

    if covariates_cont is not None:
        vals = []
        for name, cov_true in covariates_cont.items():
            # Skip if model doesn't predict this covariate
            if name not in cov_cont_pred:
                continue
            pred = cov_cont_pred[name]   # shape (B,1)
            mask = ~torch.isnan(cov_true)

            if mask.any():
                vals.append(
                    F.mse_loss(pred[mask], cov_true[mask].unsqueeze(-1), reduction="mean")
                )

        if vals:
            cov_cont_loss = torch.stack(vals).mean()

 
    # 6. Total loss

    total_loss = (
        alpha_recon * recon_loss +
        beta * kl_loss +
        alpha_class * class_ce +
        alpha_cov_cat * cov_cat_loss +
        alpha_cov_cont * cov_cont_loss -
        gamma * (class_entropy + cov_cat_entropy)   # subtract entropy to encourage confidence
    )

    loss_dict = {
        "total": total_loss.item(),
        "recon": recon_loss.item(),
        "recon_rna": recon_rna.mean().item(),
        "recon_adt": recon_adt.mean().item(),
        "kl": kl_loss.item(),
        "class_ce": class_ce.item(),
        "class_entropy": class_entropy.item(),
        "cov_cat_ce": cov_cat_loss.item(),
        "cov_cat_entropy": cov_cat_entropy.item(),
        "cov_cont_mse": cov_cont_loss.item(),
        "rna_pf": rna_per_feature.item(),
        "adt_pf": adt_per_feature.item(),
    }

    return total_loss, loss_dict, outputs


def encode_sequence_with_vae(vae, x):
    """
    x: (batch, time_steps, input_dim)
    returns: z_sequence (batch, time_steps, latent_dim)
    """

    batch, time_steps, _ = x.shape
    z_list = []

    for t in range(time_steps):
        x_t = x[:, t, :]              # (batch, input_dim)
        vae_out = vae(x_t)            # forward pass through VAE encoder+decoder
        z_t = vae_out["z_fused"]      # (batch, latent_dim)
        z_list.append(z_t)

    
    z_sequence = torch.stack(z_list, dim=1)
    return z_sequence



















    