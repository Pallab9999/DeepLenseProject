"""
models/adda.py
==============
Adversarial Discriminative Domain Adaptation (ADDA) for transferring the
physics-informed LensPINN encoder from simulated to real HSC images.

Reference: Tzeng et al. (2017)  https://arxiv.org/abs/1702.05464
Applied to DeepLense: Alexander et al. (2023) https://arxiv.org/abs/2112.12121

This implementation:
  ─ Source encoder + classifier are trained in Stage 1 (see train.py)
  ─ Stage 2 (this file) freezes the source encoder + classifier, clones the
    encoder as the *target encoder*, then adversarially aligns target features
    to match source features.
  ─ The domain discriminator tells apart source / target encodings.
  ─ The target encoder is updated to fool the discriminator.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Domain Discriminator
# ---------------------------------------------------------------------------

class DomainDiscriminator(nn.Module):
    """
    Binary classifier: 0 = source domain (simulated), 1 = target domain (real).

    Parameters
    ----------
    feature_dim : int
        Dimensionality of the encoder output (must match both source & target
        encoder output sizes).
    hidden_dim  : int
        Width of the two hidden layers.
    """

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2),   # 2-class for CrossEntropyLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# ADDA Trainer
# ---------------------------------------------------------------------------

class ADDATrainer:
    """
    Manages the two-phase ADDA training loop.

    Usage
    -----
    Stage 1 — train source model on simulated data (done in train.py)
    Stage 2 — call `adapt()` to align target encoder to source domain
    Stage 3 — evaluate target encoder + frozen classifier on target test set

    Parameters
    ----------
    source_encoder  : pre-trained encoder (e.g. LensPINN ViT part)
    target_encoder  : freshly-cloned copy of source_encoder to be adapted
    classifier      : pre-trained classification head (frozen during adaptation)
    discriminator   : DomainDiscriminator instance
    device          : torch device
    """

    def __init__(
        self,
        source_encoder: nn.Module,
        target_encoder: nn.Module,
        classifier: nn.Module,
        discriminator: DomainDiscriminator,
        device: torch.device,
    ):
        self.source_encoder = source_encoder.to(device)
        self.target_encoder = target_encoder.to(device)
        self.classifier     = classifier.to(device)
        self.discriminator  = discriminator.to(device)
        self.device         = device

        # Freeze source encoder and classifier
        for p in self.source_encoder.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = False

        self.history: Dict[str, List[float]] = {
            "loss_discriminator": [],
            "loss_target":        [],
            "accuracy":           [],
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        source_encoder: nn.Module,
        classifier: nn.Module,
        feature_dim: int = 512,
        device: Optional[torch.device] = None,
    ) -> "ADDATrainer":
        """
        Convenience constructor: load a pre-trained checkpoint and clone the
        encoder for adaptation.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(checkpoint_path, map_location=device)
        source_encoder.load_state_dict(ckpt["encoder_state_dict"])
        classifier.load_state_dict(ckpt["classifier_state_dict"])

        target_encoder = copy.deepcopy(source_encoder)
        discriminator  = DomainDiscriminator(feature_dim=feature_dim)

        return cls(source_encoder, target_encoder, classifier, discriminator, device)

    # ------------------------------------------------------------------
    def _encode_source(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using the (frozen) source encoder."""
        out = self.source_encoder(x)
        if isinstance(out, (tuple, list)):
            # LensPINN returns (theta_E, source, logits); take logits as proxy
            # or we can grab an intermediate representation — here we use the
            # penultimate feature before the final linear head.
            # For flexibility we return the last element (logits work for
            # discriminator training, or override this method).
            out = out[-1]
        return out

    def _encode_target(self, x: torch.Tensor) -> torch.Tensor:
        out = self.target_encoder(x)
        if isinstance(out, (tuple, list)):
            out = out[-1]
        return out

    def adapt(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        target_loader_val: DataLoader,
        epochs: int = 50,
        lr_target: float = 1e-5,
        lr_disc: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 15,
        save_path: str | Path = "results/checkpoints/adda_best.pt",
        limit_batches: Optional[int] = None,
    ) -> None:
        """
        Run the ADDA adversarial adaptation loop.

        Parameters
        ----------
        source_loader      : DataLoader for labelled simulated data (labels ignored)
        target_loader      : DataLoader for unlabelled target data (HSC / Model IV)
        target_loader_val  : DataLoader for evaluating adaptation progress
        epochs             : number of adaptation epochs
        lr_target          : learning rate for target encoder
        lr_disc            : learning rate for domain discriminator
        weight_decay       : L2 regularisation
        patience           : early stopping patience (val accuracy)
        save_path          : where to save the best target encoder + classifier
        limit_batches      : limit number of batches per epoch (for fast training/debugging)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        criterion = nn.CrossEntropyLoss()

        opt_target = torch.optim.Adam(
            self.target_encoder.parameters(), lr=lr_target, weight_decay=weight_decay
        )
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_disc, weight_decay=weight_decay
        )

        best_val_acc = 0.0
        bad_epochs   = 0

        self.source_encoder.eval()
        self.classifier.eval()

        iters = max(len(source_loader), len(target_loader))
        if limit_batches is not None:
            iters = min(iters, limit_batches)

        for epoch in range(epochs):
            self.discriminator.train()
            self.target_encoder.train()

            running_loss_disc   = 0.0
            running_loss_target = 0.0

            src_iter = iter(source_loader)
            tgt_iter = iter(target_loader)

            for _ in range(iters):
                # ── Sample batches ─────────────────────────────────────────
                try:
                    x_src, _ = next(src_iter)
                except StopIteration:
                    src_iter = iter(source_loader)
                    x_src, _ = next(src_iter)

                try:
                    x_tgt, _ = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(target_loader)
                    x_tgt, _ = next(tgt_iter)

                x_src = x_src.to(self.device)
                x_tgt = x_tgt.to(self.device)
                bs_s  = x_src.size(0)
                bs_t  = x_tgt.size(0)

                # ── Train discriminator ────────────────────────────────────
                opt_disc.zero_grad()

                with torch.no_grad():
                    feat_src = self._encode_source(x_src)
                feat_tgt = self._encode_target(x_tgt).detach()

                lbl_src = torch.zeros(bs_s, dtype=torch.long, device=self.device)
                lbl_tgt = torch.ones (bs_t, dtype=torch.long, device=self.device)

                out_src = self.discriminator(feat_src)
                out_tgt = self.discriminator(feat_tgt)

                loss_disc = (
                    criterion(out_src, lbl_src) + criterion(out_tgt, lbl_tgt)
                ) / 2.0
                loss_disc.backward()
                opt_disc.step()

                # ── Train target encoder (fool discriminator) ──────────────
                opt_disc.zero_grad()
                opt_target.zero_grad()

                feat_tgt = self._encode_target(x_tgt)
                out_tgt  = self.discriminator(feat_tgt)

                # target encoder wants discriminator to label it as *source* (0)
                lbl_src_fake = torch.zeros(bs_t, dtype=torch.long, device=self.device)
                loss_target  = criterion(out_tgt, lbl_src_fake)
                loss_target.backward()
                opt_target.step()

                running_loss_disc   += loss_disc.item()
                running_loss_target += loss_target.item()

            # ── Epoch summary ──────────────────────────────────────────────
            avg_disc   = running_loss_disc   / iters
            avg_target = running_loss_target / iters
            val_acc    = self._eval_accuracy(target_loader_val)

            self.history["loss_discriminator"].append(avg_disc)
            self.history["loss_target"].append(avg_target)
            self.history["accuracy"].append(val_acc)

            print(
                f"[ADDA Epoch {epoch+1:3d}/{epochs}]  "
                f"disc_loss: {avg_disc:.4f}  "
                f"enc_loss: {avg_target:.4f}  "
                f"val_acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                bad_epochs   = 0
                torch.save(
                    {
                        "encoder_state_dict":   self.target_encoder.state_dict(),
                        "classifier_state_dict": self.classifier.state_dict(),
                        "epoch": epoch,
                        "val_acc": val_acc,
                    },
                    save_path,
                )
                print(f"  [New Best] Val Acc {best_val_acc:.4f} - checkpoint saved.")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch+1}.")
                    break

        # Restore best weights
        best = torch.load(save_path, map_location=self.device, weights_only=False)
        self.target_encoder.load_state_dict(best["encoder_state_dict"])
        self.classifier.load_state_dict(best["classifier_state_dict"])
        print(f"Adaptation complete. Best val acc: {best_val_acc:.4f}")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_accuracy(self, loader: DataLoader) -> float:
        self.target_encoder.eval()
        self.classifier.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            feat   = self._encode_target(x)
            logits = self.classifier(feat)
            preds  = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
        return correct / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    def plot_history(self, save_path: Optional[str] = None) -> None:
        epochs = len(self.history["loss_discriminator"])
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(range(1, epochs + 1), self.history["loss_discriminator"])
        axes[0].set(title="Discriminator Loss", xlabel="Epoch", ylabel="Loss")

        axes[1].plot(range(1, epochs + 1), self.history["loss_target"])
        axes[1].set(title="Target Encoder Loss", xlabel="Epoch", ylabel="Loss")

        axes[2].plot(range(1, epochs + 1), self.history["accuracy"])
        axes[2].set(title="Target Val Accuracy", xlabel="Epoch", ylabel="Accuracy")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
