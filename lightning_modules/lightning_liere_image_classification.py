from models.model_args import ModelArgs
import math

import lightning
import torch
import torch.nn as nn
from scipy import stats
from models.rope_vit import RoPEViT
import numpy as np
import ipdb

from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch
import wandb
import sys
from lightning.pytorch.callbacks import Callback

import ipdb
from pytorch_lightning import seed_everything

seed_everything(42)


class FLOPsAnalysisCallback(Callback):
    def __init__(self, input_shape: list):
        self.input_shape = input_shape

    def on_fit_start(self, trainer, pl_module):
        sample_input = torch.rand(self.input_shape).to(pl_module.device)

        flops = FlopCountAnalysis(pl_module, sample_input)
        flops_table = flop_count_table(flops)
        total_flops = flops.total()

        print(flops_table)
        print(f"Total FLOPs: {total_flops}")

        if wandb.run is not None:
            wandb.log({"total_flops": total_flops})

        print(trainer.model)


class LiereImageClassification(lightning.LightningModule):
    def __init__(
        self,
        learning_rate,
        imsize,
        patch_size,
        **kwargs,
    ):
        super().__init__()
        params = ModelArgs(**kwargs)
        self.learning_rate = learning_rate

        size_params = params.size_params
        dim, depth, heads, mlp_dim = size_params["dim"], size_params["depth"], size_params["heads"], size_params["mlp_dim"]

        self.model = None
        if params.model_architecture == "rope_mixed":
            self.model = RoPEViT(
                image_size=imsize,
                patch_size=patch_size,
                num_classes=params.num_classes,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                positional_encoding_type="liere",
                input_dimensionality=params.input_dimensionality,
                phase_type="naver",
                position_sequencing_type="sequential",
                generator_dim=2,  ############## NAVER #############
                force_absolute_encodings=False,  ############## NAVER #############
                rotary_embedding_per_layer=params.rotary_embedding_per_layer,  ############## NAVER #############
                rotary_embedding_per_head=params.rotary_embedding_per_head,  ############## NAVER #############
                shuffle_patches=params.shuffle_patches,
                checkpoint_attn=params.checkpoint_attn,
                enable_ape=False
            )
        elif params.model_architecture == "liere":
            self.model = RoPEViT(
                image_size=imsize,
                patch_size=patch_size,
                num_classes=params.num_classes,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                positional_encoding_type="liere",
                input_dimensionality=params.input_dimensionality,
                phase_type="naver",  # this is correct
                position_sequencing_type="sequential",
                generator_dim=params.generator_dim,  ############## LIERE #############
                force_absolute_encodings=False,
                rotary_embedding_per_layer=params.rotary_embedding_per_layer,
                rotary_embedding_per_head=params.rotary_embedding_per_head,
                shuffle_patches=params.shuffle_patches,
                checkpoint_attn=params.checkpoint_attn,
                enable_ape=False
            )
        elif params.model_architecture == "absolute":
            self.model = RoPEViT(
                image_size=imsize,
                patch_size=patch_size,
                num_classes=params.num_classes,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                positional_encoding_type="absolute",  ############## absolute #############
                input_dimensionality=params.input_dimensionality,
                shuffle_patches=params.shuffle_patches,
            )
        elif params.model_architecture == "visionllama":
            self.model = RoPEViT(
                image_size=imsize,
                patch_size=patch_size,
                num_classes=params.num_classes,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                positional_encoding_type="tiled",
                input_dimensionality=params.input_dimensionality,
                phase_type="inverse_exp",  ############## visionllama #############
                shuffle_patches=params.shuffle_patches,
            )
        else:
            ValueError(
                "Please specify the model as one of ['visionllama','absolute', 'rope_mixed', 'liere]"
            )

        self.criterion_train = nn.CrossEntropyLoss()
        self.criterion_test = nn.CrossEntropyLoss( reduction='none')

        self.model_architecture = params.model_architecture
        self.shuffle_patches = params.shuffle_patches
        self.validation_step_outputs = []

        if hasattr(self.model, "position_encoder") and hasattr(self.model.position_encoder, "initialization_factor"):
            self.save_hyperparameters({"initialization_factor":self.model.position_encoder.initialization_factor})

    def setup(self, stage=None):
        # This method is called by PyTorch Lightning after the trainer is set
        if self.trainer is not None:
            self.accumulation_factor = self.trainer.accumulate_grad_batches

    def parse_batch(self, batch):
        if len(batch) == 2:
            inputs, targets = batch
            # Handle the case where the batch has (inputs, targets) structure
        elif len(batch) == 3:
            inputs, _, targets = batch
        elif len(batch) == 4:
            inputs, targets, _, _ = batch
        elif len(batch) == 5:
            inputs, targets, _, _, _ = batch
        else:
            raise ValueError(f"Unexpected batch structure with len {len(batch)}")

        return inputs, targets

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        assert (
            self.shuffle_patches == False
        ), f"Shuffle patches not allowed for training, currently set to {self.shuffle_patches}"

        inputs, targets = self.parse_batch(batch)
  
        outputs = self.model(inputs)
        loss = self.criterion_train(outputs, targets)
        self.log(name="train_loss", value=loss, prog_bar=True)
        self.log(name="lr", value=self.lr_schedulers().get_last_lr()[0])

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = self.parse_batch(batch)
        outputs = self.model(inputs)
        loss = self.criterion_test(outputs, targets)

        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).detach()

        self.log(
            "val_acc_dist",
            correct.sum() / len(predicted) * 100,
            sync_dist=True,
            rank_zero_only=False,
        )
        self.log(
            "val_loss_dist",
            loss.mean(),
            sync_dist=True,
            rank_zero_only=False,
        )

        return {"correct": correct, "loss": loss}

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.validation_step_outputs.append(outputs)

    def on_validation_epoch_end(self):
        def process_results(validation_step_outputs, part="val"):
            all = torch.cat([x[part] for x in validation_step_outputs])
            gathered = self.all_gather(all)
            gathered = gathered.reshape(-1).cpu().numpy().astype(np.float32)
            mean = np.mean(gathered)
            se = stats.sem(gathered)

            # Calculate 95% confidence interval
            ci = stats.t.interval(
                confidence=0.95, df=len(gathered) - 1, loc=mean, scale=se
            )
            return mean, ci
        # Process results for each dataloader
        mean, ci = process_results(self.validation_step_outputs, "correct")
        mean_loss, ci_loss = process_results(self.validation_step_outputs, "loss")

        # Log metrics
        self.log("val_acc", mean * 100, sync_dist=False, rank_zero_only=True)
        self.log("val_acc_ci_lower", ci[0] * 100, sync_dist=False, rank_zero_only=True)
        self.log("val_acc_ci_upper", ci[1] * 100, sync_dist=False, rank_zero_only=True)
        self.log("val_loss_mean", mean_loss, sync_dist=False, rank_zero_only=True)
        self.log("val_loss_ci_lower", ci_loss[0], sync_dist=False, rank_zero_only=True)
        self.log("val_loss_ci_upper", ci_loss[1], sync_dist=False, rank_zero_only=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }
