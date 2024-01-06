import datasets
import lightning.pytorch as pl
from model import MicroplasticCNNModel, MicroplasticCNNModel2
from lightning.pytorch.callbacks import ModelSummary
import os
import shutil
from lightning.pytorch.loggers import WandbLogger
import time
from lightning.pytorch.callbacks import ModelCheckpoint

def main():
    pl.seed_everything(1996)
    batch_size = 4

    genomic_data_module = datasets.MicroplasticDataModule(batch_size)

    model = MicroplasticCNNModel()

    checkpoint_callback_best = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        dirpath=f"models/",
        filename="best_"+type(model).__name__,
        mode="min"
    )

    logger = WandbLogger(project=f"Microplastics", log_model=True, name=type(model).__name__)
    trainer = pl.Trainer(logger=logger, gradient_clip_val=1, detect_anomaly=True, callbacks=[ModelSummary(max_depth=2), checkpoint_callback_best], max_epochs=50, num_sanity_val_steps=1, accumulate_grad_batches=2)
    logger.watch(model, log="all", log_freq=10)
    trainer.fit(model, datamodule=genomic_data_module)

if __name__ == "__main__":
    main()