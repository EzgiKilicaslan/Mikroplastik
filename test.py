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

    model = MicroplasticCNNModel.load_from_checkpoint("models/best_MicroplasticCNNModel.ckpt")
    logger = WandbLogger(project=f"Microplastics", log_model=True, name="Test " + type(model).__name__)
    trainer = pl.Trainer(logger=logger, callbacks=[ModelSummary(max_depth=2)], devices=1, num_sanity_val_steps=0)
    logger.watch(model, log="all", log_freq=10)
    trainer.test(model, datamodule=genomic_data_module)

if __name__ == "__main__":
    main()