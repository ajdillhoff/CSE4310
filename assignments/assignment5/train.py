import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from NYULightningModule import NYUDataModule
from ViTHPELightningModule import ViTHPELightningModule


def main():
    hparams = {
        "learning_rate": 1e-4,
        "weight_decay": 0.0001,
        "scheduler_step_size": 10,
        "scheduler_gamma": 0.1,
        "batch_size": 64,
        "num_workers": 12,
        "img_size": 224,
        "epochs": 10,
        "num_kp": 22,
    }

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='vit-hpe-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    model = ViTHPELightningModule(hparams)
    dm = NYUDataModule(data_dir="/mnt/Data/NYU/",
                       batch_size=hparams["batch_size"],
                       num_workers=hparams["num_workers"],
                       img_size=hparams["img_size"])
    trainer = pl.Trainer(accelerator='gpu',
                         max_epochs=hparams["epochs"],
                         callbacks=[checkpoint_callback])
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()