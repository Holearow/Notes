## [LightenlingModule](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#)

------

A `LightningModule` organizes your PyTorch code into 6 sections:

1. Computations (init).
2. Train Loop (training_step)
3. Validation Loop (validation_step)
4. Test Loop (test_step)
5. Prediction Loop (predict_step)
6. Optimizers and LR Schedulers (configure_optimizers)

------

#### 1. 把tensor转移的GPU的方法：

```python
# don't do in Lightning
x = torch.Tensor(2, 3)
x = x.cuda()
x = x.to(device)

# do this instead
x = x  # leave it alone!

# or to init a new tensor
new_x = torch.Tensor(2, 3)
new_x = new_x.to(x)
```

#### 2. LightningModule的方法：

| Name                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| init                 | Define computations here                                     |
| forward              | <font color='red'>**Use for inference only (separate from training_step)**</font> |
| training_step        | the complete training loop                                   |
| validation_step      | the complete validation loop                                 |
| test_step            | the complete test loop                                       |
| predict_step         | the complete prediction loop                                 |
| configure_optimizers | define optimizers and LR schedulers                          |

#### 3. Training/Validation/Testing

- If you want to calculate epoch-level metrics and log them, use `log()`.

```python
def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x)
    loss = F.cross_entropy(y_hat, y)

    # logs metrics for each training_step,
    # and the average across the epoch, to the progress bar and logger
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss

'''
# The log() object automatically reduces the requested metrics across a complete epoch and devices. Here’s the pseudocode of what it does under the hood:

outs = []
for batch_idx, batch in enumerate(train_dataloader):
    # forward
    loss = training_step(batch, batch_idx)
    outs.append(loss)

    # clear gradients
    optimizer.zero_grad()

    # backward
    loss.backward()

    # update parameters
    optimizer.step()

epoch_metric = torch.mean(torch.stack([x for x in outs]))
'''
```

- If you need to do something with all the outputs of each `training_step()`, override the `training_epoch_end()` method.

```python
def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x)
    loss = F.cross_entropy(y_hat, y)
    preds = ...
    return {"loss": loss, "other_stuff": preds}


def training_epoch_end(self, training_step_outputs):
    all_preds = torch.stack(training_step_outputs)
    ...
```

#### 4. Inference

- There are two ways to call `predict()`:

```python
# call after training
trainer = Trainer()
trainer.fit(model)

# automatically auto-loads the best weights from the previous run
predictions = trainer.predict(dataloaders=predict_dataloader)

# or call with pretrained model
model = MyLightningModule.load_from_checkpoint(PATH)
trainer = Trainer()
predictions = trainer.predict(model, dataloaders=test_dataloader)
```

- If you want to perform inference with the system, you can add a `forward` method to the LightningModule:

```python
# When using forward, you are responsible to call eval() and use the no_grad() context manager.
class Autoencoder(pl.LightningModule):
    def forward(self, x):
        return self.decoder(x)

model = Autoencoder()
model.eval()
with torch.no_grad():
    reconstruction = model(embedding)

# The advantage of adding a forward is that in complex systems, you can do a much more involved inference procedure, such as text generation:
class Seq2Seq(pl.LightningModule):
    def forward(self, x):
        embeddings = self(x)
        hidden_states = self.encoder(embeddings)
        for h in hidden_states:
            # decode
            ...
        return decoded
    
# In the case where you want to scale your inference, you should be using predict_step().
class Autoencoder(pl.LightningModule):
    def forward(self, x):
        return self.decoder(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # this calls forward
        return self(batch)


data_module = ...
model = Autoencoder()
trainer = Trainer(accelerator="gpu", devices=2)
trainer.predict(model, data_module)
```

## [Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#)

#### 1. Reproducibility

- To ensure full reproducibility from run to run you need to set seeds for pseudo-random generators, and set `deterministic` flag in `Trainer`.

```python
from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)
# sets seeds for numpy, torch and python.random.
model = Model()
trainer = Trainer(deterministic=True)
```

