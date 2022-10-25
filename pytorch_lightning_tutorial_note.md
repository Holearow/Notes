### 1 Train a model

- 第一步需要define a LightningModule，里面要定义training_step和configure_optimizers，如下例子：

```python
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```

<font color=red>**# Encoder和Decoder这两个类其实可以直接在LitAutoEncoder的init里定义**</font>

- 然后定义数据集开始训练。lightning会自动进行遍历batch，算loss，`loss.backward()`以及`optimizer.step()`这些流程，不用自己写：

```python
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset)

# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# train model
trainer = pl.Trainer()
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
```

### 2.1 Validate and test a model

- 添加test loop的例子：

```python
class LitAutoEncoder(pl.LightningModule):
	def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

# initialize the Trainer
trainer = Trainer()

# test the model
trainer.test(model, dataloaders=DataLoader(test_set)) 
```

<font color='red'>**# 这里的test_step可以return其他东西吗，不行的话需要在这里写很多代码，比如输出和评分的代码...**</font>

- 添加validate loop的例子，<font color='orange'>Trainer里面有怎么指定显卡，有个页面详细介绍Train</font>，看完基础教程看看：

```python
class LitAutoEncoder(pl.LightningModule):
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)
        
# train with both splits
trainer = Trainer()
trainer.fit(model, train_loader, valid_loader)
```

### 2.2 Save your model progress

- Checkpoint相关的用法：

```python
# saves checkpoints to 'some/path/' at every epoch end（手动更改路径的方法）
trainer = Trainer(default_root_dir="some/path/")

# load checkpoints
model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")

# resume training state - automatically restores model, epoch, step, LR schedulers, apex, etc...
trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")
```

- 保存超参数那个函数叫save_hyperparameters()，需要的话去tutorial看看。

<font color='red'>**# 在后面看看有没有根据evaluation metric择优手动保存checkpoint的方法**</font>

### 2.3 Enable early stopping

- 早停的第一种方法是override钩子`on_train_batch_start()`这个方法，出情况return -1就可以结束训练。
- 第二种方法是使用EarlyStopping Callback，<font color='orange'>EarlyStopping的其他参数以及监测内容/方法参考tutorial</font>：

```python
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class LitModel(LightningModule):
    def validation_step(self, batch, batch_idx):
        loss = ...
        self.log("val_loss", loss)

model = LitModel()
trainer = Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
trainer.fit(model)

# You can customize the callbacks behaviour by changing its parameters.
early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")
trainer = Trainer(callbacks=[early_stop_callback])
```

### 3 Use pretrained models

- 用BERT的一个例子，<font color='orange'>这个例子返回的不是loss，那loss在哪里算？目前想法是training_step_end()里算？或者直接在这里算loss？（这个例子或许是Inference用的，所以不用训练...）</font>：

```python
class BertMNLIFinetuner(LightningModule):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased", output_attentions=True)
        self.W = nn.Linear(bert.config.hidden_size, 3)
        self.num_classes = 3

    def forward(self, input_ids, attention_mask, token_type_ids):
        h, _, attn = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        h_cls = h[:, 0]
        logits = self.W(h_cls)
        return logits, attn
```

<font color='red'>**# Prefix-tuning的时候要冻结BART，冻结模型用`model.freeze()`**</font>

### 4 Enable script parameters

- 配置Trainer Arguments，Model Specific Arguments & Program Arguments的例子：

```python
# First, in your LightningModule, define the arguments specific to that module. Remember that data splits or data paths may also be specific to a module.
# ----------------
# model.py
# ----------------
class LitModel(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--encoder_layers", type=int, default=12)
        parser.add_argument("--data_path", type=str, default="/some/path")
        return parent_parser
    
# Second, in your main trainer file, add the Trainer args, the program args, and add the model args.
# ----------------
# trainer_main.py
# ----------------
from argparse import ArgumentParser

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument("--conda_env", type=str, default="some_name")
parser.add_argument("--notification_email", type=str, default="will@email.com")

# add model specific args
parser = LitModel.add_model_specific_args(parser)

# add all the available trainer options to argparse
# ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()
```

<font color='red'>**# Trainer的argument怎么添加啊——不用添加，Arg得是Train-flag里的东西...**</font>

- 添加后的使用方法，<font color='orange'>以及错误例子示范</font>：

```python
# init the trainer like this
trainer = Trainer.from_argparse_args(args, early_stopping_callback=...)

# NOT like this
# trainer = Trainer(accelerator=hparams.accelerator, devices=hparams.devices, ...)

# init the model with Namespace directly
model = LitModel(args)

# or init the model with all the key-value pairs
dict_args = vars(args)
model = LitModel(**dict_args)
```

- 跑程序的时候就可以：

```bash
python trainer_main.py --accelerator 'gpu' --devices 2 --num_nodes 2 --conda_env 'my_env' --encoder_layers 12
```

- save_hyperparameters()可以保存参数到属性self.hparams以及checkpoint，用法：

```python
class LitMNIST(LightningModule):
    def __init__(self, layer_1_dim=128, learning_rate=1e-2):
        super().__init__()
        # call this to save (layer_1_dim=128, learning_rate=1e-4) to the checkpoint
        self.save_hyperparameters()

        # equivalent
        self.save_hyperparameters("layer_1_dim", "learning_rate")

        # Now possible to access layer_1_dim from hparams
        self.hparams.layer_1_dim

# Excluding hyperparameters
class LitMNIST(LightningModule):
    def __init__(self, loss_fx, generator_network, layer_1_dim=128):
        super().__init__()
        self.layer_1_dim = layer_1_dim
        self.loss_fx = loss_fx

        # call this to save only (layer_1_dim=128) to the checkpoint
        self.save_hyperparameters("layer_1_dim")

        # equivalent
        self.save_hyperparameters(ignore=["loss_fx", "generator_network"])
```

### 5.1 Debug your model

- 一些debug方法：

```python
# Run all your model code once quickly -- fast_dev_run argument
Trainer(fast_dev_run=True)
Trainer(fast_dev_run=7)

# Shorten the epoch length using a fraction of your training, val, test, or predict dat
# use only 10% of training data and 1% of val data
trainer = Trainer(limit_train_batches=0.1, limit_val_batches=0.01)
# use 10 batches of train and 5 batches of val
trainer = Trainer(limit_train_batches=10, limit_val_batches=5)

# Run a Sanity Check
trainer = Trainer(num_sanity_val_steps=2)

# Print LightningModule weights summary
''' 
Whenever the .fit() function gets called, the Trainer will print the weights summary for the LightningModule.

this generate a table like:

  | Name  | Type        | Params
----------------------------------
0 | net   | Sequential  | 132 K
1 | net.0 | Linear      | 131 K
2 | net.1 | BatchNorm1d | 1.0 K
''' 

# Print input output layer dimensions - 同上也是trainer.fit()自动
```

### 6.2 Predict with LightningModule

- 读取模型加预测的一个简单例子：

```python
model = LitModel.load_from_checkpoint("best_model.ckpt")
model.eval()
x = torch.randn(1, 64)
with torch.no_grad():
    y_hat = model(x)
    
# 可以改进成：
class MyModel(LightningModule):
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    
data_loader = DataLoader(...)
model = MyModel()
trainer = Trainer()
predictions = trainer.predict(model, data_loader)

# Enable complicated predict logic -- When you need to add complicated pre-processing or post-processing logic to your data use the predict step. For example here we do Monte Carlo Dropout for predictions:
class LitMCdropoutModel(pl.LightningModule):
    def __init__(self, model, mc_iteration):
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout()
        self.mc_iteration = mc_iteration

    def predict_step(self, batch, batch_idx):
        # enable Monte Carlo Dropout
        self.dropout.train()

        # take average of `self.mc_iteration` iterations
        pred = [self.dropout(self.model(x)).unsqueeze(0) for _ in range(self.mc_iteration)]
        pred = torch.vstack(pred).mean(dim=0)
        return pred
```

### 7.1 Hardware agnostic training

- Init tensors using Tensor.to and register_buffer：

```python
'''When you need to create a new tensor, use Tensor.to. This will make your code scale to any arbitrary number of GPUs or TPUs with Lightning.'''
# before lightning
def forward(self, x):
    z = torch.Tensor(2, 3)
    z = z.cuda(0)

# with lightning
def forward(self, x):
    z = torch.Tensor(2, 3)
    z = z.to(x)

'''The LightningModule knows what device it is on. You can access the reference via self.device. Sometimes it is necessary to store tensors as module attributes. However, if they are not parameters they will remain on the CPU even if the module gets moved to a new device. To prevent that and remain device agnostic, register the tensor as a buffer in your modules’ __init__ method with register_buffer().'''
class LitModel(LightningModule):
    def __init__(self):
        ...
        self.register_buffer("sigma", torch.eye(3))
        # you can now access self.sigma anywhere in your module
```

### 7.3 GPU Training

- 定义GPU的方法：

```python
# DEFAULT (int) specifies how many GPUs to use per node
Trainer(accelerator="gpu", devices=k)

# Above is equivalent to
Trainer(accelerator="gpu", devices=list(range(k)))

# Specify which GPUs to use (don't use when running on cluster)
Trainer(accelerator="gpu", devices=[0, 1])

# Equivalent using a string
Trainer(accelerator="gpu", devices="0, 1")

# To use all available GPUs put -1 or '-1'
# equivalent to list(range(torch.cuda.device_count()))
Trainer(accelerator="gpu", devices=-1)
```

### 10.1 Alter checkpoint behavior

- 保存前k个最好的checkpoint方法：

```python
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(dirpath="my/path/", save_top_k=2, monitor="val_loss")
trainer = Trainer(callbacks=[checkpoint_callback])
trainer.fit(model)
checkpoint_callback.best_model_path		#这句代码美没看懂有什么用...

class LitModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        self.log("my_metric", x)

# 'my_metric' is now able to be monitored
checkpoint_callback = ModelCheckpoint(monitor="my_metric")
```

- 定义checkpoint的例子：

```python
from pytorch_lightning.callbacks import ModelCheckpoint

# saves top-K checkpoints based on "val_loss" metric
checkpoint_callback = ModelCheckpoint(
    save_top_k=10,
    monitor="val_loss",
    mode="min",
    dirpath="my/path/",
    filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
)

# saves last-K checkpoints based on "global_step" metric
# make sure you log it inside your LightningModule
checkpoint_callback = ModelCheckpoint(
    save_top_k=10,
    monitor="global_step",
    mode="max",
    dirpath="my/path/",
    filename="sample-mnist-{epoch:02d}-{global_step}",
)
```

- 又一个例子：

```python
from pytorch_lightning.callbacks import ModelCheckpoint

class LitAutoEncoder(LightningModule):
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        # 1. calculate loss
        loss = F.cross_entropy(y_hat, y)
        # 2. log val_loss
        self.log("val_loss", loss)

# 3. Init ModelCheckpoint callback, monitoring "val_loss"
checkpoint_callback = ModelCheckpoint(monitor="val_loss")

# 4. Add your callback to the callbacks list
trainer = Trainer(callbacks=[checkpoint_callback])
```

- Save checkpoints manually：

```python
model = MyLightningModule(hparams)
trainer.fit(model)
trainer.save_checkpoint("example.ckpt")

# load the checkpoint later as normal
new_model = MyLightningModule.load_from_checkpoint(checkpoint_path="example.ckpt")
```

