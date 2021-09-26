import os
import comet_ml
from comet_ml import API
from torch import nn, optim
from torch.utils.data import DataLoader
from catalyst import dl, utils
from catalyst.data import ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.loggers.comet import CometLogger

logger = CometLogger(project_name='Catalyst Integration')

model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)



loaders = {
    "train": DataLoader(
        MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32
    ),
    "valid": DataLoader(
        MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32
    ),
}

runner = dl.SupervisedRunner(
    input_key="features", output_key="logits", target_key="targets", loss_key="loss"
)

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    num_epochs=1,
    callbacks=[
        dl.AccuracyCallback(input_key="logits", target_key="targets", topk_args=(1, 3, 5)),
        dl.PrecisionRecallF1SupportCallback(
            input_key="logits", target_key="targets", num_classes=10
        ),
    ],
    logdir="./logs",
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
    verbose=True,
    load_best_on_end=True,
    loggers={"comet": logger}
)

ID = comet_ml.get_global_experiment().id

api = API()
api_experiment = api.get_get_experiment_by_id(ID)

def test_metrics():
    metrics = api_experiment.get_metrics_summary()
    assert len(metrics) == 2