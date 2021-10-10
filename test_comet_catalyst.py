import os
import comet_ml
from comet_ml import API
from torch import nn, optim
from torch.utils.data import DataLoader
from catalyst import dl
from catalyst.data import ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.loggers.comet import CometLogger
from catalyst_pytest_consts import EXPECTED_METRICS
import multiprocessing


class AlwaysEquals(object):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


def train_model(logger):
    """
    Initiates and trains the model.
    """

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
            dl.AccuracyCallback(input_key="logits",
                                target_key="targets", topk_args=(1, 3, 5)),
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


def build_result(name):
    if 'sys.' not in name:
        return {'name': name, 'valueMax': AlwaysEquals(), 'valueMin': AlwaysEquals(), 'valueCurrent': AlwaysEquals(), 'timestampMax': AlwaysEquals(), 'timestampMin': AlwaysEquals(), 'timestampCurrent': AlwaysEquals(), 'stepMax': AlwaysEquals(), 'stepMin': AlwaysEquals(), 'stepCurrent': AlwaysEquals(), 'editable': False}
    else:
        return {'name': name, 'valueMax': AlwaysEquals(), 'valueMin': AlwaysEquals(), 'valueCurrent': AlwaysEquals(), 'timestampMax': AlwaysEquals(), 'timestampMin': AlwaysEquals(), 'timestampCurrent': AlwaysEquals(), 'editable': False}


def get_system_cpus():
    """
    Returns the amount of cpu's of the machine that the test is running on.
    """
    number_of_cpus = multiprocessing.cpu_count()
    return ['sys.cpu.percent.' + str(number).zfill(2) for number in range(1, number_of_cpus + 1)]


def test_metrics():
    logger = CometLogger(project_name='catalyst-pytest', logging_frequency=10)
    train_model(logger=logger)
    experiment_id = comet_ml.get_global_experiment().id

    api = API()
    api_experiment = api.get_experiment_by_id(experiment_id)

    metrics = api_experiment.get_metrics_summary()

    cpus = get_system_cpus()
    expected_metric_names = EXPECTED_METRICS + cpus

    for item in expected_metric_names:
        assert build_result(item) in metrics
