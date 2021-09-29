import os
import comet_ml
from comet_ml import API
from torch import nn, optim
from torch.utils.data import DataLoader
from catalyst import dl, utils
from catalyst.data import ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.loggers.comet import CometLogger

class AlwaysEquals(object):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

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


def train_model():
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
    return {"name": name, 'valueMax': AlwaysEquals(), 'valueMin': AlwaysEquals(), 'valueCurrent': AlwaysEquals(), 'timestampMax': AlwaysEquals(), 'timestampMin': AlwaysEquals(), 'timestampCurrent': AlwaysEquals(), 'stepMax': AlwaysEquals(), 'stepMin': AlwaysEquals(), 'stepCurrent': AlwaysEquals(), 'editable': False}

def test_metrics():
    train_model()
    ID = comet_ml.get_global_experiment().id

    api = API()
    api_experiment = api.get_experiment_by_id(ID)

    metrics = api_experiment.get_metrics_summary()
    assert len(metrics) == 69
    for item in ["loss", "sys.cpu.percent.01", 'sys.cpu.percent.avg', 'sys.load.avg', 'sys.ram.total']:
        assert build_result(item) in metrics
    metrics_result = [{'name': 'loss', 'valueMax': AlwaysEquals(), 'valueMin': '0.019660653546452522', 'valueCurrent': '0.14634065330028534', 'timestampMax': 1632936038457, 'timestampMin': 1632936038457, 'timestampCurrent': 1632936074272, 'stepMax': 0, 'stepMin': 330, 'stepCurrent': 1610, 'editable': False}, 
                      {'name': 'sys.cpu.percent.01', 'valueMax': '48.2', 'valueMin': '48.2', 'valueCurrent': '48.2', 'timestampMax': 1632936038112, 'timestampMin': 1632936038112, 'timestampCurrent': 1632936038112, 'editable': False},
                      {'name': 'sys.cpu.percent.02', 'valueMax': '7.4', 'valueMin': '7.4', 'valueCurrent': '7.4', 'timestampMax': 1632936038112, 'timestampMin': 1632936038112, 'timestampCurrent': 1632936038112, 'editable': False}, 
                      {'name': 'sys.cpu.percent.03', 'valueMax': '40.7', 'valueMin': '40.7', 'valueCurrent': '40.7', 'timestampMax': 1632936038112, 'timestampMin': 1632936038112, 'timestampCurrent': 1632936038112, 'editable': False},
                      {'name': 'sys.cpu.percent.04', 'valueMax': '4.9', 'valueMin': '4.9', 'valueCurrent': '4.9', 'timestampMax': 1632936038112, 'timestampMin': 1632936038112, 'timestampCurrent': 1632936038112, 'editable': False},
                      {'name': 'sys.cpu.percent.05', 'valueMax': '34.0', 'valueMin': '34.0', 'valueCurrent': '34.0', 'timestampMax': 1632936038112, 'timestampMin': 1632936038112, 'timestampCurrent': 1632936038112, 'editable': False},
                      {'name': 'sys.cpu.percent.06', 'valueMax': '4.2', 'valueMin': '4.2', 'valueCurrent': '4.2', 'timestampMax': 1632936038112, 'timestampMin': 1632936038112, 'timestampCurrent': 1632936038112, 'editable': False},
                      {'name': 'sys.cpu.percent.07', 'valueMax': '31.1', 'valueMin': '31.1', 'valueCurrent': '31.1', 'timestampMax': 1632936038112, 'timestampMin': 1632936038112, 'timestampCurrent': 1632936038112, 'editable': False},
                      {'name': 'sys.cpu.percent.08', 'valueMax': '4.2', 'valueMin': '4.2', 'valueCurrent': '4.2', 'timestampMax': 1632936038112, 'timestampMin': 1632936038112, 'timestampCurrent': 1632936038112, 'editable': False},
                      {'name': 'sys.cpu.percent.avg', 'valueMax': '21.8375', 'valueMin': '21.8375', 'valueCurrent': '21.8375', 'timestampMax': 1632936038112, 'timestampMin': 1632936038112, 'timestampCurrent': 1632936038112, 'editable': False},
                      {'name': 'sys.load.avg', 'valueMax': '2.7265625', 'valueMin': '2.7265625', 'valueCurrent': '2.7265625', 'timestampMax': 1632936038112, 'timestampMin': 1632936038112, 'timestampCurrent': 1632936038112, 'editable': False},
                      {'name': 'sys.ram.total', 'valueMax': '3.4359738368E10', 'valueMin': '3.4359738368E10', 'valueCurrent': '3.4359738368E10', 'timestampMax': 1632936038112, 'timestampMin': 1632936038112, 'timestampCurrent': 1632936038112, 'editable': False},
                      {'name': 'sys.ram.used', 'valueMax': '1.1982712832E10', 'valueMin': '1.1982712832E10', 'valueCurrent': '1.1982712832E10', 'timestampMax': 1632936038112, 'timestampMin': 1632936038112, 'timestampCurrent': 1632936038112, 'editable': False},
                      {'name': 'train/train_batch_accuracy', 'valueMax': '1.0', 'valueMin': '0.09375', 'valueCurrent': '0.84375', 'timestampMax': 1632936038459, 'timestampMin': 1632936038459, 'timestampCurrent': 1632936075223, 'stepMax': 195, 'stepMin': 1, 'stepCurrent': 1654, 'editable': False},
                      {'name': 'train/train_batch_accuracy01', 'valueMax': '1.0', 'valueMin': '0.09375', 'valueCurrent': '0.84375', 'timestampMax': 1632936038459, 'timestampMin': 1632936038459, 'timestampCurrent': 1632936075199, 'stepMax': 195, 'stepMin': 1, 'stepCurrent': 1653, 'editable': False},
                      {'name': 'train/train_batch_accuracy03', 'valueMax': '1.0', 'valueMin': '0.28125', 'valueCurrent': '1.0', 'timestampMax': 1632936038459, 'timestampMin': 1632936038459, 'timestampCurrent': 1632936075138, 'stepMax': 7, 'stepMin': 1, 'stepCurrent': 1650, 'editable': False}, 
                      {'name': 'train/train_batch_accuracy05', 'valueMax': '1.0', 'valueMin': '0.4375', 'valueCurrent': '1.0', 'timestampMax': 1632936038459, 'timestampMin': 1632936038459, 'timestampCurrent': 1632936074998, 'stepMax': 7, 'stepMin': 1, 'stepCurrent': 1644, 'editable': False}, 
                      {'name': 'train/train_batch_f1/_macro', 'valueMax': '0.999995000025', 'valueMin': '0.07657038882081445', 'valueCurrent': '0.7093607564003154', 'timestampMax': 1632936038459, 'timestampMin': 1632936038459, 'timestampCurrent': 1632936075158, 'stepMax': 195, 'stepMin': 1, 'stepCurrent': 1651, 'editable': False},
                      {'name': 'train/train_batch_f1/_micro', 'valueMax': '0.9999950000249999', 'valueMin': '0.09374500026665245', 'valueCurrent': '0.906245000027586', 'timestampMax': 1632936038459, 'timestampMin': 1632936038459, 'timestampCurrent': 1632936074886, 'stepMax': 195, 'stepMin': 1, 'stepCurrent': 1639, 'editable': False}, 
                      {'name': 'train/train_batch_f1/_weighted', 'valueMax': '0.9999950000250001', 'valueMin': '0.04785649301300903', 'valueCurrent': '0.8131581552521063', 'timestampMax': 1632936038459, 'timestampMin': 1632936038459, 'timestampCurrent': 1632936074906, 'stepMax': 1019, 'stepMin': 1, 'stepCurrent': 1640, 'editable': False}, 
                      {'name': 'train/train_batch_f1/class_00', 'valueMax': '0.9999950000249999', 'valueMin': '0.0', 'valueCurrent': '0.8888839488050712', 'timestampMax': 1632936038459, 'timestampMin': 1632936038459, 'timestampCurrent': 1632936075244, 'stepMax': 7, 'stepMin': 1, 'stepCurrent': 1655, 'editable': False}, 
                      {'name': 'train/train_batch_f1/class_01', 'valueMax': '0.9999950000249999', 'valueMin': '0.0', 'valueCurrent': '0.9999950000249999', 'timestampMax': 1632936038459, 'timestampMin': 1632936038459, 'timestampCurrent': 1632936075223, 'stepMax': 7, 'stepMin': 1, 'stepCurrent': 1654, 'editable': False}, 
                      {'name': 'train/train_batch_f1/class_02', 'valueMax': '0.9999950000249999', 'valueMin': '0.0', 'valueCurrent': '0.9999950000249999', 'timestampMax': 1632936038459, 'timestampMin': 1632936038459, 'timestampCurrent': 1632936075092, 'stepMax': 7, 'stepMin': 1, 'stepCurrent': 1648, 'editable': False},
                      {'name': 'train/train_batch_f1/class_03', 'valueMax': '0.9999950000249999', 'valueMin': '0.0', 'valueCurrent': '0.3333288889481474', 'timestampMax': 1632936038459, 'timestampMin': 1632936038459, 'timestampCurrent': 1632936075244, 'stepMax': 26, 'stepMin': 1, 'stepCurrent': 1655, 'editable': False}, 
                      {'name': 'train/train_batch_f1/class_04', 'valueMax': '0.9999950000249999', 'valueMin': '0.0', 'valueCurrent': '0.4999962276764944', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075138, 'stepMax': 7, 'stepMin': 1, 'stepCurrent': 1650, 'editable': False}, 
                      {'name': 'train/train_batch_f1/class_05', 'valueMax': '0.9999950000249999', 'valueMin': '0.0', 'valueCurrent': '0.8571379592116616', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075114, 'stepMax': 7, 'stepMin': 1, 'stepCurrent': 1649, 'editable': False}, 
                      {'name': 'train/train_batch_f1/class_06', 'valueMax': '0.9999950000249999', 'valueMin': '0.0', 'valueCurrent': '0.9999950000249999', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075044, 'stepMax': 4, 'stepMin': 2, 'stepCurrent': 1646, 'editable': False}, 
                      {'name': 'train/train_batch_f1/class_07', 'valueMax': '0.9999950000249999', 'valueMin': '0.0', 'valueCurrent': '0.9999950000249999', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936074975, 'stepMax': 14, 'stepMin': 2, 'stepCurrent': 1643, 'editable': False}, 
                      {'name': 'train/train_batch_f1/class_08', 'valueMax': '0.9999950000249999', 'valueMin': '0.0', 'valueCurrent': '0.7999951928762566', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936074975, 'stepMax': 4, 'stepMin': 2, 'stepCurrent': 1643, 'editable': False}, 
                      {'name': 'train/train_batch_f1/class_09', 'valueMax': '0.9999950000249999', 'valueMin': '0.0', 'valueCurrent': '0.7999951928762566', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075223, 'stepMax': 7, 'stepMin': 1, 'stepCurrent': 1654, 'editable': False}, 
                      {'name': 'train/train_batch_loss', 'valueMax': '2.5467100143432617', 'valueMin': '0.009165357798337936', 'valueCurrent': '0.5332935452461243', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936074975, 'stepMax': 1576, 'stepMin': 1514, 'stepCurrent': 1643, 'editable': False}, 
                      {'name': 'train/train_batch_lr', 'valueMax': '0.02', 'valueMin': '0.02', 'valueCurrent': '0.02', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075092, 'stepMax': 1, 'stepMin': 1, 'stepCurrent': 1648, 'editable': False}, 
                      {'name': 'train/train_batch_momentum', 'valueMax': '0.9', 'valueMin': '0.9', 'valueCurrent': '0.9', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075244, 'stepMax': 1, 'stepMin': 1, 'stepCurrent': 1655, 'editable': False}, 
                      {'name': 'train/train_batch_precision/_macro', 'valueMax': '1.0', 'valueMin': '0.057681155204772946', 'valueCurrent': '0.9833333328366279', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936074929, 'stepMax': 195, 'stepMin': 1, 'stepCurrent': 1641, 'editable': False}, 
                      {'name': 'train/train_batch_precision/_micro', 'valueMax': '1.0', 'valueMin': '0.09375', 'valueCurrent': '0.84375', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075224, 'stepMax': 195, 'stepMin': 1, 'stepCurrent': 1654, 'editable': False},
                      {'name': 'train/train_batch_precision/_weighted', 'valueMax': '1.0', 'valueMin': '0.03605072200298309', 'valueCurrent': '0.8406249992549419', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075044, 'stepMax': 170, 'stepMin': 1, 'stepCurrent': 1646, 'editable': False}, 
                      {'name': 'train/train_batch_precision/class_00', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '1.0', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936074929, 'stepMax': 8, 'stepMin': 1, 'stepCurrent': 1641, 'editable': False}, 
                      {'name': 'train/train_batch_precision/class_01', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '1.0', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075179, 'stepMax': 3, 'stepMin': 1, 'stepCurrent': 1652, 'editable': False}, 
                      {'name': 'train/train_batch_precision/class_02', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '1.0', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075244, 'stepMax': 6, 'stepMin': 1, 'stepCurrent': 1655, 'editable': False}, 
                      {'name': 'train/train_batch_precision/class_03', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '1.0', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075065, 'stepMax': 6, 'stepMin': 1, 'stepCurrent': 1647, 'editable': False}, 
                      {'name': 'train/train_batch_precision/class_04', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '0.75', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075200, 'stepMax': 3, 'stepMin': 1, 'stepCurrent': 1653, 'editable': False}, 
                      {'name': 'train/train_batch_precision/class_05', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '1.0', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075179, 'stepMax': 5, 'stepMin': 1, 'stepCurrent': 1652, 'editable': False}, 
                      {'name': 'train/train_batch_precision/class_06', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '1.0', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075021, 'stepMax': 4, 'stepMin': 2, 'stepCurrent': 1645, 'editable': False}, 
                      {'name': 'train/train_batch_precision/class_07', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '1.0', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075224, 'stepMax': 4, 'stepMin': 2, 'stepCurrent': 1654, 'editable': False}, 
                      {'name': 'train/train_batch_precision/class_08', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '1.0', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075066, 'stepMax': 4, 'stepMin': 2, 'stepCurrent': 1647, 'editable': False}, 
                      {'name': 'train/train_batch_precision/class_09', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '1.0', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936074976, 'stepMax': 7, 'stepMin': 1, 'stepCurrent': 1643, 'editable': False}, 
                      {'name': 'train/train_batch_recall/_macro', 'valueMax': '1.0', 'valueMin': '0.15', 'valueCurrent': '0.775', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075158, 'stepMax': 195, 'stepMin': 1, 'stepCurrent': 1651, 'editable': False}, 
                      {'name': 'train/train_batch_recall/_micro', 'valueMax': '1.0', 'valueMin': '0.09375', 'valueCurrent': '0.75', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075244, 'stepMax': 195, 'stepMin': 1, 'stepCurrent': 1655, 'editable': False}, 
                      {'name': 'train/train_batch_recall/_weighted', 'valueMax': '1.0', 'valueMin': '0.09375', 'valueCurrent': '0.7812499995343387', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936074698, 'stepMax': 195, 'stepMin': 1, 'stepCurrent': 1630, 'editable': False}, 
                      {'name': 'train/train_batch_recall/class_00', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '1.0', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075200, 'stepMax': 7, 'stepMin': 1, 'stepCurrent': 1653, 'editable': False}, 
                      {'name': 'train/train_batch_recall/class_01', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '1.0', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936074953, 'stepMax': 7, 'stepMin': 1, 'stepCurrent': 1642, 'editable': False}, 
                      {'name': 'train/train_batch_recall/class_03', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '0.5', 'timestampMax': 1632936038461, 'timestampMin': 1632936038461, 'timestampCurrent': 1632936075114, 'stepMax': 2, 'stepMin': 1, 'stepCurrent': 1649, 'editable': False}, 
                      {'name': 'train/train_batch_recall/class_02', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '0.5999999940395355', 'timestampMax': 1632936038460, 'timestampMin': 1632936038460, 'timestampCurrent': 1632936075244, 'stepMax': 7, 'stepMin': 1, 'stepCurrent': 1655, 'editable': False}, 
                      {'name': 'train/train_batch_recall/class_04', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '1.0', 'timestampMax': 1632936038461, 'timestampMin': 1632936038461, 'timestampCurrent': 1632936075066, 'stepMax': 7, 'stepMin': 1, 'stepCurrent': 1647, 'editable': False}, 
                      {'name': 'train/train_batch_recall/class_05', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '0.5', 'timestampMax': 1632936038461, 'timestampMin': 1632936038461, 'timestampCurrent': 1632936074907, 'stepMax': 7, 'stepMin': 1, 'stepCurrent': 1640, 'editable': False}, 
                      {'name': 'train/train_batch_recall/class_06', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '1.0', 'timestampMax': 1632936038461, 'timestampMin': 1632936038461, 'timestampCurrent': 1632936075114, 'stepMax': 7, 'stepMin': 2, 'stepCurrent': 1649, 'editable': False}, 
                      {'name': 'train/train_batch_recall/class_07', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '1.0', 'timestampMax': 1632936038461, 'timestampMin': 1632936038461, 'timestampCurrent': 1632936075139, 'stepMax': 7, 'stepMin': 2, 'stepCurrent': 1650, 'editable': False}, 
                      {'name': 'train/train_batch_recall/class_08', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '0.75', 'timestampMax': 1632936038461, 'timestampMin': 1632936038461, 'timestampCurrent': 1632936075201, 'stepMax': 4, 'stepMin': 2, 'stepCurrent': 1653, 'editable': False}, 
                      {'name': 'train/train_batch_recall/class_09', 'valueMax': '1.0', 'valueMin': '0.0', 'valueCurrent': '1.0', 'timestampMax': 1632936038461, 'timestampMin': 1632936038461, 'timestampCurrent': 1632936075180, 'stepMax': 7, 'stepMin': 1, 'stepCurrent': 1652, 'editable': False}, 
                      {'name': 'train/train_batch_support/class_00', 'valueMax': '8.0', 'valueMin': '0.0', 'valueCurrent': '5.0', 'timestampMax': 1632936038461, 'timestampMin': 1632936038461, 'timestampCurrent': 1632936075245, 'stepMax': 43, 'stepMin': 18, 'stepCurrent': 1655, 'editable': False}, 
                      {'name': 'train/train_batch_support/class_01', 'valueMax': '9.0', 'valueMin': '0.0', 'valueCurrent': '3.0', 'timestampMax': 1632936038461, 'timestampMin': 1632936038461, 'timestampCurrent': 1632936075224, 'stepMax': 545, 'stepMin': 65, 'stepCurrent': 1654, 'editable': False}, 
                      {'name': 'train/train_batch_support/class_02', 'valueMax': '8.0', 'valueMin': '0.0', 'valueCurrent': '4.0', 'timestampMax': 1632936038461, 'timestampMin': 1632936038461, 'timestampCurrent': 1632936075201, 'stepMax': 185, 'stepMin': 2, 'stepCurrent': 1653, 'editable': False}, 
                      {'name': 'train/train_batch_support/class_03', 'valueMax': '8.0', 'valueMin': '0.0', 'valueCurrent': '1.0', 'timestampMax': 1632936038461, 'timestampMin': 1632936038461, 'timestampCurrent': 1632936074657, 'stepMax': 1111, 'stepMin': 569, 'stepCurrent': 1628, 'editable': False}, 
                      {'name': 'train/train_batch_support/class_04', 'valueMax': '10.0', 'valueMin': '0.0', 'valueCurrent': '1.0', 'timestampMax': 1632936038461, 'timestampMin': 1632936038461, 'timestampCurrent': 1632936074636, 'stepMax': 150, 'stepMin': 42, 'stepCurrent': 1627, 'editable': False}, 
                      {'name': 'train/train_batch_support/class_05', 'valueMax': '7.0', 'valueMin': '0.0', 'valueCurrent': '3.0', 'timestampMax': 1632936038461, 'timestampMin': 1632936038461, 'timestampCurrent': 1632936075114, 'stepMax': 17, 'stepMin': 32, 'stepCurrent': 1649, 'editable': False}, 
                      {'name': 'train/train_batch_support/class_06', 'valueMax': '8.0', 'valueMin': '0.0', 'valueCurrent': '3.0', 'timestampMax': 1632936038461, 'timestampMin': 1632936038461, 'timestampCurrent': 1632936075139, 'stepMax': 60, 'stepMin': 110, 'stepCurrent': 1650, 'editable': False}, 
                      {'name': 'train/train_batch_support/class_07', 'valueMax': '8.0', 'valueMin': '0.0', 'valueCurrent': '2.0', 'timestampMax': 1632936038461, 'timestampMin': 1632936038461, 'timestampCurrent': 1632936075180, 'stepMax': 42, 'stepMin': 115, 'stepCurrent': 1652, 'editable': False}, 
                      {'name': 'train/train_batch_support/class_08', 'valueMax': '10.0', 'valueMin': '0.0', 'valueCurrent': '5.0', 'timestampMax': 1632936038461, 'timestampMin': 1632936038461, 'timestampCurrent': 1632936075245, 'stepMax': 110, 'stepMin': 21, 'stepCurrent': 1655, 'editable': False}, 
                      {'name': 'train/train_batch_support/class_09', 'valueMax': '9.0', 'valueMin': '0.0', 'valueCurrent': '3.0', 'timestampMax': 1632936038461, 'timestampMin': 1632936038461, 'timestampCurrent': 1632936075224, 'stepMax': 55, 'stepMin': 27, 'stepCurrent': 1654, 'editable': False}]
    assert False
