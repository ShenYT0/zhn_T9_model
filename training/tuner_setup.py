import tensorflow as tf
from kerastuner import HyperModel, HyperParameters
from kerastuner.tuners import RandomSearch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from bert_next_token_model import BertNextTokenModel


class TransformerHyperModel(HyperModel):
    def __init__(self):
        super().__init__()
        self.batch_size = None

    def build(self, hp):
        hp.Choice('embed_dim', [64, 128, 256])
        hp.Choice('num_heads', [2, 4, 8])
        hp.Choice('ff_dim', [128, 256, 512])
        hp.Int('num_layers', 1, 4)
        hp.Choice('batch_size', [16, 32, 64])
        return tf.keras.Sequential()  # Dummy model for KerasTuner


def train_model(model, train_loader, val_loader, device, epochs=5, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return evaluate_model(model, val_loader, device)


def evaluate_model(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


def run_tuning(x_train, y_train, x_val, y_val, vocab_size):
    hypermodel = TransformerHyperModel()
    hypermodel.build(HyperParameters())  # register all hps

    tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='bert_t9_tune'
    )

    def score_fn(hp):
        embed_dim = hp.get('embed_dim')
        num_heads = hp.get('num_heads')
        ff_dim = hp.get('ff_dim')
        num_layers = hp.get('num_layers')
        batch_size = hp.get('batch_size')

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)

        model = BertNextTokenModel(vocab_size, embed_dim, num_heads, ff_dim, num_layers)
        accuracy = train_model(model, train_loader, val_loader, device, epochs=5)
        return accuracy

    tuner.search_space_summary()
    for trial in tuner.oracle.trials.values():
        trial.score = score_fn(trial.hyperparameters)
        tuner.oracle.update_trial(trial.trial_id, {'val_accuracy': trial.score})

    best_hps = tuner.get_best_hyperparameters(1)[0]
    print("Best hyperparameters found:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")

    best_batch_size = best_hps.get("batch_size")
    best_model = BertNextTokenModel(
        vocab_size,
        best_hps.get("embed_dim"),
        best_hps.get("num_heads"),
        best_hps.get("ff_dim"),
        best_hps.get("num_layers")
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=best_batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=best_batch_size)
    train_model(best_model, train_loader, val_loader, device, epochs=5)
    return best_model
