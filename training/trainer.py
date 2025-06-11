import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import matplotlib.pyplot as plt

from IPython import get_ipython
if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def collate_fn(batch):
    return {
        "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long),
        "attention_mask": torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long),
        "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
    }


class SimpleTrainer:
    def __init__(self, model, train_dataset, val_dataset=None, batch_size=32, learning_rate=5e-5, num_epochs=3, device=None, eval_strategy="batch", eval_interval=1000):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.current_epoch = 1
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn) if val_dataset else None

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        self.eval_strategy = eval_strategy
        self.eval_interval = eval_interval

        self.best_val_loss = float("inf")
        self.best_model_state_dict = None

        self.batch_training_log_history = []
        self.batch_eval_log_history = []

    def train(self):
        for epoch in range(self.current_epoch, self.current_epoch + self.num_epochs):
            accumulated_training_lost = 0
            for batch_count, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
                self.model.train()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
                logits = outputs["logits"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                accumulated_training_lost += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                step_correct = (predictions == labels).sum().item()
                step_total = labels.size(0)
                step_acc = step_correct / step_total

                if (
                    self.eval_strategy == "batch"
                    and self.val_loader is not None
                    and (batch_count + 1) % self.eval_interval == 0
                ):
                    tqdm.write(f"[Step {batch_count + 1}] Training - loss: {accumulated_training_lost:.4f}, accuracy: {step_acc:.4f}")
                    self.batch_training_log_history.append({
                        "epoch": epoch,
                        "step": batch_count + 1,
                        "train_loss": accumulated_training_lost,
                        "train_accuracy": step_acc
                    })
                    self.evaluate(epoch, step=batch_count + 1)
                    accumulated_training_lost = 0
        self.current_epoch += self.num_epochs

    def evaluate(self, epoch, step=None):
        if self.val_dataset is None:
            raise self.ValSetNotProvidedError("Validation set not found.")
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.inference_mode():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
                logits = outputs["logits"]
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0

        if step is not None:
            tqdm.write(f"[Step {step}] Validation - loss: {total_loss:.4f}, accuracy: {accuracy:.4f}")
            self.batch_eval_log_history.append({
                "epoch": epoch,
                "step": step,
                "eval_loss": total_loss,
                "eval_accuracy": accuracy
            })
        else:
            tqdm.write(f"Validation - loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}\n")

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.best_model_state_dict = self.model.state_dict()
            tqdm.write("New best model saved")

    def save_best_model(self, path="../models/best_model.pth"):
        if self.best_model_state_dict is not None:
            torch.save(self.best_model_state_dict, path)
            print(f"Best model saved to {path}")
        else:
            print("No model was saved during training.")

    def plot_train_loss(self):
        train_steps = [f"{entry['epoch']}-{entry['step']}" for entry in self.batch_training_log_history]
        train_loss = [entry["train_loss"] for entry in self.batch_training_log_history]

        plt.figure(figsize=(10, 4))
        plt.plot(train_steps, train_loss, label="Train Loss", color="tab:blue")
        plt.xlabel("Epoch-Step")
        plt.ylabel("Loss")
        plt.title("Train Loss over Steps")
        plt.xticks(rotation=45, fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_eval_loss(self):
        eval_steps = [f"{entry['epoch']}-{entry['step']}" for entry in self.batch_eval_log_history]
        eval_loss = [entry["eval_loss"] for entry in self.batch_eval_log_history]

        plt.figure(figsize=(10, 4))
        plt.plot(eval_steps, eval_loss, label="Eval Loss", color="tab:orange")
        plt.xlabel("Epoch-Step")
        plt.ylabel("Loss")
        plt.title("Eval Loss over Steps")
        plt.xticks(rotation=45, fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_loss(self):
        train_steps = [f"{entry['epoch']}-{entry['step']}" for entry in self.batch_training_log_history]
        train_loss = [entry["train_loss"] for entry in self.batch_training_log_history]
        eval_steps = [f"{entry['epoch']}-{entry['step']}" for entry in self.batch_eval_log_history]
        eval_loss = [entry["eval_loss"] for entry in self.batch_eval_log_history]

        plt.figure(figsize=(10, 4))
        plt.plot(train_steps, train_loss, label="Train Loss", color="tab:blue", marker="o")
        plt.plot(eval_steps, eval_loss, label="Eval Loss", color="tab:orange", marker="x")
        plt.xlabel("Epoch-Step")
        plt.ylabel("Loss")
        plt.title("Train & Eval Loss over Steps")
        plt.xticks(rotation=45, fontsize=8)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_accuracy(self):
        train_steps = [f"{entry['epoch']}-{entry['step']}" for entry in self.batch_training_log_history]
        train_acc = [entry["train_accuracy"] for entry in self.batch_training_log_history]
        eval_steps = [f"{entry['epoch']}-{entry['step']}" for entry in self.batch_eval_log_history]
        eval_acc = [entry["eval_accuracy"] for entry in self.batch_eval_log_history]

        plt.figure(figsize=(10, 5))
        plt.plot(train_steps, train_acc, label="Train Accuracy", marker="o")
        plt.plot(eval_steps, eval_acc, label="Eval Accuracy", marker="x")
        plt.xlabel("Epoch-Step")
        plt.ylabel("Accuracy")
        plt.title("Train & Eval Accuracy over Steps")
        plt.xticks(rotation=45, fontsize=8)
        plt.ylim(0, 1)  # 因为 accuracy ∈ [0, 1]
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_training_log(self) -> None:
        self.plot_loss()
        self.plot_accuracy()

    def set_num_epochs(self, num_epochs: int) -> None:
        self.num_epochs = num_epochs

    class ValSetNotProvidedError(Exception):
        pass
