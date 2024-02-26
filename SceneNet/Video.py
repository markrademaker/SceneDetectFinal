import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader, random_split
# Assuming BiGRUModel and EmbeddingDataset are defined as in previous examples
from Embedder import VideoEmbedder, LoadDF, load_dataset
import torchvision.models as models
import tqdm
import copy
import matplotlib.pyplot as plt
import json
from itertools import product
from argparse import ArgumentParser

class VideoModelTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, num_epochs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.training_losses = []  # Store average training loss per epoch
        self.validation_losses = []  # Store average validation loss per epoch

    def train(self):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_val_loss = float('inf')

        # Initialize lists to store loss values for every report
        all_training_losses = []
        all_validation_losses = []

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            total_batches = len(self.train_loader)
            intra_epoch_interval = max(1, total_batches // 5)  # Determine interval for intra-epoch validation

            for batch_idx, (inputs, labels) in enumerate(self.train_loader, start=1):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                # Intra-Epoch Validation Check
                if batch_idx % intra_epoch_interval == 0 or batch_idx == total_batches:
                    current_train_loss = train_loss / batch_idx
                    avg_val_loss = self.validate()  # Assuming self.validate() computes the validation loss
                    
                    # Store the current training and validation loss
                    all_training_losses.append(current_train_loss)
                    all_validation_losses.append(avg_val_loss)

                    print(f'Epoch {epoch+1}/{self.num_epochs}, Step {batch_idx}/{total_batches}, Training Loss: {current_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
                    
                    # Check if this is the best model so far and save it
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        torch.save(self.model.state_dict(), 'best_model.pth')  # Save the best model
            # At the end of training, you can choose to load the best model weights
            self.model.load_state_dict(best_model_wts)
        # Optionally, you can return the best model and the loss lists
        return self.model, all_training_losses, all_validation_losses

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(self.val_loader)
        return avg_val_loss
    
    def save_training(self):
        name = f"training_lr{self.optimizer.learning_rate}_wd{self.optimizer.weight_decay}_layers{self.model.num_layers}_epochs{self.model.hidden_size}.txt"
        
        # Open the file for writing
        with open(name, "w") as f:
            # Write header
            f.write("Epoch,Training Loss,Validation Loss\n")
            
            # Write losses per epoch
            for epoch, (train_loss, val_loss) in enumerate(zip(self.training_losses, self.validation_losses), 1):
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f}\n")
        
        print(f"Training results saved to {name}")
    def report(self):
        """Plot the training and validation loss."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses, label='Training Loss')
        plt.plot(self.validation_losses, label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

class VideoCNNBiGRU(nn.Module):
    def __init__(self,  hidden_size, num_layers, num_classes,pretrained_cnn='resnet18'):
        super(VideoCNNBiGRU, self).__init__()
        # Load a pretrained CNN and replace the top layer for feature extraction
        self.cnn = self._prepare_cnn(pretrained_cnn)
        cnn_output_size = self.cnn.fc.in_features  # Assuming the CNN has a fully connected layer at the end
        
        # BiGRU for temporal dynamics
        self.bigru = nn.GRU(cnn_output_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
    
    def _prepare_cnn(self, model_name):
        """Prepare the CNN for feature extraction."""
        model = getattr(models, model_name)(pretrained=True)
        # Remove the top layer (fully connected) to use as a feature extractor
        model.fc = nn.Identity()
        return model
    
    def forward(self, x):
        # x: batch, seq, C, H, W
        batch_size, seq_length, C, H, W = x.size()
        # Reshape x to process each frame through the CNN
        c_in = x.view(batch_size * seq_length, C, H, W)
        c_out = self.cnn(c_in)
        # Reshape back to (batch, seq, feature)
        r_in = c_out.view(batch_size, seq_length, -1)
        # Process through BiGRU
        r_out, _ = self.bigru(r_in)
        # Take the output of the last time step
        r_out = r_out[:, -1, :]
        out = self.fc(r_out)
        return out
    
class VideoBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(VideoBiGRU, self).__init__()
        self.bigru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        out, _ = self.bigru(x)  # out: batch, seq, hidden*2
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)
        return out
    
def parse_args(config, combination):
    parser = ArgumentParser(description='Train a BiGRU model on audio embeddings.')
    for key, value in combination.items():
        parser.add_argument(f'--{key}', type=type(value), default=value, help=f'{key}: {value}')
    return parser.parse_args([])

if __name__ == "__main__":
    # Load configuration file
    with open('/Users/markrademaker/Downloads/Work/Scriptie/Code/SceneNet/config.json', 'r') as file:
        config = json.load(file)
    
    modality='video'
    # Generate all combinations of hyperparameters
    keys, values = zip(*config.items())
    print(keys, values)
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    df = LoadDF()  # Load your dataframe here
    embedder = VideoEmbedder()
    for combination in combinations:
        args = parse_args(config, combination)

        dataset, tensor_size = load_dataset(df,modality, embedder, chunk_duration=1, resample_rate=16000)
        total_size = len(dataset)
        train_size = int(total_size * 0.9)
        val_size = total_size - train_size
        print(f"Training with: {combination}")
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        model = VideoBiGRU(pretrained_cnn='resnet18', hidden_size=args.hidden_size, num_layers=args.num_layers, num_classes=1)

        # Set up the criterion and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        # Initialize and run the trainer
        trainer = VideoModelTrainer(model, train_loader, val_loader, criterion, optimizer, args.num_epochs)
        trainer.train()
        trainer.save_training()
        trainer.report()