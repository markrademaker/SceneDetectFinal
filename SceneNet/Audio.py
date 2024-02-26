import torch
import torchaudio
from torch import nn, optim
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader, random_split
# Assuming BiGRUModel and EmbeddingDataset are defined as in previous examples
from Embedder import AudioEmbedder, LoadDF, load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model
import tqdm
import copy
import matplotlib.pyplot as plt
import json
from itertools import product
from argparse import ArgumentParser
#from YAMnet import YAMnetClassifier
class AudioModelTrainer:
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

class AudioBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AudioBiGRU, self).__init__()
        self.bigru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        out, _ = self.bigru(x)  # out: batch, seq, hidden*2
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)
        return out
    
class FineTuneAudioBiGRU(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes, embedder_model_name="facebook/wav2vec2-base-960h"):
        super(FineTuneAudioBiGRU, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(embedder_model_name)
        self.audio_model = Wav2Vec2ForCTC.from_pretrained(embedder_model_name)

        # Dynamically determine input_size from the embedding model's output features
        self.input_size = self.audio_model.config.hidden_size
        
        self.bigru = nn.GRU(self.input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, waveform, sampling_rate=16000):
        # Process waveform with the audio processor
        inputs = self.processor(waveform.squeeze(0), sampling_rate=sampling_rate, return_tensors="pt", padding=True, truncation=True).input_values
        inputs = inputs.to(waveform.device)

        # Generate embeddings
        embeddings = self.audio_model(inputs).last_hidden_state

        # Since all sequences for a given batch are of equal length, no need to pack
        output, _ = self.bigru(embeddings)

        # Assuming you want to aggregate information across the entire sequence
        # You might consider pooling or selecting a representative vector (e.g., last timestep)
        # Here, we simply apply the FC layer to the output of the last time step
        out = self.fc(output[:, -1, :])

        return out
        
class RawAudioDataset(Dataset):
    def __init__(self, processed_data):
        """
        Args:
            processed_data (list of tuples): Each tuple contains a pair of waveforms 
            (accompaniment, vocals) and a label.
        """
        self.processed_data = processed_data

    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        # Extract accompaniment and vocals waveforms, and label for the given index
        (waveform_acc, waveform_voc), label = self.processed_data[idx]
        
        # Your model will need to handle these inputs accordingly.
        # This returns a tuple of accompaniment waveform, vocals waveform, and label.
        return waveform_acc, waveform_voc, label
    
def process_audio_and_create_labels(audio_paths,dt=2, intro_outro_duration=5):
    processed_data = []  # List to store tuples of (segmented waveform, label)
    resample_rate=16000
    for path in audio_paths:
        acc_path = path+"/accompaniment.wav"
        voc_path = path+"/vocals.wav"

        waveform_acc, sample_rate_acc = torchaudio.load(acc_path)
        waveform_voc, sample_rate_voc = torchaudio.load(voc_path)

        if sample_rate_acc != resample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate_acc, new_freq=resample_rate)
            waveform_acc = resampler(waveform_acc)
        if sample_rate_voc != resample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate_voc, new_freq=resample_rate)
            waveform_voc = resampler(waveform_voc)

        # If nr  channels >1 take average, converts to mono
        waveform_acc = torch.mean(waveform_acc, dim=0, keepdim=True)
        waveform_voc = torch.mean(waveform_voc, dim=0, keepdim=True)

        assert waveform_voc==waveform_acc

        # Calculate the number of samples per dt seconds
        samples_per_dt = int(resample_rate * dt)
        total_samples = waveform_voc.size(1)
        num_segments = int(total_samples / samples_per_dt)  # Adjust to handle partial segments

        # Create labels, defaulting to 0 except for the first and last intro_outro_duration seconds
        labels = torch.zeros(num_segments)
        intro_outro_segments = int(intro_outro_duration / dt)
        
        labels[:intro_outro_segments] = 1
        labels[-intro_outro_segments:] = 1 if intro_outro_segments < num_segments else 0  # Check needed if audio is very short

        # Segment the waveform into chunks of dt seconds
        for i in range(num_segments):
            start_sample = i * samples_per_dt
            end_sample = start_sample + samples_per_dt
            segment_voc = waveform_voc[:, start_sample:end_sample]
            segment_acc = waveform_acc[:, start_sample:end_sample]
            # Check if the segment is less than dt seconds, pad it if necessary
            if segment_acc.size(1) < samples_per_dt:
                padding = samples_per_dt - segment_acc.size(1)
                segment_acc = torch.nn.functional.pad(segment_acc, (0, padding))
                segment_voc = torch.nn.functional.pad(segment_voc, (0, padding))
            label = labels[i]
            processed_data.append(((segment_acc, segment_voc), torch.tensor(label, dtype=torch.float)))

    return processed_data

def yamnet(audio_paths):
    yamnet = YAMnetClassifier()
    yamnet_embeddings=[]
    for path in audio_paths:
        audio_data = path
        yamnet_predictions = yamnet.predict(audio_data)
        yamnet_class_input = yamnet.process(audio_data)
        yamnet_embedding = yamnet.get_embedding(audio_data)
        yamnet_embeddings.append(yamnet_embedding)
    return yamnet_embeddings    

def parse_args(config, combination):
    parser = ArgumentParser(description='Train a BiGRU model on audio embeddings.')
    for key, value in combination.items():
        parser.add_argument(f'--{key}', type=type(value), default=value, help=f'{key}: {value}')
    return parser.parse_args([])


if __name__ == "__main__":
    # Load configuration file
    with open('/Users/markrademaker/Downloads/Work/Scriptie/Code/SceneNet/config.json', 'r') as file:
        config = json.load(file)
    modality='audio'
    embed_type='pretrained'

    # Generate all combinations of hyperparameters
    keys, values = zip(*config.items())
    print(keys, values)
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    df = LoadDF()  # Load your dataframe here
    embedder = AudioEmbedder()
    for combination in combinations:
        args = parse_args(config, combination)
        print("Start embedding audio streams... ")
        if(embed_type=="pretrained"):
            dataset, tensor_size = load_dataset(df,modality, embedder, chunk_duration=3, resample_rate=16000)
        elif(embed_type=="raw"):
            processed_data = process_audio_and_create_labels(df['audio_path'])

            # Initialize your dataset with the processed data
            dataset = RawAudioDataset(processed_data)

            # Example: Access the first item in the dataset
            waveform_acc, waveform_voc, label = dataset[0]
            print(waveform_acc,waveform_voc)
        total_size = len(dataset)
        train_size = int(total_size * 0.9)
        val_size = total_size - train_size
        print(f"Training with: {combination}")
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        model = AudioBiGRU(input_size=tensor_size, hidden_size=args.hidden_size, num_layers=args.num_layers, num_classes=1)

        # Set up the criterion and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        # Initialize and run the trainer
        trainer = AudioModelTrainer(model, train_loader, val_loader, criterion, optimizer, args.num_epochs)
        trainer.train()
        trainer.save_training()
        trainer.report()