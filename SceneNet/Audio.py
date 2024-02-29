import torch
import torchaudio
from torch import nn, optim
import torch.nn.init as init
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader, random_split
# Assuming BiGRUModel and EmbeddingDataset are defined as in previous examples
from Embedder import AudioEmbedder, embedder_main
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model
from Dataloader import LoadDF
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
            intra_epoch_interval = max(1, total_batches // 2)  # Determine interval for intra-epoch validation

            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                avg_val_loss,val_loss = self.validate()
                print(f'Epoch {epoch+1}/{self.num_epochs}, Step {batch_idx+1}/{total_batches}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')
                    
                # Check if this is the best model so far and save it
                if val_loss< best_val_loss:
                    best_val_loss = val_loss
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
        return avg_val_loss, loss.item()
    
    def save_training(self,args):
        name = f"training_lr{args.lr}_wd{args.wd}_layers{args.num_layers}_hidden{args.hidden_size}_batch{args.batch_size}_epochs{args.num_epochs}.txt"
        
        # Open the file for writing
        with open(name, "w") as f:
            # Write header
            f.write("Epoch,Training Loss,Validation Loss\n")
            
            # Write losses per epoch
            for epoch, (train_loss, val_loss) in enumerate(zip(self.training_losses, self.validation_losses), 1):
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f}\n")
        
        print(f"Training results saved to {name}")

    def get_majority_class_loss(self,dataloader):
        all_targets = []
        for _, target in dataloader:
            all_targets.append(target)
        all_targets = torch.cat(all_targets)  # Concatenate list of tensors into a single tensor

        predictions = torch.zeros_like(all_targets) # Always predicting 0 logits

        # Define the criterion
        criterion = nn.BCEWithLogitsLoss(reduction='mean')

        # Calculate the loss
        loss = criterion(predictions, all_targets)

        return loss
    
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
    def __init__(self, embed_size, hidden_size, num_layers, num_classes, with_pooling):
        super(AudioBiGRU, self).__init__()
        #input_size = tracks * (embed_dim_0 / pool_stride_0) * (embed_dim_1 / pool_stride_1)
        embed_dim_0, embed_dim_1 = embed_size
        pool_kernel = (2, 2)  # Pool size
        pool_stride = (2, 2)  # Assuming stride is the same as pool size

        # Calculate output dimensions after pooling
        output_dim_0 = (embed_dim_0 - pool_kernel[0]) // pool_stride[0] + 1
        output_dim_1 = (embed_dim_1 - pool_kernel[1]) // pool_stride[1] + 1

        # Calculate the flattened size assuming 2 tracks (channels)
        tracks = 2
        flattened_size = tracks * output_dim_0 * output_dim_1
        self.bigru = nn.GRU(flattened_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        # Initialize the pooling layer here to apply across embed_dim_0 and embed_dim_1

        self.with_pooling = with_pooling
        # Average pooling will find a more generale transition in sound instead of finding a single sign
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)  # Example values, adjust as needed
        self.initialize_weights()

    def initialize_weights(self):
        # Set a seed for reproducibility
        torch.manual_seed(42)

        # Iterate over all modules in your model
        for m in self.modules():
            if isinstance(m, nn.GRU):
                # GRU weights initialization
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        init.constant_(param, 0.01)  # Example: setting to a small constant
            elif isinstance(m, nn.Linear):
                init.constant_(m.weight, 0.01)  # Example: setting to a small constant
                if m.bias is not None:
                    init.constant_(m.bias, 0.01)
    def forward(self, x):

        batch_size, seq_length, tracks, embed_dim_0, embed_dim_1 = x.size()
        if(self.with_pooling):
            # Reshape to combine batch and seq_length dimensions to apply pooling independently per timestep and track
            x_reshaped = x.view(batch_size * seq_length * tracks, embed_dim_0, embed_dim_1)
            
            # Apply pooling - since the input to AvgPool2d should be (N, C, H, W), where N is the batch size,
            # C is the number of channels (in this case, 1, since we're pooling over the embedding dimensions),
            # H is embed_dim_0, and W is embed_dim_1, we need to add a dummy channel dimension
            x_reshaped = x_reshaped.unsqueeze(1)  # Add a channel dimension
            pooled_x = self.pool(x_reshaped)
            
            # Remove the dummy channel dimension and reshape back to original batch, seq_length, and tracks structure
            pooled_x = pooled_x.squeeze(1)  # Remove channel dimension
            pooled_x = pooled_x.view(batch_size, seq_length, tracks, -1)  # Reshape back
            
            # Flatten the last three dimensions to create a feature vector for each sequence element
            new_x = pooled_x.view(batch_size, seq_length, -1)
        else:
            batch_size, seq_length, tracks, embed_dim_0, embed_dim_1 = x.size()
            feature_size = tracks * embed_dim_0 * embed_dim_1

            # This is flattening the embedding with 2 tracks
            new_x = x.view(batch_size, seq_length, feature_size)
        out, _ = self.bigru(new_x)  # out: batch_size, seq_length, hidden_size*2 due to bidirectionality

        # Applying the fully connected layer to each timestep
        out = out.reshape(-1, out.shape[2]) # Flatten for FC layer
        out = self.fc(out)  # Now, out is of shape (batch_size*seq_length, num_classes)
        
        # Reshape out back to (batch_size, seq_length, num_classes) for sequence labeling tasks
        out = out.view(batch_size, seq_length, -1)
        
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
    data_folder="/Volumes/SeaGate/MoviesCMD"
    with open('SceneNet/config.json', 'r') as file:
        config = json.load(file)
    modality='audio'
    embed_type='pretrained'
    load_type="creating_and_writing" #[creating, creating_and_writing, loading]
    chunk_duration=3
    test_threshold=10
    resample_rate=16000
    # Generate all combinations of hyperparameters
    keys, values = zip(*config.items())
    print(keys, values)

    combinations = [dict(zip(keys, v)) for v in product(*values)]

    df = LoadDF(data_folder)  # Load your dataframe here
    embedder = AudioEmbedder()
    for combination in combinations:
        args = parse_args(config, combination)

        print(f"arguments for current run {args}")
        print("Start embedding audio streams... ")

        if(embed_type=="pretrained"):
            dataset, embed_size = embedder_main(data_folder,df ,modality, embedder, chunk_duration, load_type,resample_rate, test_threshold)

        elif(embed_type=="train_embeds"):
            processed_data = process_audio_and_create_labels(df['audio_path'])

            # Initialize your dataset with the processed data
            dataset = RawAudioDataset(processed_data)

            # Example: Access the first item in the dataset
            waveform_acc, waveform_voc, label = dataset[0]

        total_size = len(dataset)
        train_size = 16#int(total_size * 0.2)
        val_size = total_size - train_size
        print(f"Training with: {combination}")

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        print(f"batches {len(train_loader)}")
        model = AudioBiGRU(embed_size=embed_size, hidden_size=args.hidden_size, num_layers=args.num_layers, num_classes=1, with_pooling=True)
        
        # Set up the criterion and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        # Initialize and run the trainer
        trainer = AudioModelTrainer(model, train_loader, val_loader, criterion, optimizer, args.num_epochs)
        trainer.train()
        majority_loss=trainer.get_majority_class_loss(val_loader)
        print(f" Always predicting 0 would get validation loss: {majority_loss}")
        trainer.save_training(args)
        #trainer.report()