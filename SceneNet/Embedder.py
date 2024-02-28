import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model
import torchaudio
from torchaudio.transforms import Resample
import torchvision
from torchvision.models.video import r3d_18  # Example: using a pretrained R3D model
from torchvision.io import read_video
import tqdm
from torch.utils.data import Dataset, DataLoader, random_split  
import os
import time
import torch.nn.functional as F
import torchvision.models as models
import pandas as pd
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.io import read_video
from transformers import AutoModelForVideoClassification
import numpy as np


class AudioDataset(Dataset):
    def __init__(self, audio_file_paths, chunk_duration, resample_rate=16000):
        self.audio_file_paths = audio_file_paths
        self.chunk_duration = chunk_duration
        self.resample_rate = resample_rate

    def __len__(self):
        return len(self.audio_file_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_file_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        # Resample waveform if necessary
        if sample_rate != self.resample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.resample_rate)
            waveform = resampler(waveform)
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono
        return waveform, self.resample_rate
    
    
class AudioEmbedder:
    def __init__(self):
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h",output_hidden_states = True)

    def get_waveforms_all_clips(self, audio_df, resample_rate):
        accumulated_waveform_acc = torch.FloatTensor().reshape(0)  # Empty tensor for accompaniment
        accumulated_waveform_voc = torch.FloatTensor().reshape(0)  # Empty tensor for vocals
        part_boundaries = []  # To track start and end of each part in the concatenated waveform

        # Load and concatenate waveforms
        current_sample = 0  # Track current end sample of concatenated waveform
        for _, row in audio_df.iterrows():
            waveform_acc, sample_rate_acc = torchaudio.load(row['acc_path'])
            waveform_voc, sample_rate_voc = torchaudio.load(row['voc_path'])
            
            waveform_acc = torch.mean(waveform_acc, dim=0).unsqueeze(0)  # Ensure mono and keep batch dimension
            waveform_voc = torch.mean(waveform_voc, dim=0).unsqueeze(0)  # Ensure mono and keep batch dimension
            
            if sample_rate_acc != resample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate_acc, new_freq=resample_rate)
                waveform_acc = resampler(waveform_acc)
                
            if sample_rate_voc != resample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate_voc, new_freq=resample_rate)
                waveform_voc = resampler(waveform_voc)
            
            part_length = waveform_acc.size(1)
            accumulated_waveform_acc = torch.cat((accumulated_waveform_acc, waveform_acc), dim=1)
            accumulated_waveform_voc = torch.cat((accumulated_waveform_voc, waveform_voc), dim=1)
            
            # Update part boundaries
            part_boundaries.append((current_sample, current_sample + part_length))
            current_sample += part_length
        return accumulated_waveform_acc, accumulated_waveform_voc,part_boundaries
    
    def process_audio(self, audio_df, chunk_duration, intro_outro_duration=6, resample_rate=16000):
        accumulated_waveform_acc, accumulated_waveform_voc, part_boundaries =self.get_waveforms_all_clips(audio_df,resample_rate)
        # Chunk and process waveforms
        embeddings_list = []
        labels_list = []
        chunk_size = int(chunk_duration * resample_rate)
        total_samples = accumulated_waveform_acc.size(1)
        num_chunks = (total_samples + chunk_size - 1) // chunk_size  # Ceiling division

        for i in range(num_chunks):
            start_sample = i * chunk_size
            end_sample = min(start_sample + chunk_size, total_samples)
            chunk_waveform_acc = accumulated_waveform_acc[:, start_sample:end_sample]
            chunk_waveform_voc = accumulated_waveform_voc[:, start_sample:end_sample]
            # Check if the last chunk needs padding
            if chunk_waveform_acc.size(1) < chunk_size:
                # Calculate padding size
                padding_size = chunk_size - chunk_waveform_acc.size(1)
                # Pad the waveform
                chunk_waveform_acc = F.pad(chunk_waveform_acc, (0, padding_size), "constant", 0)
                chunk_waveform_voc = F.pad(chunk_waveform_voc, (0, padding_size), "constant", 0)

            # Assuming audio_processor and audio_model are set to handle the chunks
            # Replace process_chunk() with your actual processing logic
            embeddings_acc = self.process_chunk(chunk_waveform_acc.squeeze(0))  
            embeddings_voc = self.process_chunk(chunk_waveform_voc.squeeze(0))  

            embeddings = torch.cat((embeddings_acc, embeddings_voc), dim=0)
            embeddings_list.append(embeddings)

            # Label creation
            label = 0
            for (part_start, part_end) in part_boundaries:
                if start_sample <= part_start < end_sample or start_sample < part_end <= end_sample:
                    label = 1  # This chunk overlaps with an intro or outro
                    break
            labels_list.append(label)
        labels_list = self.widen_transition(labels_list,margin=1)
        # Convert lists to tensors
        embeddings_tensor = torch.stack(embeddings_list, dim=0)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        
        return embeddings_tensor, labels_tensor
    import torch


    def widen_transition(self, labels_list, margin):
        adjusted_labels_list = labels_list[:]  # Create a copy to preserve original labels during iteration
        for i, label in enumerate(labels_list):
            if label == 1:
                # Set preceding label to 1 if it exists
                if i > margin:
                    adjusted_labels_list[i-margin] = 1
                # Set next label to 1 if it exists
                if i < len(labels_list) - margin:
                    adjusted_labels_list[i+margin] = 1
        return adjusted_labels_list
    
    def process_chunk(self,chunk_waveform, resample_rate=16000):
        inputs = self.audio_processor(chunk_waveform, sampling_rate=resample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings= self.audio_model(**inputs).hidden_states[-1]
        return embeddings
    
class VideoEmbedder:
    def __init__(self, model_name='resnet18', frame_rate=1, resize_shape=(224, 224), intro_outro_duration=5):
        self.model = getattr(models, model_name)(pretrained=True)
        self.frame_rate = frame_rate
        self.transform = Compose([
            Resize(resize_shape),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.intro_outro_duration = intro_outro_duration

    def process_video(self, video_paths):
        embeddings_list = []
        labels_list = []
        for path in video_paths:
            # Read video
            vframes, _, info = read_video(path, pts_unit='sec')
            
            # Calculate total video duration and frame count
            total_frames = vframes.shape[0]
            video_duration = info['duration']
            
            # Determine frames per second from the video info
            fps = total_frames / video_duration
            
            # Sample frames at the specified frame rate
            sampled_frames = vframes[::int(fps // self.frame_rate)]
            
            # Transform frames
            transformed_frames = torch.stack([self.transform(frame) for frame in sampled_frames])
            
            # Reshape to (num_frames, C, H, W) for the model
            transformed_frames = transformed_frames.permute(0, 3, 1, 2)
            
            # Process frames through the model
            with torch.no_grad():
                outputs = self.model(transformed_frames)
                embeddings = outputs.last_hidden_state  # Adjust based on your model's output
            
            # Pool the embeddings, if necessary
            pooled_embeddings = torch.mean(embeddings, dim=0)
            embeddings_list.append(pooled_embeddings)
            
            # Calculate the number of embeddings for the first and last 5 seconds
            num_intro_outro_frames = int(self.intro_outro_duration * self.frame_rate)
            num_embeddings = transformed_frames.size(0)
            labels = torch.zeros(num_embeddings)
            
            # Label the first and last 5 seconds (or the corresponding number of frames) as 1
            labels[:num_intro_outro_frames] = 1
            labels[-num_intro_outro_frames:] = 1 if num_intro_outro_frames < num_embeddings else 0  # Adjust if video is very short
            
            labels_list.append(labels)
        
        # Concatenate the embeddings and labels for all video paths
        combined_embeddings = torch.cat(embeddings_list, dim=0)
        combined_labels = torch.cat(labels_list, dim=0)
        print(combined_embeddings, combined_labels)
        return combined_embeddings, combined_labels
    
class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def pad_sequences_and_labels(embeddings, labels, target_seq_length):
    # embeddings shape: [seq_length, tracks, size_embedding[0], size_embedding[1]]
    # labels shape: [seq_length]
    
    current_seq_length = embeddings.shape[0]
    padding_needed = target_seq_length - current_seq_length
    
    # Pad embeddings
    if padding_needed > 0:
        pad_tensor_embeddings = torch.zeros((padding_needed, *embeddings.shape[1:]), dtype=embeddings.dtype)
        padded_embeddings = torch.cat([embeddings, pad_tensor_embeddings], dim=0)
    else:
        padded_embeddings = embeddings
    
    # Pad labels
    if padding_needed > 0:
        # Assuming -1 as the ignore_index for labels
        pad_tensor_labels = torch.full((padding_needed,), -1, dtype=labels.dtype)
        padded_labels = torch.cat([labels, pad_tensor_labels], dim=0)
    else:
        padded_labels = labels
    
    return padded_embeddings, padded_labels
def write_embeddings_to_file(data_folder,mov_embeddings,mov_labels, key, chunk_duration):
    # Convert embeddings and labels to NumPy arrays for saving (if not already)
    mov_embeddings_np = np.array(mov_embeddings)
    mov_labels_np = np.array(mov_labels)
    
    # Define file paths (customize as needed)
    embeddings_file_path = f'{data_folder}/embeddings/movie_{key}_chunk:{chunk_duration}_embeddings.npy'
    labels_file_path = f'{data_folder}/embeddings/movie_{key}_chunk:{chunk_duration}_labels.csv'
    
    # Save embeddings to a binary file (.npy format)
    np.save(embeddings_file_path, mov_embeddings_np)
    
    # Save labels to a CSV file
    pd.DataFrame(mov_labels_np).to_csv(labels_file_path, index=False, header=False)
    
    print(f'Saved embeddings to {embeddings_file_path} and labels to {labels_file_path}')
    return
def load_embeddings_from_file(data_folder,key, chunk_duration):
    # Define file paths (must match those used in write_data)
    embeddings_file_path = f'/{data_folder}/embeddings/movie_{key}_chunk:{chunk_duration}_embeddings.npy'
    labels_file_path = f'{data_folder}/embeddings/movie_{key}_chunk:{chunk_duration}_labels.csv'
    
    # Load embeddings from the .npy file
    mov_embeddings = np.load(embeddings_file_path, allow_pickle=True)
    mov_labels = np.loadtxt(labels_file_path, delimiter=',')

    mov_embeddings=torch.from_numpy(mov_embeddings)
    mov_labels=torch.from_numpy(mov_labels)

    return mov_embeddings, mov_labels

def embedder_main(data_folder,df,modality, embedder, chunk_duration=3, load_type="loading", resample_rate=16000, test_threshold=100):
    embeddings_list = []
    labels_list = []
    count=0
    intro_outro_duration=3
    reverse_group_keys = list(df.groups.keys())[::-1]

    for key in reverse_group_keys:

        group_df = df.get_group(key)
        count+=1

        print(f'Movie {key} number {count}/{test_threshold} loading....')
        # Set maximum number of movies to load
        if(count>test_threshold):
            print(f"Testing for {count} movies")
            break

        elif(modality=='audio'):
            if(load_type=="loading"):
                mov_embeddings, mov_labels = load_embeddings_from_file(data_folder,key, chunk_duration)
            elif(load_type=="creating"):
                mov_embeddings, mov_labels = embedder.process_audio(group_df, chunk_duration, intro_outro_duration, resample_rate)
            elif(load_type=="creating_and_writing"):
                labels_file_path = f'{data_folder}/embeddings/movie_{key}_chunk:{chunk_duration}_labels.csv'
                # if the movie embeddings with cunk durations do already exist do not create and write
                if os.path.exists(labels_file_path):
                    count -=1
                    continue
                mov_embeddings, mov_labels = embedder.process_audio(data_folder,group_df, chunk_duration, intro_outro_duration, resample_rate)
                write_embeddings_to_file(mov_embeddings,mov_labels,key,chunk_duration)
            else:
                print(f"Specify how you want to retrieve embeddings data with: load_type={load_type} as either loading, creating or creating_and_writing")
        elif(modality=='video'):
            embeddings, labels = embedder.process_video(group_df)
        else:
            print("Modality given to function load_dataset() must either be video or audio")
            break

        embeddings_tensor,labels_tensor= pad_sequences_and_labels(mov_embeddings,mov_labels, target_seq_length=1000)
        embeddings_list.append(embeddings_tensor)
        labels_list.append(labels_tensor)
    embeddings_tensor = torch.stack(embeddings_list)
    labels_tensor = torch.stack(labels_list)
    labels_tensor = labels_tensor.unsqueeze(-1)
    batch_size, seq_length, tracks, embed_dim_0, embed_dim_1 = embeddings_tensor.size()

    return EmbeddingsDataset(embeddings_tensor, labels_tensor), (embed_dim_0,embed_dim_1)