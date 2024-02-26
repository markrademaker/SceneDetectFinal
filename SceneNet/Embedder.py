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
def load_metadata(meta_path):
    """
    Load all metadata from CSV files into pandas DataFrames.
    """
    casts_df = pd.read_csv(os.path.join(meta_path, 'casts.csv'))
    clips_df = pd.read_csv(os.path.join(meta_path, 'clips.csv'))
    descriptions_df = pd.read_csv(os.path.join(meta_path, 'descriptions.csv'))
    durations_df = pd.read_csv(os.path.join(meta_path, 'durations.csv'))
    movies_df = pd.read_csv(os.path.join(meta_path, 'movies.csv'))
    movie_info_df = pd.read_csv(os.path.join(meta_path, 'movie_info.csv'))
    split_df = pd.read_csv(os.path.join(meta_path, 'split.csv'))
    return casts_df, clips_df, descriptions_df, durations_df, movies_df, movie_info_df, split_df

def merge_dataframes(*dfs):
    """
    Merge multiple pandas DataFrames on 'imdbid' and return the merged DataFrame.
    """
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='imdbid', how='outer')
    return merged_df

def load_video_files(folder_path):
    """
    Load video file paths from the specified directory into a pandas DataFrame.
    """
    data_from_files = []
    for folder in os.listdir(folder_path):
        if folder.startswith('.') or folder.endswith("Store"):
            continue
        print(f"Year {folder} movies loading..")
        subfolder_path = os.path.join(folder_path, folder)
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            if os.path.isfile(file_path) and file_path.endswith(".mkv"):
                data_from_files.append({
                    'videoid': file[:-4],
                    'path': file_path,
                    'audio_path': file_path[:-4] + "_audio.opus",
                    'acc_path': file_path[:-4] + "_audio/accompaniment.wav",
                    'voc_path': file_path[:-4] + "_audio/vocals.wav",
                    'video_path': file_path[:-4] + "_video.xxx",
                })
    return pd.DataFrame(data_from_files)

def LoadDF():
    folder_path = '/Volumes/SeaGate/MoviesCMD/videos'
    meta_path = "/Users/markrademaker/Downloads/Work/Scriptie/Code/CondensedMovies-master/data/metadata"
    
    # Load metadata
    casts_df, clips_df, descriptions_df, _, movies_df, movie_info_df, split_df = load_metadata(meta_path)
    
    # Merge metadata DataFrames
    merged_df = merge_dataframes(movies_df, movie_info_df, casts_df, split_df)
    
    # Load video files
    files_df = load_video_files(folder_path)
    
    # Merge video files DataFrame with metadata
    video_df = pd.merge(clips_df, descriptions_df, on=['videoid','imdbid'], how='outer')
    video_df = pd.merge(video_df, merged_df, on='imdbid', how='outer')
    video_df = pd.merge(video_df, files_df, on='videoid', how='outer')
    
    # Filter rows where 'path' is not NA
    filtered_video_df = video_df[video_df['path'].notna()]
    # Step 1: Ensure data types are correct (if necessary)
    filtered_video_df = filtered_video_df.dropna(subset=['clip_idx'])

    filtered_video_df['clip_idx'] = filtered_video_df['clip_idx'].astype(int)  # Example if conversion is needed

    # Step 2: Sort by 'imdbid' and then by 'clip_idx' in ascending order
    sorted_video_df = filtered_video_df.sort_values(by=['imdbid', 'clip_idx'], ascending=[True, True])

    # Step 3: Group by 'imdbid' (if you're performing some grouped operations afterwards)
    grouped = sorted_video_df.groupby('imdbid')
    return grouped


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
    
    def process_audio(self, audio_df, chunk_duration, intro_outro_duration=6, resample_rate=16000):
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

        # Convert lists to tensors
        embeddings_tensor = torch.stack(embeddings_list, dim=0)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        print(embeddings_tensor, labels_tensor)
        return embeddings_tensor, labels_tensor
    
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
    
def load_dataset(df,modality, embedder, chunk_duration=3, resample_rate=16000):
    embeddings_list = []
    labels_list = []
    count=0
    test_threshold=10
    for group_name, group_df in tqdm.tqdm(df, total=df.ngroups):
        count+=1
        if(count>test_threshold):
            print(f"Testing for {count} movies")
            break
        for _, row in group_df.iterrows():
            if(modality=='audio'):
                mov_embeddings, mov_labels = embedder.process_audio(group_df, chunk_duration, intro_outro_duration=3, resample_rate=16000)
            elif(modality=='video'):
                embeddings, labels = embedder.process_video(row['video_path'])
            else:
                print("Modality given to function load_dataset() must either be video or audio")
                break

            embeddings_list.append(mov_embeddings)
            labels_list.append(mov_labels)

    embeddings_tensor = torch.tensor(embeddings_list, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    return EmbeddingsDataset(embeddings_tensor, labels_tensor), embeddings_tensor.size(-1)