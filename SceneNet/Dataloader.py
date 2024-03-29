import pandas as pd
import os
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
    id_file_path = folder_path + "/scene_id.txt"
    with open(id_file_path, 'r') as f:
        video_ids = f.read().splitlines()
    for video_id in video_ids:
        file_path = folder_path +"/videos/"+ video_id
        data_from_files.append({
            'videoid': file_path,
            'path': file_path+".mkv",
            'audio_path': file_path + "_audio.opus",
            'acc_path': file_path + "_audio/accompaniment.wav",
            'voc_path': file_path + "_audio/vocals.wav",
            'video_path': file_path+ "_video.xxx",
        })
    print(len(data_from_files))
    pd.DataFrame(data_from_files)
    return pd.DataFrame(data_from_files)

def LoadDF(data_folder):
    folder_path = data_folder+'/videos'
    meta_path = data_folder+"/data/metadata"
    
    # Load metadata
    casts_df, clips_df, descriptions_df, _, movies_df, movie_info_df, split_df = load_metadata(meta_path)
    
    # Merge metadata DataFrames
    merged_df = merge_dataframes(movies_df, movie_info_df, casts_df, split_df)
    
    # Load video files
    files_df = load_video_files(data_folder)
    
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
