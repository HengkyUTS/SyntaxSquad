import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from IPython.display import clear_output
from video2landmarks import VideoLandmarksExtractor
from clearml import Task, Dataset


class WLASLLandmarksExtractor(VideoLandmarksExtractor): # Initialize the WLASLVideoProcessor with directories.
    def __init__(self, clearml_raw_dataset_id: str = '921fdb13ed94464ebcf0dd0586856a5c', **kwargs: dict):
        super().__init__(**kwargs)
        self.task = Task.init(project_name='SyntaxSquad', task_name='step0_landmarks_extraction', task_type=Task.TaskTypes.data_processing)
        self.task.set_parameter('clearml_raw_dataset_id', clearml_raw_dataset_id)
        self.task.execute_remotely()

        self.clearml_raw_dataset = Dataset.get(dataset_id=clearml_raw_dataset_id)
        self.wlasl_path = Path(self.clearml_raw_dataset.get_local_copy())
        print('Store ClearML dataset in:', self.wlasl_path)

        self._obtain_full_dataset()
        self.metadata_path = self.wlasl_path / 'WLASL_parsed_metadata.json'
        if self.metadata_path.exists():
            print(f"Metadata file found at {self.metadata_path}. Loading existing metadata.")
            with open(self.metadata_path, 'r') as json_file:
                self.parsed_metadata = json.load(json_file)
                print(f'Loaded metadata with {len(self.parsed_metadata)} entries.')
        else:
            print(f"Metadata file not found. Parsing WLASL metadata from {self.risangbaskoro_videos_dir / '../WLASL_v0.3.json'}")
            self._parse_wlasl_metadata()

    
    def _obtain_full_dataset(self):
        self.risangbaskoro_zip = Path(self.wlasl_path / 'wlasl-processed.zip')
        self.sttaseen_zip = Path(self.wlasl_path / 'wlasl2000-resized.zip')
        risangbaskoro_please_download_text = 'Please download the dataset from https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed/data'
        sttaseen_please_download_text = 'Please download the dataset from https://www.kaggle.com/datasets/sttaseen/wlasl2000-resized/data'

        if not self.risangbaskoro_zip.exists():
            print(self.risangbaskoro_zip)
            raise ValueError('WLASL dataset not found. ' + risangbaskoro_please_download_text)
        
        if not self.sttaseen_zip.exists():
            print(self.sttaseen_zip)
            raise ValueError('Invalid directory for backup WLASL dataset. ' + sttaseen_please_download_text)
        
        if not (self.wlasl_path / 'wlasl-processed').exists(): # Check if the unzipped directory exists
            print(f'Unzipping {self.risangbaskoro_zip}...')
            os.system(f"unzip -q {self.risangbaskoro_zip} -d {self.wlasl_path / 'wlasl-processed'}")
            self.risangbaskoro_videos_dir = Path(self.wlasl_path / 'wlasl-processed/videos')
            if not (self.risangbaskoro_videos_dir / '../WLASL_v0.3.json').exists():
                print(self.risangbaskoro_videos_dir / '../WLASL_v0.3.json')
                raise ValueError('WLASL metadata file not found. ' + risangbaskoro_please_download_text)
            else: print(f"WLASL metadata file found at {self.risangbaskoro_videos_dir / '../WLASL_v0.3.json'}") 
        
        if not (self.wlasl_path / 'wlasl2000-resized').exists(): # Backup directory for missing videos
            print(f'Unzipping {self.sttaseen_zip}...')
            os.system(f"unzip -q {self.sttaseen_zip} -d {self.wlasl_path / 'wlasl2000-resized'}")
            self.sttaseen_videos_dir = Path(self.wlasl_path / 'wlasl2000-resized/wlasl-complete/videos') 
            if not self.sttaseen_videos_dir.exists():
                print(self.sttaseen_videos_dir)
                raise ValueError('Invalid directory for backup WLASL dataset. ' + sttaseen_please_download_text)
            else: print(f'Backup WLASL dataset found at {self.sttaseen_videos_dir}')


    def _parse_wlasl_metadata(self):
        """
        Parse the WLASL metadata and extract relevant information
        This method reads the WLASL metadata JSON file, extracts video paths, glosses, and frame indices, 
        and saves the parsed metadata to a JSON file. It also counts the number of instances for each gloss.
        The metadata is saved in the format:
        {
            "video_id": "unique_video_id",
            "gloss": "word_being_expressed",
            "video_path": "path_to_video",
            "frame_start": start_frame,
            "frame_end": end_frame,
            "split": "train/val/test"
        }
        """
        with open(self.risangbaskoro_videos_dir / '../WLASL_v0.3.json', 'r') as json_file:
            raw_metadata = json.load(json_file)
            print('Total number of Glosses:', len(raw_metadata))

        self.parsed_metadata = []
        for item in tqdm(raw_metadata):
            gloss = item['gloss'] if isinstance(item, dict) else item
            instances = item['instances'] if isinstance(item, dict) else []

            for instance in instances: 
                video_id = instance['video_id']
                if (self.risangbaskoro_videos_dir / f'{video_id}.mp4').exists():
                    video_path = self.risangbaskoro_videos_dir / f'{video_id}.mp4'
                elif (self.sttaseen_videos_dir / f'{video_id}.mp4').exists():
                    video_path = self.sttaseen_videos_dir / f'{video_id}.mp4' # Add missing videos from wlasl2000-resized
                else: continue

                self.parsed_metadata.append({
                    'video_id': video_id, # Unique identifier for the video
                    'gloss': gloss, # The word being expressed
                    'video_path': str(video_path), # Path to the video in the datasets
                    'frame_start': instance['frame_start'], # Frame number where the word starts
                    'frame_end': instance['frame_end'], # Frame number where the word ends
                    'split': instance['split'] # Subset type of data when modeling (train, val, test)
                })

        # Save parsed metadata to JSON file
        with open(self.metadata_path, 'w') as json_file:
            json.dump(self.parsed_metadata, json_file, indent=4)
            print(f'Parsed metadata saved to {self.metadata_path}. Total videos: {len(self.parsed_metadata)}')


    def extract_wlasl_landmarks(self): # Extract landmarks from WLASL videos and save them to a ClearML dataset.
        dataset = Dataset.create(
            dataset_name=self.clearml_raw_dataset.name, 
            dataset_project='SyntaxSquad', 
            parent_datasets=[self.clearml_raw_dataset.id],
        )
        output_dir = Path(self.wlasl_path / 'WLASL_landmarks')
        output_dir.mkdir(parents=True, exist_ok=True)
        landmarks_dict = {}

        for item in tqdm(self.parsed_metadata, total=len(self.parsed_metadata)):
            video_path = Path(item['video_path'])
            start, end = item['frame_start'], item['frame_end']
            try:
                video_landmarks = self.extract_video_landmarks(video_path, start, end)
                if video_landmarks is not None:
                    saved_data = {**item, 'landmarks': video_landmarks}
                    landmarks_dict[video_path.stem] = saved_data

                    # npy_path = output_dir / f'{video_path.stem}.npy'
                    # np.save(npy_path, saved_data)
                    # print(f'Saved landmarks to {npy_path}')
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
            clear_output(wait=True)

        np.savez_compressed(self.wlasl_path / 'WLASL_landmarks.npz', **landmarks_dict)
        dataset.add_files(self.wlasl_path / 'WLASL_landmarks.npz', recursive=False)
        dataset.upload(show_progress=True, verbose=True)
        dataset.finalize(verbose=True)
        print(f"Dataset '{dataset.name}' expanded from '{self.clearml_raw_dataset.id}' and uploaded successfully with ID: {dataset.id}")


if __name__ == '__main__':
    extractor = WLASLLandmarksExtractor(clearml_raw_dataset_id='921fdb13ed94464ebcf0dd0586856a5c')
    extractor.extract_wlasl_landmarks()