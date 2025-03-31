import os
import json
import logging
import numpy as np

from tqdm import tqdm
from pathlib import Path
from IPython.display import clear_output
from video2landmarks import VideoLandmarksExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('landmarks_extraction.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class WLASLLandmarksExtractor(VideoLandmarksExtractor):
    def __init__(self, 
        risangbaskoro_videos_dir: str = '/tmp/wlasl-processed/videos',
        sttaseen_videos_dir: str = '/tmp/wlasl2000-resized/wlasl-complete/videos', 
        metadata_path: str = '/tmp/WLASL_parsed_metadata.json',
        **kwargs):
        """
        Initialize the WLASLVideoProcessor with directories.
        
        Args:
            risangbaskoro_videos_dir (str): Directory with WLASL videos. Downloaded from https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed/data.
            sttaseen_videos_dir (str): Directory with backup WLASL videos. Download from https://www.kaggle.com/datasets/sttaseen/wlasl2000-resized/data.
            metadata_path (str): Path to save parsed metadata.
            **kwargs: Additional arguments for the VideoLandmarksExtractor.
        """
        super().__init__(**kwargs)
        self.risangbaskoro_videos_dir = Path(risangbaskoro_videos_dir)
        self.sttaseen_videos_dir = Path(sttaseen_videos_dir) # Backup directory for missing videos
        if not os.path.exists(self.risangbaskoro_videos_dir / '../WLASL_v0.3.json'):
            raise ValueError('WLASL metadata file not found. Please download the dataset from https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed/data.')
        if not os.path.exists(self.sttaseen_videos_dir):
            raise ValueError("Invalid directory for backup WLASL dataset. Please download the dataset from https://www.kaggle.com/datasets/sttaseen/wlasl2000-resized/data.")
        
        self.metadata_path = Path(metadata_path) # Path to save parsed metadata
        if self.metadata_path.exists():
            logger.info(f"Metadata file found at {self.metadata_path}. Loading existing metadata.")
            with open(self.metadata_path, 'r') as json_file:
                self.parsed_metadata = json.load(json_file)
                logger.info(f'Loaded metadata with {len(self.parsed_metadata)} entries.')
        else:
            logger.info(f"Metadata file not found. Parsing WLASL metadata from {self.risangbaskoro_videos_dir / '../WLASL_v0.3.json'}")
            self._parse_wlasl_metadata()


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
                if os.path.exists(self.risangbaskoro_videos_dir / f'{video_id}.mp4'):
                    video_path = self.risangbaskoro_videos_dir / f'{video_id}.mp4'
                elif os.path.exists(self.sttaseen_videos_dir / f'{video_id}.mp4'):
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
            logger.info(f'Parsed metadata saved to {self.metadata_path}. Total videos: {len(self.parsed_metadata)}')


    def extract_wlasl_landmarks(self, output_dir='landmarks', final_npz_path='landmarks.npz'):
        """
        Extract landmarks from videos based on metadata and save them to a specified directory.
        Args:
            output_dir (str): Directory to save the extracted landmarks.
            metadata_path (str): Path to the metadata file."
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        landmarks_dict = {}

        for item in tqdm(self.parsed_metadata, total=len(self.parsed_metadata)):
            video_path = Path(item['video_path'])
            start, end = item['frame_start'], item['frame_end']
            try:
                video_landmarks = self.extract_video_landmarks(video_path, start, end)
                if video_landmarks is not None:
                    npy_path = output_dir / f'{video_path.stem}.npy'
                    saved_data = {**item, 'landmarks': video_landmarks}
                    np.save(npy_path, saved_data)
                    landmarks_dict[video_path.stem] = saved_data
                    logger.info(f'Saved landmarks to {npy_path}')
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                continue
            clear_output(wait=True)

        # np.save(final_npz_path, landmarks_dict)
        np.savez_compressed(final_npz_path, **landmarks_dict)
        logger.info(f'All landmarks saved to {final_npz_path}')


if __name__ == "__main__":
    extractor = WLASLLandmarksExtractor(
        risangbaskoro_videos_dir='datasets/wlasl-processed/videos',
        sttaseen_videos_dir='datasets/wlasl2000-resized/wlasl-complete/videos',
        metadata_path='datasets/WLASL_parsed_metadata.json',
        use_adaptive_sampling=False
    )
    extractor.extract_wlasl_landmarks(output_dir='datasets/WLASL_landmarks', final_npz_path='datasets/WLASL_landmarks.npz')