import cv2
import logging
import mediapy
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor

# Configure logging with file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('landmarks_extraction.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class VideoLandmarksExtractor:
    def __init__(
        self, use_adaptive_sampling=False, 
        min_detection_confidence: float = 0.7, min_tracking_confidence: float = 0.7,
        filtered_hand=list(range(0, 21)), filtered_pose=list(range(11, 17)), filtered_face=[
            0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58,
            61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105,
            107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154,
            155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191,
            234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291,
            293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324,
            332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380,
            381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409,
            415, 454, 466, 468, 473]):
        """
        Initialize the VideoLandmarksExtractor with MediaPipe solutions for hands, pose, and face landmarks.
        Args:
            min_detection_confidence (float): Minimum confidence for detection.
            min_tracking_confidence (float): Minimum confidence for tracking.
            filtered_hand (list): List of hand landmarks to extract.
            filtered_pose (list): List of pose landmarks to extract.
            filtered_face (list): List of face landmarks to extract.
        """
        # Landmarks to extract
        self.filtered_hand, self.hand_cnt = filtered_hand, len(filtered_hand)
        self.filtered_pose, self.pose_cnt = filtered_pose, len(filtered_pose)
        self.filtered_face, self.face_cnt = filtered_face, len(filtered_face)
        self.total_landmarks = self.hand_cnt * 2 + self.pose_cnt + self.face_cnt
        
        # Initialize MediaPipe solutions
        self.mp_hands = mp.solutions.hands.Hands()
        self.mp_pose = mp.solutions.pose.Pose()
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence,
            static_image_mode = False, # Process video frames
            refine_landmarks = True # This will result in 478 landmarks instead of 468
        )
        self.use_adaptive_sampling = use_adaptive_sampling # Use adaptive sampling based on motion
        logger.info("MediaPipe solutions initialized for hand, pose, and face landmarks.")


    def extract_video_landmarks(self, video_path, start_frame=1, end_frame=-1):
        """
        This function extracts hand, pose, and face landmarks from a video file.
        The landmarks are stored in a numpy array with shape (total_landmarks, 3) for each frame.
        The function uses MediaPipe to process the video frames and extract the landmarks.
        The landmarks are filtered based on the specified indices for hands, pose, and face.
        Args:
            video_path (str): Path to the input video file.
            start_frame (int): Starting frame for processing.
            end_frame (int): Ending frame for processing. If -1, process until the end of the video.
        Returns:
            np.ndarray: Landmarks for each frame with shape (num_frames, total_landmarks, 3).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f'Failed to open video: {video_path}')
            return None, None

        if start_frame <= 1: start_frame = 1 # If the starting is 0
        elif start_frame > int(cap.get(cv2.CAP_PROP_FRAME_COUNT)): # If the starting frame > the total frames
            start_frame = 1
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if end_frame < 0: end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # If the final frame was not given (-1)
        all_frame_landmarks = np.zeros((end_frame - start_frame + 1, self.total_landmarks, 3))
        frame_index, prev_frame = 1, None
        
        while cap.isOpened() and frame_index <= end_frame:
            ret, frame = cap.read()
            if not ret: break

            if self.use_adaptive_sampling and prev_frame is not None: # Adaptive sampling based on motion
                if self._compute_motion_score(prev_frame, frame) < 1.0: 
                    frame_index += 1
                    continue # Skip frames with low motion

            if frame_index >= start_frame:
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_landmarks = self.extract_frame_landmarks(frame)
                all_frame_landmarks[frame_index - start_frame] = frame_landmarks
                
            prev_frame = frame
            frame_index += 1
        
        cap.release()
        self.mp_hands.reset()
        self.mp_pose.reset()
        self.mp_face.reset()
        return all_frame_landmarks


    def extract_frame_landmarks(self, frame):
        """
        This function processes the frame to extract hand, pose, and face landmarks.
        The landmarks are stored in a numpy array with shape (total_landmarks, 3).
        The function uses MediaPipe to process the frame and extract the landmarks.
        The landmarks are filtered based on the specified indices for hands, pose, and face.
        Args:
            frame (np.ndarray): The video frame.
        Returns:
            np.ndarray: Landmarks for the frame with shape (total_landmarks, 3).
        """
        frame_landmarks = np.zeros((self.total_landmarks, 3), dtype=np.float32)
        
        # Hands: 21 landmarks * 3 (x, y, z) * 2 (left and right) = 126 values
        def get_hand_landmarks(frame):
            results_hands = self.mp_hands.process(frame)
            if not results_hands.multi_hand_landmarks: return
            for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                if results_hands.multi_handedness[i].classification[0].index == 0: 
                    frame_landmarks[:self.hand_cnt, :] = np.array([
                        (lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark
                    ])[self.filtered_hand] # Right hand
                else: 
                    frame_landmarks[self.hand_cnt:self.hand_cnt * 2, :] = np.array([
                        (lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark
                    ])[self.filtered_hand] # Left hand
        
        # Pose: 6 upper body * 3 (x, y, z) = 18 values
        def get_pose_landmarks(frame):
            results_pose = self.mp_pose.process(frame)
            if not results_pose.pose_landmarks: return
            frame_landmarks[self.hand_cnt * 2:self.hand_cnt * 2 + self.pose_cnt, :] = np.array([
                (lm.x, lm.y, lm.z) for lm in results_pose.pose_landmarks.landmark
            ])[self.filtered_pose]

        # Face: 132 landmarks * 3 (x, y, z) = 396 values
        def get_face_landmarks(frame):
            results_face = self.mp_face.process(frame)
            if not results_face.multi_face_landmarks: return
            frame_landmarks[self.hand_cnt * 2 + self.pose_cnt:, :] = np.array([
                (lm.x, lm.y, lm.z) for lm in results_face.multi_face_landmarks[0].landmark
            ])[self.filtered_face]

        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(get_hand_landmarks, frame)
            executor.submit(get_pose_landmarks, frame)
            executor.submit(get_face_landmarks, frame)
        return frame_landmarks


    def _compute_motion_score(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """
        Compute the motion score between two frames using optical flow.
        Args:
            prev_frame (np.ndarray): The previous video frame.
            curr_frame (np.ndarray): The current video frame.
        Returns:
            float: The motion score.
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(np.sum(flow ** 2, axis=2))
        return np.mean(magnitude)


    @staticmethod
    def draw_frame_landmarks(frame, frame_landmarks, radius=5, color=(0, 255, 0), thickness=-1):
        """
        Draw landmarks on a single frame.
        Args:
            frame (np.ndarray): The video frame.
            landmarks (np.ndarray): Landmarks to draw.
            radius (int): Radius of the circles to draw.
            color (tuple): Color of the circles to draw.
            thickness (int): Thickness of the circles to draw.
        """
        for landmark in frame_landmarks:
            x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
            cv2.circle(frame, (x, y), radius, color, thickness)

        plt.imshow(frame[:, :, ::-1])
        plt.axis('off')
        plt.show()


    @staticmethod
    def draw_video_landmarks(video_path, output_path, video_landmarks, start_frame=1, end_frame=-1, radius=5, color=(0, 255, 0), thickness=-1):
        """
        Draw landmarks on a video and save the output.
        Args:
            video_path (str): Path to the input video.
            output_path (str): Path to save the output video with landmarks.
            video_landmarks (np.ndarray): Landmarks for each frame.
            start_frame (int): Starting frame for processing.
            end_frame (int): Ending frame for processing.
            radius (int): Radius of the circles to draw.
            color (tuple): Color of the circles to draw.
            thickness (int): Thickness of the circles to draw.
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        if start_frame <= 1: start_frame = 1 # If the starting is 0
        elif start_frame > int(cap.get(cv2.CAP_PROP_FRAME_COUNT)): # If the video is precropped
            start_frame = 1
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame < 0: end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # If the final frame was not given (-1)
            
        frame_index = 1
        while cap.isOpened() and frame_index <= end_frame:
            ret, frame = cap.read()
            if not ret: break
            if frame_index >= start_frame:
                frame_landmarks = video_landmarks[frame_index - start_frame]
                landmarks = [(int(x * width), int(y * height)) for x, y, _ in frame_landmarks]
                for x, y in landmarks: cv2.circle(frame, (x, y), radius, color, thickness)
                out.write(frame)
            # else: out.write(frame) # Enable for full video
            frame_index += 1

        cap.release()
        out.release()
        mediapy.show_video(mediapy.read_video(output_path), height=500)