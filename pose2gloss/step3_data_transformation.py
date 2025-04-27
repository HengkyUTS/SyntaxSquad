import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from clearml import Task

# Initialize the ClearML task
task = Task.init(
    project_name='SyntaxSquad', task_type=Task.TaskTypes.data_processing,
    task_name='Step 3: Perform padding or truncation on X_train/X_val/X_test and label encoding on y_train/y_val/y_test'
)
args = {
    'data_splitting_task_id': '',  # ID of the task that performed data splitting
    'data_augmentation_task_id': '',  # ID of the task that performed data augmentation
    'max_frames': 195, # Maximum number of frames for padding/truncating
    'pad_value': -100, # Value to pad with
}
task.connect(args)
task.execute_remotely()


def pad_or_truncate(X, y, max_frames=None, pad_value=0):
    if max_frames is None: max_frames = max(len(video_landmarks) for video_landmarks in X)
    X_padded = np.array([video_landmarks[:max_frames] if len(video_landmarks) > max_frames else np.pad(video_landmarks, (
        (0, max_frames - len(video_landmarks)), # 0 for no padding before existing frames, the other for padding needed to reach max_frames
        (0, 0), # Apply no padding to the 2nd dimension (landmarks)
        (0, 0)  # Apply no padding to the 3rd dimension (coordinates)
    ), mode='constant', constant_values=pad_value) for video_landmarks in tqdm(X)])
    return X_padded, y

# Load the data from the previous tasks
data_splitting_task = Task.get_task(task_id=args['data_splitting_task_id'])
data_augmentation_task = Task.get_task(task_id=args['data_augmentation_task_id'])
X_train, y_train = data_augmentation_task.artifacts['X_train'].get(), data_augmentation_task.artifacts['y_train'].get()
X_val, y_val = data_splitting_task.artifacts['X_val'].get(), data_splitting_task.artifacts['y_val'].get()
X_test, y_test = data_splitting_task.artifacts['X_test'].get(), data_splitting_task.artifacts['y_test'].get()

# Pad or truncate the sequences to the maximum number of frames
X_train, y_train = pad_or_truncate(X_train, y_train, max_frames=args['max_frames'], pad_value=args['pad_value'])
X_val, y_val = pad_or_truncate(X_val, y_val, max_frames=args['max_frames'], pad_value=args['pad_value'])
X_test, y_test = pad_or_truncate(X_test, y_test, max_frames=args['max_frames'], pad_value=args['pad_value'])
print(X_train.shape, X_val.shape, X_test.shape)
print('X_train:', X_train.shape, '- X_val:', X_val.shape, '- X_test:', X_test.shape)

# Convert the labels to integers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)
print('y_train:', y_train.shape, '- y_val:', y_val.shape, '- y_test:', y_test.shape)

# Save the processed data as artifacts
task.upload_artifact('X_train', artifact_object=X_train)
task.upload_artifact('X_val', artifact_object=X_val)
task.upload_artifact('X_test', artifact_object=X_test)
task.upload_artifact('y_train', artifact_object=y_train)
task.upload_artifact('y_val', artifact_object=y_val)
task.upload_artifact('y_test', artifact_object=y_test)
task.upload_artifact('label_encoder', artifact_object=label_encoder)