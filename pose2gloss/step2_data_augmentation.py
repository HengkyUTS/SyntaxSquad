import numpy as np
from tqdm import tqdm
from clearml import Task

# Initialize the ClearML task
task = Task.init(project_name='SyntaxSquad', task_name='step2_data_augmentation', task_type=Task.TaskTypes.data_processing)
task.set_parameter('data_splitting_task_id', '9ebed244b2ab4a448bf076e6334279f4')
task.execute_remotely()


def rotate(data, rotation_matrix):
    frames, landmarks, _ = data.shape
    center = np.array([0.5, 0.5, 0])
    non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))
    data = data.reshape(-1, 3)
    data[non_zero] -= center
    data[non_zero] = np.dot(data[non_zero], rotation_matrix.T)
    data[non_zero] += center
    data = data.reshape(frames, landmarks, 3)
    out_of_range = np.any((data[:, :, :2] < 0) | (data[:, :, :2] > 1), axis=2)
    data[out_of_range] = 0
    return data

def rotate_x(data):
    angle = np.random.choice([np.random.uniform(-30, -10), np.random.uniform(10, 30)])
    theta = np.radians(angle)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    return rotate(data, rotation_matrix)

def rotate_y(data):
    angle = np.random.choice([np.random.uniform(-30, -10), np.random.uniform(10, 30)])
    theta = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return rotate(data, rotation_matrix)

def rotate_z(data):
    angle = np.random.choice([np.random.uniform(-30, -10), np.random.uniform(10, 30)])
    theta = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return rotate(data, rotation_matrix)

def zoom(data):
    factor = np.random.uniform(0.8, 1.2)
    center = np.array([0.5, 0.5])
    non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))
    data[non_zero[:, 0], non_zero[:, 1], :2] = ((
        data[non_zero[:, 0], non_zero[:, 1], :2] - center
    ) * factor + center)
    out_of_range = np.any((data[:, :, :2] < 0) | (data[:, :, :2] > 1), axis=2)
    data[out_of_range] = 0
    return data

def shift(data):
    x_shift = np.random.uniform(-0.2, 0.2)
    y_shift = np.random.uniform(-0.2, 0.2)
    non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))
    data[non_zero[:, 0], non_zero[:, 1], 0] += x_shift
    data[non_zero[:, 0], non_zero[:, 1], 1] += y_shift
    out_of_range = np.any((data[:, :, :2] < 0) | (data[:, :, :2] > 1), axis=2)
    data[out_of_range] = 0
    return data

def mask(data):
    frames, landmarks, _ = data.shape
    num_hands = int(0.3 * 42)
    num_rest = int(0.6 * (landmarks - 42))
    mask = np.zeros(landmarks, dtype=bool)
    indices = np.concatenate([
        np.random.choice(42, num_hands, replace=False),
        np.random.choice(landmarks - 42, num_rest, replace=False) + 42
    ])
    mask[indices] = True
    data[:, mask] = 0
    return data

def hflip(data):
    data[:, :, 0] = 1 - data[:, :, 0]
    return data

def speedup(data):
    return data[::2]

def apply_augmentation(data):
    aug_funcs = [rotate_x, rotate_y, rotate_z, zoom, shift, mask, hflip, speedup]
    np.random.shuffle(aug_funcs)
    count = 0
    for func in aug_funcs:
        if np.random.rand() < 0.5:
            data = func(data)
            count += 1
    if count == 0: data = apply_augmentation(data)
    return data

def augment(X, y, num=None):
    X_aug, y_aug = X.copy(), y.copy()
    for i in tqdm(range(len(y))):
        for _ in range(num or np.random.choice([1, 2, 3])):
            X_aug.append(apply_augmentation(X[i].copy()))
            y_aug.append(y[i])
    return X_aug, y_aug


data_splitting_task = Task.get_task(task_id=task.get_parameter('data_splitting_task_id'))
X_train = data_splitting_task.artifacts['X_train'].get()
y_train = data_splitting_task.artifacts['y_train'].get()
X_train, y_train = augment(X_train, y_train, num=1)

print('The Training set has', len(X_train), 'videos')
print('First video has', len(X_train[0]), 'frames')
print('Each frame has', len(X_train[0][0]), 'landmarks')
print('Each landmark has', len(X_train[0][0][0]), 'coordinates')

# Save the processed data as artifacts
task.upload_artifact('X_train', artifact_object=X_train)
task.upload_artifact('y_train', artifact_object=y_train)