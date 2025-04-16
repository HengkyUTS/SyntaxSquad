import json
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import Counter
from wordcloud import WordCloud
from clearml import Task, Dataset

# Initialize the ClearML task
task = Task.init(project_name='SyntaxSquad', task_name='step1_data_splitting', task_type=Task.TaskTypes.data_processing)
args = {
    'wlasl_landmarks_dataset_id': '', # ClearML dataset ID for WLASL landmarks dataset
    'chosen_landmarks': None, # List of landmarks to keep (None for all landmarks)
    'chosen_labels': None, # List of labels to keep (None for all labels)
    'max_labels': 100, # Top N labels to keep (None for all labels)
    'max_samples': None # Maximum number of samples to keep (None for all samples)
}
task.connect(args)
task.execute_remotely()

# Load the WLASL landmarks dataset and metadata
wlasl_path = Dataset.get(dataset_id=args['wlasl_landmarks_dataset_id']).get_local_copy()
landmarks_dict = np.load(f'{wlasl_path}/WLASL_landmarks.npz', allow_pickle=True)
with open(f'{wlasl_path}/WLASL_parsed_metadata.json', 'r') as json_file:
    parsed_metadata = json.load(json_file)
    glosses_counts = dict(Counter(item['gloss'] for item in parsed_metadata).most_common())
    print(glosses_counts)

# Visualize the distribution of glosses
top_n = 20
glosses = list(glosses_counts.keys())[:top_n]
counts = list(glosses_counts.values())[:top_n]

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.bar(glosses, counts)
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.hist(counts, edgecolor='black')
plt.ylabel('Frequency')
plt.tight_layout()
task.logger.report_matplotlib_figure(figure=plt, title=f"Distribution of Top {top_n}/{args['max_labels']} Glosses", series='Statistics')

wordcloud = WordCloud(width=1000, height=500, background_color='white').generate_from_frequencies(glosses_counts)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
task.logger.report_matplotlib_figure(figure=plt, title=f"Word Cloud of Top {top_n}/{args['max_labels']} Glosses", series='Statistics')


# Splitting the dataset into train, validation, and test sets
def load_subset(split, chosen_landmarks=None, chosen_labels=None, max_labels=None, max_samples=None):
    chosen_landmarks = chosen_landmarks or list(range(landmarks_dict['0'].shape[1]))
    if max_labels: chosen_labels = set(list(glosses_counts.keys())[:max_labels])
    elif chosen_labels: chosen_labels = set(chosen_labels)
    X, y = [], []

    for k, landmarks in tqdm(landmarks_dict.items(), desc=f'Splitting {split} subset'):
        meta = parsed_metadata[int(k)]
        if meta['split'] != split: continue
        if chosen_labels is None or meta['gloss'] in chosen_labels:
            X.append(landmarks[:, chosen_landmarks, :])
            y.append(meta['gloss'])
    return (X[:max_samples], y[:max_samples]) if max_samples else (X, y)


def subset_statistics(X, y, subset_name='Subset'):
    num_videos, num_classes = len(X), len(np.unique(y))
    avg_frames_per_video, avg_landmarks_per_frame = 0, 0
    if num_videos > 0:
        avg_frames_per_video = np.mean([len(video) for video in X])
        avg_landmarks_per_frame = np.mean([len(video[0]) for video in X if len(video) > 0])

    return pd.DataFrame({
        'Subset': [subset_name],
        'Number of Videos': [num_videos],
        'Number of Classes': [num_classes],
        'Average Frames/Video': [avg_frames_per_video],
        'Number of Landmarks/Frame': [avg_landmarks_per_frame]
    })

# Load the dataset and split it into train, validation, and test sets
X_train, y_train = load_subset('train', chosen_landmarks=args['chosen_landmarks'], chosen_labels=args['chosen_labels'], max_labels=args['max_labels'])
X_val, y_val = load_subset('val', chosen_landmarks=args['chosen_landmarks'], chosen_labels=args['chosen_labels'], max_labels=args['max_labels'])
X_test, y_test = load_subset('test', chosen_landmarks=args['chosen_landmarks'], chosen_labels=args['chosen_labels'], max_labels=args['max_labels'])
print('First video has', len(X_train[0]), 'frames with', len(X_train[0][0]), 'landmarks')

# Save the processed data as artifacts
task.upload_artifact('X_train', artifact_object=X_train)
task.upload_artifact('X_val', artifact_object=X_val)
task.upload_artifact('X_test', artifact_object=X_test)
task.upload_artifact('y_train', artifact_object=y_train)
task.upload_artifact('y_val', artifact_object=y_val)
task.upload_artifact('y_test', artifact_object=y_test)

# Compute high-level statistics
train_stats = subset_statistics(X_train, y_train, 'Train')
val_stats = subset_statistics(X_val, y_val, 'Validation')
test_stats = subset_statistics(X_test, y_test, 'Test')
subset_summary = pd.concat([train_stats, val_stats, test_stats], ignore_index=True)

# Create detailed statistics DataFrame
train_details = pd.DataFrame({
    'Subset': ['Train'] * len(X_train),
    'Gloss': y_train,
    'Video Length': [len(video) for video in X_train],
    'Frame Landmarks': [len(video[0]) for video in X_train]
})

val_details = pd.DataFrame({
    'Subset': ['Validation'] * len(X_val),
    'Gloss': y_val,
    'Video Length': [len(video) for video in X_val],
    'Frame Landmarks': [len(video[0]) for video in X_val]
})

test_details = pd.DataFrame({
    'Subset': ['Test'] * len(X_test),
    'Gloss': y_test,
    'Video Length': [len(video) for video in X_test],
    'Frame Landmarks': [len(video[0]) for video in X_test]
})

# Combine descriptive statistics
combined_details = pd.concat([train_details, val_details, test_details], ignore_index=True)
detailed_stats = combined_details.groupby('Subset').agg({'Video Length': ['mean', 'std', 'min', 'max'],}).reset_index()

# Merge high-level and detailed statistics
detailed_stats.columns = ['Subset'] + [f"{col[0]} ({col[1]})" for col in detailed_stats.columns[1:]]
final_stats = pd.merge(subset_summary, detailed_stats, on='Subset')
task.logger.report_table(table_plot=final_stats, title=f"Dataset Statistics for Top {args['max_labels']} Glosses", series='Statistics')

# Visualize the statistics
fig = px.histogram(combined_details, x='Video Length', color='Subset', barmode='overlay', nbins=20)
fig.update_layout(xaxis_title='Video Length', yaxis_title='Frequency', legend_title='Subset')
task.logger.report_plotly(figure=fig, title=f"Histogram of Video Length by Subset (Top {args['max_labels']} Glosses)", series='Statistics')

fig = px.box(combined_details, x='Subset', y='Video Length', color='Subset')
fig.update_layout(xaxis_title='Subset', yaxis_title='Video Length (Frames)')
task.logger.report_plotly(figure=fig, title=f"Boxplot of Video Length by Subset (Top {args['max_labels']} Glosses)", series='Statistics')