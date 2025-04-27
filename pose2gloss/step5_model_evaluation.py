import numpy as np
import pandas as pd
from clearml import Task
from sklearn.metrics import classification_report, confusion_matrix
from model_utils import build_and_compile_GISLR, prepare_tf_dataset

# Initialize the ClearML task
Task.add_requirements('nvidia-cudnn-cu12', '9.3.0.75')
task = Task.init(project_name='SyntaxSquad', task_type=Task.TaskTypes.testing, task_name='Step 5: Evaluate the model')
args = {
    'data_transformation_task_id': '', # ID of the task that performed data transformation
    'model_training_task_id': '', # ID of the task that performed model training
    'max_frames': 195, # Maximum number of frames for padding/truncating
    'pad_value': -100, # Value to pad with
    'learning_rate': 0.001, # Learning rate for the optimizer
    'batch_size': 128, # Batch size for the dataset
    'conv1d_dropout': 0.2, # Dropout rate for the Conv1D layers
    'last_dropout': 0.2, # Dropout rate for the last Dense layer
}
task.connect(args)
task.execute_remotely()

# Get the artifacts from data transformation task
data_transformation_task = Task.get_task(task_id=args['data_transformation_task_id'])
X_train, y_train = data_transformation_task.artifacts['X_train'].get(), data_transformation_task.artifacts['y_train'].get()
X_val, y_val = data_transformation_task.artifacts['X_val'].get(), data_transformation_task.artifacts['y_val'].get()
X_test, y_test = data_transformation_task.artifacts['X_test'].get(), data_transformation_task.artifacts['y_test'].get()
gloss_labels = data_transformation_task.artifacts['label_encoder'].get().classes_
num_landmarks, num_glosses = X_train.shape[-2], len(set(y_train))

# Build and compile the model
model = build_and_compile_GISLR(
    args['max_frames'], num_landmarks=num_landmarks, num_glosses=num_glosses, pad_value=args['pad_value'], 
    conv1d_dropout=args['conv1d_dropout'], last_dropout=args['last_dropout'], learning_rate=args['learning_rate'], is_training=False,
)
model.summary()

# Load the model from the previous task and evaluate it
model_training_task = Task.get_task(task_id=args['model_training_task_id'])
weights_path = model_training_task.models['output'][-1].get_local_copy() # Last snapshot
model.load_weights(weights_path)

# # Prepare the TF dataset for efficient data loading
train_tf_dataset = prepare_tf_dataset(X_train, y_train, batch_size=args['batch_size'], shuffle=True)
val_tf_dataset = prepare_tf_dataset(X_val, y_val, batch_size=args['batch_size'], shuffle=False)
test_tf_dataset = prepare_tf_dataset(X_test, y_test, batch_size=args['batch_size'], shuffle=False)

# Evaluation
train_loss, train_accuracy, train_top5_accuracy = model.evaluate(train_tf_dataset, batch_size=args['batch_size'], verbose=1)
val_loss, val_accuracy, val_top5_accuracy = model.evaluate(val_tf_dataset, batch_size=args['batch_size'], verbose=1)
test_loss, test_accuracy, test_top5_accuracy = model.evaluate(test_tf_dataset, batch_size=args['batch_size'], verbose=1)

metrics_table = pd.DataFrame({
    'Subset': ['Train', 'Validation', 'Test'],
    'Loss': [train_loss, val_loss, test_loss],
    'Accuracy': [train_accuracy, val_accuracy, test_accuracy],
    'Top-5 Accuracy': [train_top5_accuracy, val_top5_accuracy, test_top5_accuracy],
})
task.logger.report_table(table_plot=metrics_table, title='Model evaluation metrics on 3 subsets', series='Statistics')

y_test_preds = np.argmax(model.predict(test_tf_dataset), axis=1)
cls_report_df = pd.DataFrame(classification_report(y_test, y_test_preds, output_dict=True, target_names=gloss_labels, zero_division=0)).T
task.logger.report_table(table_plot=cls_report_df, title='Classification report on test set', series='Statistics')

cm = confusion_matrix(y_test, y_test_preds)
task.logger.report_confusion_matrix(
    matrix=cm, title='Confusion matrix on test set', series='ignored', yaxis_reversed=True,
    xlabels=gloss_labels, ylabels=gloss_labels, xaxis='Predicted gloss', yaxis='True gloss'
)