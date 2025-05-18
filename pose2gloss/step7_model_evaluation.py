import numpy as np
import pandas as pd
from clearml import Task
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from model_utils import *

# Initialize the ClearML task
Task.add_requirements('nvidia-cudnn-cu12', '9.3.0.75')
task = Task.init(project_name='SyntaxSquad', task_type=Task.TaskTypes.testing, task_name='Step 7: Evaluate the model on all 3 subsets')
args = {
    'data_transformation_task_id': '', # ID of the task that performed data transformation
    'hyperparameter_tuning_task_id': '', # ID of the task that performed hyperparameter tuning
}
task.connect(args)
task.execute_remotely()

# Get the artifacts from data transformation task
data_transformation_task = Task.get_task(task_id=args['data_transformation_task_id'])
X_train, y_train = data_transformation_task.artifacts['X_train'].get(), data_transformation_task.artifacts['y_train'].get()
X_val, y_val = data_transformation_task.artifacts['X_val'].get(), data_transformation_task.artifacts['y_val'].get()
X_test, y_test = data_transformation_task.artifacts['X_test'].get(), data_transformation_task.artifacts['y_test'].get()
gloss_labels = data_transformation_task.artifacts['label_encoder'].get().classes_

# Load the best model from the previous task and evaluate it
hyperparameter_tuning_task = Task.get_task(task_id=args['hyperparameter_tuning_task_id'])
best_job_id = hyperparameter_tuning_task.get_parameter('General/best_job_id')
best_model_training_task = Task.get_task(task_id=best_job_id)
batch_size = int(best_model_training_task.get_parameter('General/batch_size'))
model = load_model(best_model_training_task.models['output'][-1].get_local_copy()) # Last snapshot
model.summary()

# Prepare the TF dataset for efficient data loading
train_tf_dataset = prepare_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
val_tf_dataset = prepare_tf_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)
test_tf_dataset = prepare_tf_dataset(X_test, y_test, batch_size=batch_size, shuffle=False)

# Evaluation
train_loss, train_accuracy, train_top5_accuracy = model.evaluate(train_tf_dataset, batch_size=batch_size, verbose=1)
val_loss, val_accuracy, val_top5_accuracy = model.evaluate(val_tf_dataset, batch_size=batch_size, verbose=1)
test_loss, test_accuracy, test_top5_accuracy = model.evaluate(test_tf_dataset, batch_size=batch_size, verbose=1)

# Report metrics to ClearML
task.logger.report_single_value('test_loss', test_loss)
task.logger.report_single_value('test_accuracy', test_accuracy)
task.logger.report_single_value('test_top5_accuracy', test_top5_accuracy)

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