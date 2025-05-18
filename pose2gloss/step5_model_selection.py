import pandas as pd
from clearml import Task
from tensorflow.keras.models import load_model
from model_utils import *

# Initialize the ClearML task
Task.add_requirements('nvidia-cudnn-cu12', '9.3.0.75')
task = Task.init(
    project_name='SyntaxSquad', task_type=Task.TaskTypes.testing, 
    task_name='Step 5: Model selection for HPO top performers based on validation metrics',
)
args = {
    'data_transformation_task_id': '', # ID of the task that performed data transformation
    'hpo_task_ids': '', # List of task IDs for the hyperparameter optimization tasks
}
task.connect(args)
task.execute_remotely()

# Get the artifacts from data transformation task and Prepare the TF dataset for efficient data loading
data_transformation_task = Task.get_task(task_id=args['data_transformation_task_id'])
X_val, y_val = data_transformation_task.artifacts['X_val'].get(), data_transformation_task.artifacts['y_val'].get()
val_tf_dataset = prepare_tf_dataset(X_val, y_val, batch_size=128, shuffle=False)

# Initialize variables to keep track of the best model
best_model_training_task_id = args['hpo_task_ids'][0]
best_val_accuracy = 0.0
metrics_table = {
    'Task ID': [],
    'Model Name': [],
    'Validation Loss': [],
    'Validation Accuracy': [],
    'Validation Top-5 Accuracy': [],
}

# Load the models from the HPO tasks and evaluate them on validation set to choose the best one
for hpo_task_id in args['hpo_task_ids']:
    hpo_task = Task.get_task(task_id=hpo_task_id)
    model_name = hpo_task.get_parameter('model_name')
    hpo_top_experiment_id = hpo_task.get_reported_single_value('best_job_id')
    hpo_top_experiment = Task.get_task(task_id=hpo_top_experiment_id)
    model = load_model(hpo_top_experiment.models['output'][-1].get_local_copy()) # Last snapshot
    task.logger.report_text(model.summary())

    # Evaluation on validation set
    val_loss, val_accuracy, val_top5_accuracy = model.evaluate(val_tf_dataset, batch_size=128, verbose=1)
    metrics_table['Task ID'].append(hpo_top_experiment_id)
    metrics_table['Model Name'].append(model_name)
    metrics_table['Validation Loss'].append(val_loss)
    metrics_table['Validation Accuracy'].append(val_accuracy)
    metrics_table['Validation Top-5 Accuracy'].append(val_top5_accuracy)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_training_task_id = hpo_top_experiment_id

task.logger.report_table(
    table_plot=pd.DataFrame(metrics_table), series='Statistics',
    title='Model evaluation metrics for HPO top performers on validation set'
)
task.logger.report_single_value('best_model_training_task_id', best_model_training_task_id)