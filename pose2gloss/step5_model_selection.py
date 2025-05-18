import pandas as pd
from clearml import Task

# Initialize the ClearML task
task = Task.init(
    project_name='SyntaxSquad', task_type=Task.TaskTypes.qc, 
    task_name='Step 5: Model selection based on validation metrics',
)
args = {
    'model_training_task_ids': [], # List of task IDs for the hyperparameter optimization tasks
}
task.connect(args)
task.execute_remotely()

# Initialize variables to keep track of the best model
best_model_training_task_id = args['model_training_task_ids'][0]
best_val_accuracy = 0.0
metrics_table = {
    'Task ID': [],
    'Model Name': [],
    'Validation Loss': [],
    'Validation Accuracy': [],
    'Validation Top-5 Accuracy': [],
}

# Load the models from the HPO tasks and evaluate them on validation set to choose the best one
for model_training_task_id in args['model_training_task_ids']:
    model_training_task = Task.get_task(task_id=model_training_task_id)
    model_name = model_training_task.get_parameter('General/model_name')

    val_metrics = model_training_task.get_last_scalar_metrics()['Best Metrics']
    val_loss = val_metrics['val_loss']['last']
    val_accuracy = val_metrics['val_accuracy']['last']
    val_top5_accuracy = val_metrics['val_top5_accuracy']['last']

    metrics_table['Task ID'].append(model_training_task_id)
    metrics_table['Model Name'].append(model_name)
    metrics_table['Validation Loss'].append(val_loss)
    metrics_table['Validation Accuracy'].append(val_accuracy)
    metrics_table['Validation Top-5 Accuracy'].append(val_top5_accuracy)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_training_task_id = model_training_task_id

task.set_parameter('best_model_training_task_id', best_model_training_task_id)
task.logger.report_table(
    table_plot=pd.DataFrame(metrics_table), series='Statistics',
    title='Model evaluation metrics on validation Set'
)
best_index = metrics_table['Task ID'].index(best_model_training_task_id)
print( 
    f'Best Model Training Task ID: {best_model_training_task_id}\n'
    f'Best Model Name: {metrics_table["Model Name"][best_index]}\n'
    f'Validation Loss: {metrics_table["Validation Loss"][best_index]}\n'
    f'Validation Top-5 Accuracy: {metrics_table["Validation Top-5 Accuracy"][best_index]}\n'
    f'Validation Accuracy: {best_val_accuracy}'
)