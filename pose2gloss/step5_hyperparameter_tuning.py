from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformParameterRange, UniformIntegerParameterRange
from clearml.automation.optuna import OptimizerOptuna

# Initialize the ClearML task
task = Task.init(
    project_name='SyntaxSquad', task_type=Task.TaskTypes.optimizer, reuse_last_task_id=False,
    task_name='Step 5: Hyperparameter Optimization for Pose-to-Gloss Model',
)
args = {
    'model_training_template_task_id': 'be73c0a2fe814523a54c4bf01fe9a092', # ID of the "template" task that performed model training
    'execution_queue': 'Remote_GPU_queue', # The execution queue to use for launching Tasks (experiments)
    'optimization_time_limit': None, # Maximum minutes for the entire optimization process
    'max_iteration_per_job': 200,  # Maximum number of epochs per job
    'total_max_jobs': 3, # Maximum number of jobs to launch for the optimization
}
task.connect(args)
task.execute_remotely()

# Callback function for job completion
def job_complete_callback(job_id, objective_value, objective_iteration, job_parameters, top_performance_job_id):
    print(f'Job {job_id} completed with objective value {objective_value} at iteration {objective_iteration}')
    print(f'Parameters: {job_parameters}')
    if job_id == top_performance_job_id: print('New top performer! This job broke the record.')


# Initialize HyperParameterOptimizer
optimizer = HyperParameterOptimizer(
    base_task_id=args['model_training_template_task_id'],    # The Task ID to be used as template experiment to optimize
    hyper_parameters=[
        UniformIntegerParameterRange('General/num_epochs', min_value=100, max_value=args['max_iteration_per_job'], step_size=100),  # Number of epochs
        UniformIntegerParameterRange('General/batch_size', min_value=64, max_value=64, step_size=256),    # Batch size
        UniformParameterRange('General/learning_rate', min_value=2e-4, max_value=1e-3, step_size=2e-4),   # Learning rate
        UniformParameterRange('General/dropout_rate', min_value=0.1, max_value=0.5, step_size=0.1),       # Dropout rate
    ],
    objective_metric_title=['Optimization Metric', 'Optimization Metric'],
    objective_metric_series=['val_loss', 'val_accuracy'],    # Series name in ClearML
    objective_metric_sign=['min', 'max'],                    # Maximize validation accuracy
    optimizer_class=OptimizerOptuna,                         # Optuna search strategy to perform robust and efficient hyperparameter optimization at scale
    max_number_of_concurrent_tasks=2,                        # Limit concurrent tasks to manage resources
    execution_queue=args['execution_queue'],                 # Queue for running tasks
    optimization_time_limit=args['optimization_time_limit'], # Maximum minutes for the entire optimization process
    save_top_k_tasks_only=2,                                 # Top K performing Tasks will be kept, the others will be archived
    max_iteration_per_job=args['max_iteration_per_job'],     # Maxiumum number of reported iterations for the specified objective
    total_max_jobs=args['total_max_jobs'],                   # Maximum number of jobs to launch for the optimization
    pool_period_min=1.0,                                     # Check the experiments every 1 min
)

# optimizer.set_report_period(5.0) # Report every 5 minutes
optimizer.start(job_complete_callback=job_complete_callback)
optimizer.wait() # Wait until process is done (notice we are controlling the optimization process in the background)

# Log the best parameters
best_job = optimizer.get_top_experiments(top_k=1)
if best_job:
    best_job = best_job[0]
    best_hyper_parameters, best_metrics = best_job.hyper_parameters, best_job.get_last_scalar_metrics()
    print('Best Job:', best_job.id)
    print('Best HPO Parameters:', best_hyper_parameters)
    print('Best Metrics:', best_metrics)
    task.upload_artifact('best_hyper_parameters', artifact_object=best_hyper_parameters)
    task.upload_artifact('best_metrics', artifact_object=best_metrics)
else: print('No top experiments found.')
optimizer.stop() # Stop the optimizer