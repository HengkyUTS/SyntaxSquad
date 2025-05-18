from clearml import Task
from clearml.automation import ClearmlJob, HyperParameterOptimizer, UniformParameterRange, UniformIntegerParameterRange
from clearml.automation.optuna import OptimizerOptuna

# Initialize the ClearML task
task = Task.init(
    project_name='SyntaxSquad', task_type=Task.TaskTypes.optimizer, reuse_last_task_id=False,
    task_name='Step 6: Hyperparameter optimization for Pose-to-Gloss model',
)
args = {
    'model_selection_task_id': '', # ID of the task that performed model selection
    'execution_queue': '', # The execution queue to use for launching Tasks (experiments)
    'max_iteration_per_job': 100,    # Maximum number of epochs per job
    'total_max_jobs': 2, # Maximum number of jobs to launch for the optimization
}
task.connect(args)
task.execute_remotely()

# Callback function for job completion
def job_complete_callback(job_id, objective_value, objective_iteration, job_parameters, top_performance_job_id):
    print(f'\nJob {job_id} completed with objective value {objective_value} at iteration {objective_iteration}')
    if job_id == top_performance_job_id: print('New top performer! This job broke the record.')
    print(f'Parameters: {job_parameters}\n')


# Load the selected model training task and define hyperparameters search space
model_selection_task = Task.get_task(task_id=args['model_selection_task_id'])
best_job_id = model_selection_task.get_parameter('General/best_model_training_task_id')
base_model_training_task = Task.get_task(task_id=best_job_id)
best_hyperparameters = base_model_training_task.get_parameters()
best_metrics = base_model_training_task.get_last_scalar_metrics()['Best Metrics']
model_name = base_model_training_task.get_parameter('General/model_name')

hyper_parameters = [
    UniformIntegerParameterRange('General/batch_size', min_value=128, max_value=256, step_size=128),
    UniformParameterRange('General/learning_rate', min_value=2e-4, max_value=1e-3, step_size=2e-4),
    UniformIntegerParameterRange('General/reduce_lr_patience', min_value=2, max_value=5, step_size=1),
]
if model_name == 'GISLR': hyper_parameters.extend([
    UniformParameterRange('General/conv1d_dropout', min_value=0.1, max_value=0.5, step_size=0.1), 
    UniformParameterRange('General/last_dropout', min_value=0.1, max_value=0.5, step_size=0.1),
])

# Initialize HyperParameterOptimizer
hpo = HyperParameterOptimizer(
    base_task_id=best_job_id,                               # The Task ID to be used as template experiment to optimize
    hyper_parameters=hyper_parameters,                       # The list of Parameter objects to optimize over
    objective_metric_title=['Best Metrics', 'Best Metrics'], # Multiple objective metrics to optimize
    objective_metric_series=['val_loss', 'val_accuracy'],    # Series name in ClearML
    objective_metric_sign=['min', 'max'],                    # Maximize validation accuracy
    optimizer_class=OptimizerOptuna,                         # Optuna search strategy to perform robust and efficient hyperparameter optimization at scale
    max_number_of_concurrent_tasks=2,                        # Limit concurrent tasks to manage resources
    execution_queue=args['execution_queue'],                 # The execution queue to use for launching Tasks (experiments)
    optimization_time_limit=None,                            # Maximum minutes for the entire optimization process
    save_top_k_tasks_only=1,                                 # Top K performing Tasks will be kept, the others will be archived
    max_iteration_per_job=args['max_iteration_per_job'],     # Maxiumum number of reported iterations for the specified objective
    total_max_jobs=args['total_max_jobs'],                   # Maximum number of jobs to launch for the optimization
    pool_period_min=1.0,                                     # Check the experiments every 1 min
    time_limit_per_job=30,                                   # Maximum execution time per single job in minutes
)

# optimizer.set_report_period(5.0) # Report every 5 minutes
hpo.start(job_complete_callback=job_complete_callback)
hpo.wait() # Wait until process is done (notice we are controlling the optimization process in the background)

# Log the best parameters
best_hpo_job = hpo.get_top_experiments(top_k=1)
if best_hpo_job:
    best_hpo_job = best_hpo_job[0]
    best_hpo_metrics = best_hpo_job.get_last_scalar_metrics()['Best Metrics']
    if best_hpo_metrics['val_loss']['last'] < best_metrics['val_loss']['last']:
        best_job_id = best_hpo_job.id
        best_hyperparameters = best_hpo_job.get_parameters()
        best_metrics = best_hpo_metrics
    else: print('No better job found in the optimization process => using the best model from model selection')

    print('Best Job:', best_job_id)
    print('Best HPO Parameters:', best_hyperparameters)
    print('Best Metrics:', best_metrics)
    task.upload_artifact('best_results', artifact_object={
        'best_job_id': best_job_id,
        'best_hyperparameters': best_hyperparameters,
        'best_metrics': best_metrics,
    })
    task.set_parameter('best_job_id', best_job_id)
else: print('No top experiments found.')
hpo.stop() # Make sure background optimization stopped