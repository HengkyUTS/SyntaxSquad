from clearml import Task, PipelineController
from clearml.automation import TriggerScheduler


# def get_test_accuracy(reported_plots):
#     metrics_report = next((d for d in reported_plots if '3 subsets' in d.get('metric').lower()), None)
#     metrics_report = json.loads(metrics_report['plot_str'])['data'][0]
#     df = pd.DataFrame(list(zip(*metrics_report['cells']['values'])), columns=[col[0] for col in metrics_report['header']['values']])
#     return df.loc[df['Subset'] == 'Test', 'Accuracy'].values[0]

def on_performance_drop(model_evaluation_task_id, ml_pipeline, threshold=0.6):
    model_evaluation_task = Task.get_task(task_id=model_evaluation_task_id)
    test_accuracy = model_evaluation_task.get_reported_single_value('test_accuracy')
    if test_accuracy < threshold: # Trigger if accuracy < 60%
        print(f'Test Accuracy {test_accuracy} below threshold {threshold} => Triggering retraining.')
        ml_pipeline.start_locally() # Start the pipeline inside the same machine of the schedule_queue
        

trigger = TriggerScheduler( # Set up event-based triggers to monitor events for continuous training
    pooling_frequency_minutes=1, # Checking system state every minute
    sync_frequency_minutes=15, # Sync task scheduler configuration every X minutes. Allow to change in runtime by editing Task configuration
    force_create_task_name='Trigger to retrain on new dataset arrival and performance drop',
    force_create_task_project='SyntaxSquad',
)

ml_pipeline = PipelineController.get( # Pipeline to clone when a new dataset version drops
    pipeline_project='SyntaxSquad', 
    pipeline_name='SyntaxSquad ML Pipeline', 
    pipeline_tags=['production']
)

# Add the actual trigger and enqueue the pipeline in Remote_CPU because we're cloning the
# PipelineController. The controller itself will properly enqueue the different nodes itself.
trigger.add_dataset_trigger( # Add trigger on dataset creation
    name='Trigger 1: Retrain on new dataset arrival', 
    schedule_task_id=ml_pipeline.id, 
    schedule_queue='Remote_CPU', # Cannot use the same queue with pipeline's tasks: https://github.com/clearml/clearml/issues/1328
    trigger_project='SyntaxSquad',
    trigger_on_tags=['landmarks', 'stable'],
    task_overrides={
        'Args/wlasl_landmarks_dataset_id': '${dataset.id}', # Use the dataset ID from the trigger
    },
    single_instance=True, # Not launch the Task job if the previous instance is still running
)

trigger.add_task_trigger( # Add trigger to monitor 'Test Accuracy' scalar from the Model evaluation task
    name='Trigger 2: Retrain if test accuracy below 60%',
    schedule_queue='Remote_CPU', # Cannot use the same queue with pipeline's tasks: https://github.com/clearml/clearml/issues/1328
    schedule_function=lambda model_evaluation_task_id: on_performance_drop(model_evaluation_task_id, ml_pipeline, threshold=0.6),
    trigger_project='SyntaxSquad',
    trigger_name='step7_model_evaluation',
    trigger_on_status=['completed'], # Trigger if the task is completed
    trigger_on_tags=ml_pipeline.tags, # Trigger when all tags in the list are present
    single_instance=True, # Not launch the Task job if the previous instance is still running
)

trigger.start()
# trigger.start_remotely(queue='Remote_CPU') # Start the trigger remotely, so an agent will keep track of everything