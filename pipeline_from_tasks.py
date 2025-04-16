from clearml.automation import PipelineController

def pre_execute_callback(a_pipeline, a_node, current_param_override):
    # type (PipelineController, PipelineController.Node, dict) -> bool
    print(f'Cloning Task id={a_node.base_task_id} with parameters: {current_param_override}')
    # if we want to skip this node (and subtree of this node) we return False
    # return True to continue DAG execution
    return True

def post_execute_callback(a_pipeline, a_node):
    # type (PipelineController, PipelineController.Node) -> None
    print(f'Completed Task id={a_node.executed}')
    # if we need the actual executed Task: Task.get_task(task_id=a_node.executed)
    return


pipe = PipelineController(name='SyntaxSquad ML Pipeline', project='SyntaxSquad', add_pipeline_tags=True)
pipe.add_parameter('wlasl_landmarks_dataset_id', default='3b222a4667044677b2f7bc0a628ea9f4', description='WLASL landmarks dataset ID')
pipe.add_parameter('chosen_landmarks', default=None, description='Landmarks to use for training')
pipe.add_parameter('chosen_labels', default=None, description='Labels to use for training')
pipe.add_parameter('max_labels', default=100, description='Maximum number of labels to use for training')
pipe.add_parameter('max_samples', default=None, description='Maximum number of samples to use for training')
pipe.add_parameter('max_frames', default=195, description='Maximum number of frames to use for training')
pipe.add_parameter('pad_value', default=-100, description='Padding value for frames')
pipe.add_parameter('batch_size', default=128, description='Batch size for training')
pipe.add_parameter('epochs', default=100, description='Number of epochs for training')
pipe.add_parameter('learning_rate', default=0.001, description='Learning rate for AdamW optimizer')
pipe.add_parameter('conv1d_dropout', default=0.2, description='Dropout rate for Conv1DBlock layers')
pipe.add_parameter('last_dropout', default=0.2, description='Dropout rate before the final layer')
pipe.add_parameter('weights_name', default='wlasl100.h5', description='Weights file name')
pipe.add_parameter('reduce_lr_patience', default=5, description='Patience for ReduceLROnPlateau')
pipe.add_parameter('reduce_lr_min_lr', default=1e-6, description='Minimum learning rate for ReduceLROnPlateau')
pipe.add_parameter('reduce_lr_factor', default=0.7, description='Factor for ReduceLROnPlateau')
pipe.set_default_execution_queue('SyntaxSquad_Queue')

pipe.add_step( # Step 1: Split landmarks dataset into train/val/test and perform statistics
    name='step1_data_splitting',
    base_task_project='SyntaxSquad',
    base_task_name='step1_data_splitting',
    parameter_override={
        'General/wlasl_landmarks_dataset_id': '${pipeline.wlasl_landmarks_dataset_id}',
        'General/chosen_landmarks': '${pipeline.chosen_landmarks}',
        'General/chosen_labels': '${pipeline.chosen_labels}',
        'General/max_labels': '${pipeline.max_labels}',
        'General/max_samples': '${pipeline.max_samples}',
    },
)

pipe.add_step( # Step 2: Perform random data augmentation on train set
    name='step2_data_augmentation',
    parents=['step1_data_splitting'],
    base_task_name='step2_data_augmentation',
    base_task_project='SyntaxSquad',
    parameter_override={
        'General/data_splitting_task_id': '${step1_data_splitting.id}',
    },
    pre_execute_callback=pre_execute_callback,
    post_execute_callback=post_execute_callback,
)

pipe.add_step( # Step 3: Perform padding or truncation on X_train/X_val/X_test and label encoding on y_train/y_val/y_test
    name='step3_data_transformation',
    parents=['step1_data_splitting', 'step2_data_augmentation'],
    base_task_name='step3_data_transformation',
    base_task_project='SyntaxSquad',
    parameter_override={
        'General/data_splitting_task_id': '${step1_data_splitting.id}',
        'General/data_augmentation_task_id': '${step2_data_augmentation.id}',
        'General/max_frames': '${pipeline.max_frames}',
        'General/pad_value': '${pipeline.pad_value}',
    },
    pre_execute_callback=pre_execute_callback,
    post_execute_callback=post_execute_callback,
)

pipe.add_step( # Step 4: Prepare TF dataset with nose normalization and train the model
    name='step4_model_training',
    parents=['step3_data_transformation'],
    base_task_name='step4_model_training',
    base_task_project='SyntaxSquad',
    parameter_override={
        'General/data_transformation_task_id': '${step3_data_transformation.id}',
        'General/max_frames': '${pipeline.max_frames}',
        'General/pad_value': '${pipeline.pad_value}',
        'General/batch_size': '${pipeline.batch_size}',
        'General/epochs': '${pipeline.epochs}',
        'General/learning_rate': '${pipeline.learning_rate}',
        'General/conv1d_dropout': '${pipeline.conv1d_dropout}',
        'General/last_dropout': '${pipeline.last_dropout}',
        'General/weights_name': '${pipeline.weights_name}',
        'General/reduce_lr_patience': '${pipeline.reduce_lr_patience}',
        'General/reduce_lr_min_lr': '${pipeline.reduce_lr_min_lr}',
        'General/reduce_lr_factor': '${pipeline.reduce_lr_factor}',
    },
    pre_execute_callback=pre_execute_callback,
    post_execute_callback=post_execute_callback,
)

pipe.add_step( # Step 5: Evaluate the model
    name='step5_model_evaluation',
    parents=['step3_data_transformation', 'step4_model_training'],
    base_task_name='step5_model_evaluation',
    base_task_project='SyntaxSquad',
    parameter_override={
        'General/data_transformation_task_id': '${step3_data_transformation.id}',
        'General/model_training_task_id': '${step4_model_training.id}',
        'General/max_frames': 195,
        'General/pad_value': -100,
        'General/learning_rate': 0.001,
        'General/batch_size': 128,
        'General/conv1d_dropout': 0.2,
        'General/last_dropout': 0.2,
    },
    pre_execute_callback=pre_execute_callback,
    post_execute_callback=post_execute_callback,
)

pipe.start(queue='SyntaxSquad_Queue')