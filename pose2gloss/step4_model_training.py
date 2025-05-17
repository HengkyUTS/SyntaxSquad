import tensorflow as tf
import tensorflow.keras.mixed_precision as mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
from clearml import Task, OutputModel
from model_utils import build_and_compile_GISLR, prepare_tf_dataset

# Initialize the ClearML task
Task.add_requirements('nvidia-cudnn-cu12', '9.3.0.75')
task = Task.init(
    project_name='SyntaxSquad', task_type=Task.TaskTypes.training,
    task_name='Step 4: Prepare TF dataset with nose normalization and train the model'
)
args = {
    'data_transformation_task_id': '', # ID of the task that performed data transformation
    'max_frames': 195, # Maximum number of frames for padding/truncating
    'pad_value': -100, # Value to pad with
    'batch_size': 128, # Batch size for training
    'epochs': 100, # Number of epochs for training
    'learning_rate': 0.001, # Learning rate for the optimizer
    'conv1d_dropout': 0.2, # Dropout rate for Conv1DBlock layers
    'last_dropout': 0.2, # Dropout rate before the final layer
    'weights_name': 'wlasl100.keras', # Weights file name
    'reduce_lr_patience': 5, # Patience for ReduceLROnPlateau
    'reduce_lr_min_lr': 1e-6, # Minimum learning rate for ReduceLROnPlateau
    'reduce_lr_factor': 0.7, # Factor for ReduceLROnPlateau
}
task.connect(args)
task.execute_remotely()

# Set GPU memory growth to avoid OOM errors
if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('Using GPU:', tf.test.gpu_device_name())
# else: raise ValueError('Running on CPU is not recommended.')
else: print('Running on CPU')

# Mixed precision training
try: mixed_precision.set_global_policy(mixed_precision.Policy('mixed_float16'))
except: mixed_precision.set_global_policy(mixed_precision.Policy('mixed_bfloat16'))
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)

# Load the data from the previous tasks
data_transformation_task = Task.get_task(task_id=args['data_transformation_task_id'])
X_train, y_train = data_transformation_task.artifacts['X_train'].get(), data_transformation_task.artifacts['y_train'].get()
X_val, y_val = data_transformation_task.artifacts['X_val'].get(), data_transformation_task.artifacts['y_val'].get()
num_landmarks, num_glosses = X_train.shape[-2], len(set(y_train))

# Prepare the TF dataset for efficient data loading
train_tf_dataset = prepare_tf_dataset(X_train, y_train, batch_size=args['batch_size'], shuffle=True)
val_tf_dataset = prepare_tf_dataset(X_val, y_val, batch_size=args['batch_size'], shuffle=False)

# Build and compile the model
model = build_and_compile_GISLR(
    args['max_frames'], num_landmarks=num_landmarks, num_glosses=num_glosses, pad_value=args['pad_value'], 
    conv1d_dropout=args['conv1d_dropout'], last_dropout=args['last_dropout'], learning_rate=args['learning_rate'],
)
model.summary()

# Train the model
history = model.fit(train_tf_dataset, validation_data=val_tf_dataset, callbacks = [
    ModelCheckpoint(args['weights_name'], monitor='val_accuracy', mode='max', save_best_only=True),
    ReduceLROnPlateau(
        monitor = 'val_accuracy', mode = 'max',
        patience = args['reduce_lr_patience'], # Reduce if no improvement after 5 epochs
        min_lr = args['reduce_lr_min_lr'], # Lower bound on the learning rate
        factor = args['reduce_lr_factor'], # => new_lr = lr * factor
        verbose = 1
    ),
    LambdaCallback(on_epoch_end=lambda epoch, logs: [
        task.logger.report_scalar(title='Loss', value=logs['loss'], iteration=epoch, series='Training'),
        task.logger.report_scalar(title='Loss', value=logs['val_loss'], iteration=epoch, series='Validation'),
        task.logger.report_scalar(title='Accuracy', value=logs['accuracy'], iteration=epoch, series='Training'),
        task.logger.report_scalar(title='Accuracy', value=logs['val_accuracy'], iteration=epoch, series='Validation'),
        task.logger.report_scalar(title='Top 5 Accuracy', value=logs['top5_accuracy'], iteration=epoch, series='Training'),
        task.logger.report_scalar(title='Top 5 Accuracy', value=logs['val_top5_accuracy'], iteration=epoch, series='Validation'),
    ]),
], epochs=args['epochs'], verbose=1)

# Save the model and training history
output_model = OutputModel(task=task)
output_model.update_weights(args['weights_name'], upload_uri='https://files.clear.ml')
output_model.publish()

# Calculate the validation metrics with the best weights for HPO
model.load_weights(args['weights_name'])
val_loss, val_accuracy, val_top5_accuracy = model.evaluate(val_tf_dataset, batch_size=args['batch_size'], verbose=1)
task.logger.report_scalar(title='Optimization Metric', value=val_loss, iteration=args['epochs'], series='val_loss')
task.logger.report_scalar(title='Optimization Metric', value=val_accuracy, iteration=args['epochs'], series='val_accuracy')
task.logger.report_scalar(title='Optimization Metric', value=val_top5_accuracy, iteration=args['epochs'], series='val_top5_accuracy')
print(f'Best val_loss: {val_loss}, Best val_accuracy: {val_accuracy}, Best val_top5_accuracy: {val_top5_accuracy}')