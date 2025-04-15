import tensorflow as tf
import tensorflow.keras.mixed_precision as mixed_precision
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
from clearml import Task, OutputModel
from model_utils import build_GISLR, prepare_tf_dataset

# Initialize the ClearML task
task = Task.init(project_name='SyntaxSquad', task_name='step4_model_training', task_type=Task.TaskTypes.training)
args = {
    'data_transformation_task_id': '775f62600cb64fd0bae2404a31084177',
    'max_frames': 195,
    'pad_value': -100,
    'num_landmarks': 180,
    'num_glosses': 100,
    'batch_size': 128,
    'epochs': 100,
    'learning_rate': 0.001,
    'conv1d_dropout': 0.2,
    'last_dropout': 0.2,
    'weights_name': 'wlasl100.h5',
    'reduce_lr_patience': 5,
    'reduce_lr_min_lr': 1e-6,
    'reduce_lr_factor': 0.7,
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
train_tf_dataset = prepare_tf_dataset(X_train, y_train, batch_size=args['batch_size'], shuffle=True)
val_tf_dataset = prepare_tf_dataset(X_val, y_val, batch_size=args['batch_size'], shuffle=False)

# Build and compile the model
model = build_GISLR(
    args['max_frames'], num_landmarks=args['num_landmarks'], num_glosses=args['num_glosses'], 
    pad_value=args['pad_value'], conv1d_dropout=args['conv1d_dropout'], last_dropout=args['last_dropout'],
)
model.compile(
    optimizer=AdamW(learning_rate=args['learning_rate']),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy', SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy')],
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
        task.logger.report_scalar(title='Training Loss', value=logs['loss'], iteration=epoch),
        task.logger.report_scalar(title='Validation Loss', value=logs['val_loss'], iteration=epoch),
        task.logger.report_scalar(title='Training Accuracy', value=logs['accuracy'], iteration=epoch),
        task.logger.report_scalar(title='Validation Accuracy', value=logs['val_accuracy'], iteration=epoch),
        task.logger.report_scalar(title='Top 5 Accuracy', value=logs['top5_accuracy'], iteration=epoch),
        task.logger.report_scalar(title='Top 5 Validation Accuracy', value=logs['val_top5_accuracy'], iteration=epoch)
    ]),
], epochs=args['epochs'], verbose=1).history

# Save the model and training history
output_model = OutputModel(task=task)
output_model.update_weights(args['weights_name'], upload_uri='https://files.clear.ml')
output_model.publish()
task.upload_artifact('training_history', artifact_object=history)