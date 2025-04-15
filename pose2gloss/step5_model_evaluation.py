from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
from model_utils import build_GISLR, prepare_tf_dataset
from clearml import Task

# Initialize the ClearML task
task = Task.init(project_name='SyntaxSquad', task_name='step5_model_evaluation', task_type=Task.TaskTypes.testing)
args = {
    'data_transformation_task_id': '775f62600cb64fd0bae2404a31084177',
    'model_training_task_id': '89b93c02537141b1aeb270fb1d6f0fc1',
    'max_frames': 195,
    'pad_value': -100,
    'num_landmarks': 180,
    'num_glosses': 100,
    'learning_rate': 0.001,
    'batch_size': 128,
    'conv1d_dropout': 0.2,
    'last_dropout': 0.2,
}
task.connect(args)
task.execute_remotely()

# Get test data
data_transformation_task = Task.get_task(task_id=args['data_transformation_task_id'])
X_test, y_test = data_transformation_task.artifacts['X_train'].get(), data_transformation_task.artifacts['y_train'].get()
test_tf_dataset = prepare_tf_dataset(X_test, y_test, batch_size=args['batch_size'], shuffle=True)

# Build and compile the model
model = build_GISLR(
    args['max_frames'], num_landmarks=args['num_landmarks'], num_glosses=args['num_glosses'], 
    pad_value=args['pad_value'], conv1d_dropout=args['conv1d_dropout'], 
    last_dropout=args['last_dropout'], is_training=False,
)
model.compile(
    optimizer=AdamW(learning_rate=args['learning_rate']),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy', SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy')],
)
model.summary()

# Load the model from the previous task and evaluate it
model_training_task = Task.get_task(task_id=args['model_training_task_id'])
weights_path = model_training_task.models['output'][-1].get_local_copy() # Last snapshot
model.load_weights(weights_path)
model.evaluate(test_tf_dataset, batch_size=args['batch_size'], verbose=1)