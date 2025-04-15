import tensorflow as tf
import tensorflow.keras.mixed_precision as mixed_precision
from tensorflow.keras.layers import (
    Conv1D, Dropout, ZeroPadding1D, DepthwiseConv1D, Dense, BatchNormalization,
    MultiHeadAttention, Reshape, Add, Masking, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from clearml import Task, OutputModel

# Initialize the ClearML task
task = Task.init(project_name='SyntaxSquad', task_name='step4_model_training', task_type=Task.TaskTypes.training, auto_connect_frameworks='keras')
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
    'model_file_name': 'wlasl100.h5',
    'reduce_lr_patience': 5,
    'reduce_lr_min_lr': 1e-6,
    'reduce_lr_factor': 0.7,
}
task.connect(args)
task.execute_remotely()


if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('Using GPU:', tf.test.gpu_device_name())
# else: raise ValueError('Running on CPU is not recommended.')
else: print('Running on CPU')


# Define model components
class EfficientChannelAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = Conv1D(1, kernel_size=kernel_size, strides=1, padding='same', use_bias=False)

    def call(self, inputs, mask=None):
        x = GlobalAveragePooling1D()(inputs, mask=mask)
        x = self.conv(tf.expand_dims(x, -1))
        x = tf.squeeze(x, -1)
        return inputs * tf.sigmoid(x)[:, None, :]


class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(self, kernel_size=17, dilation_rate=1, use_bias=False, depthwise_initializer='glorot_uniform', name='', **kwargs):
        super().__init__(name=name, **kwargs)
        self.causal_pad = ZeroPadding1D((dilation_rate * (kernel_size - 1), 0), name=name + '_pad')
        self.dw_conv = DepthwiseConv1D(
            kernel_size, strides=1, dilation_rate=dilation_rate, padding='valid', use_bias=use_bias,
            depthwise_initializer=depthwise_initializer, name=name + '_dwconv'
        )
        self.supports_masking = True

    def call(self, inputs):
        x = self.causal_pad(inputs)
        return self.dw_conv(x)


def Conv1DBlock(channel_size, kernel_size, dilation_rate=1, drop_rate=0.2, expand_ratio=2, activation='swish', name=None):
    if name is None: name = str(tf.keras.backend.get_uid('mbblock'))
    def apply(inputs): # Expansion phase
        channels_in = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio
        skip = inputs

        x = Dense(channels_expand, use_bias=True, activation=activation, name=name + '_expand_conv')(inputs)
        x = CausalDWConv1D(kernel_size, dilation_rate=dilation_rate, use_bias=False, name=name + '_dwconv')(x) # Depthwise Convolution
        x = BatchNormalization(momentum=0.95, name=name + '_bn')(x)
        x = EfficientChannelAttention()(x)
        x = Dense(channel_size, use_bias=True, name=name + '_project_conv')(x)
        x = Dropout(drop_rate, noise_shape=(None,1,1), name=name + '_drop')(x)
        return tf.keras.layers.add([x, skip], name=name + '_add') if channels_in == channel_size else x
    return apply


def TransformerBlock(dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2):
    def apply(inputs):
        x = BatchNormalization(momentum=0.95)(inputs)
        x = MultiHeadAttention(key_dim=dim // num_heads, num_heads=num_heads, dropout=attn_dropout)(x, x)
        x = Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = Add()([inputs, x])
        attn_out = x

        x = BatchNormalization(momentum=0.95)(x)
        x = Dense(dim*expand, use_bias=False, activation='swish')(x)
        x = Dense(dim, use_bias=False)(x)
        x = Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        return Add()([attn_out, x])
    return apply


def Conv1DTransformerBlock(x, dim, kernel_size, drop_rate=0.2):
    x = Conv1DBlock(dim, kernel_size, drop_rate=drop_rate)(x)
    x = Conv1DBlock(dim, kernel_size, drop_rate=drop_rate)(x)
    x = Conv1DBlock(dim, kernel_size, drop_rate=drop_rate)(x)
    x = TransformerBlock(dim, expand=2)(x)

    x = Conv1DBlock(dim, kernel_size, drop_rate=drop_rate)(x)
    x = Conv1DBlock(dim, kernel_size, drop_rate=drop_rate)(x)
    x = Conv1DBlock(dim, kernel_size, drop_rate=drop_rate)(x)
    return TransformerBlock(dim, expand=2)(x)


def build_GISLR(
    max_frames, num_landmarks=180, num_glosses=100, pad_value=-100, 
    dim=192, kernel_size=17, conv1d_dropout=0.2, last_dropout=0.2, is_training=True
):
    inputs = tf.keras.Input((max_frames, num_landmarks, 3)) # 180 landmarks with 3 coordinates
    x = Reshape((max_frames, num_landmarks * 3))(inputs)

    if is_training: x = Masking(mask_value=pad_value)(x)
    x = Dense(dim, use_bias=False,name='stem_conv')(x)
    x = BatchNormalization(momentum=0.95)(x)
    x = Conv1DTransformerBlock(x, dim, kernel_size, conv1d_dropout)
    x = Dense(dim * 2, activation=None, name='top_conv')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(last_dropout)(x)
    x = Dense(num_glosses, name='output')(x)
    return tf.keras.Model(inputs, x)


def pose_based_normalize(video_landmarks, label, scale_by_shoulder=False):
    nose_center = video_landmarks[:, 49, :]
    translated_landmarks = video_landmarks - nose_center[:, None, :]

    if scale_by_shoulder: # Use the distance between left shoulder (42) and right shoulder (43)
        left_shoulder = video_landmarks[:, 42, :]
        right_shoulder = video_landmarks[:, 43, :]
        shoulder_distance = tf.norm(left_shoulder - right_shoulder, axis=-1, keepdims=True)
        shoulder_distance = tf.where(shoulder_distance == 0, tf.ones_like(shoulder_distance), shoulder_distance)
        return translated_landmarks / shoulder_distance[:, None, :], label
    return translated_landmarks, label # tf.one_hot(label, len(np.unique(y_train)))


def prepare_tf_dataset(X, y, batch_size=64, drop_remainder=False, shuffle=True, use_cache=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).map(pose_based_normalize, tf.data.AUTOTUNE)
    if shuffle: dataset = dataset.shuffle(len(X))
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # When use .cache(), everything before is saved in the memory. It gives a
    # significant boost in speed but only if you can get your hands on a larger RAM
    if use_cache: dataset = dataset.cache()
    return dataset.prefetch(tf.data.AUTOTUNE)


# Load the data from the previous tasks
data_transformation_task = Task.get_task(task_id=args['data_transformation_task_id'])
X_train, y_train = data_transformation_task.artifacts['X_train'].get(), data_transformation_task.artifacts['y_train'].get()
X_val, y_val = data_transformation_task.artifacts['X_val'].get(), data_transformation_task.artifacts['y_val'].get()
train_tf_dataset = prepare_tf_dataset(X_train, y_train, batch_size=args['batch_size'], shuffle=True)
val_tf_dataset = prepare_tf_dataset(X_val, y_val, batch_size=args['batch_size'], shuffle=False)

try: mixed_precision.set_global_policy(mixed_precision.Policy('mixed_float16'))
except: mixed_precision.set_global_policy(mixed_precision.Policy('mixed_bfloat16'))
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)

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
    ModelCheckpoint(args['model_file_name'], monitor='val_accuracy', mode='max', save_best_only=True),
    ReduceLROnPlateau(
        monitor = 'val_accuracy', mode = 'max',
        patience = args['reduce_lr_patience'], # Reduce if no improvement after 5 epochs
        min_lr = args['reduce_lr_min_lr'], # Lower bound on the learning rate
        factor = args['reduce_lr_factor'], # => new_lr = lr * factor
        verbose = 1
    )
], epochs=args['epochs'], verbose=1).history

# Save the model and training history
output_model = OutputModel(task=task)
output_model.update_weights(args['model_file_name'], upload_uri='https://files.clear.ml')
output_model.publish()
task.upload_artifact('trained_model', artifact_object=args['model_file_name'])
task.upload_artifact('training_history', artifact_object=history)