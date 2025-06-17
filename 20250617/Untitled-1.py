# %%
import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# %%
classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
           'monitor', 'night_stand', 'sofa', 'table', 'toilet']

url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
zip_path = tf.keras.utils.get_file('modelnet.zip', url, extract=True)
data_dir = os.path.join(os.path.dirname(zip_path), 'ModelNet10')

# %%
def parse_dataset(num_points=2048):
    train_points, train_labels = [], []
    test_points, test_labels = [], []

    for i, class_name in enumerate(classes):
        folder = os.path.join(data_dir, class_name)
        print('데이터 읽기:', class_name)

        train_files = glob.glob(os.path.join(folder, 'train', '*.off'))
        test_files = glob.glob(os.path.join(folder, 'test', '*.off'))

        for f in train_files:
            try:
                sample = trimesh.load(f).sample(num_points)
                if not np.isnan(sample).any():
                    train_points.append(sample)
                    train_labels.append(i)
            except Exception as e:
                print(f"train 에러: {f}, {e}")

        for f in test_files:
            try:
                sample = trimesh.load(f).sample(num_points)
                if not np.isnan(sample).any():
                    test_points.append(sample)
                    test_labels.append(i)
            except Exception as e:
                print(f"test 에러: {f}, {e}")

    return (
        np.array(train_points, dtype=np.float32),
        np.array(test_points, dtype=np.float32),
        np.array(train_labels, dtype=np.int32),
        np.array(test_labels, dtype=np.int32)
    )

# %%
NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32

x_train, x_test, y_train, y_test = parse_dataset(NUM_POINTS)

# %%
def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding='valid')(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation('relu')(x)

def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation('relu')(x)

# %%
class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.matmul(x, x, transpose_b=True)
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

# %%
def tnet(inputs, num_features):
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(num_features * num_features,
                     kernel_initializer='zeros',
                     bias_initializer=bias,
                     activity_regularizer=reg)(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

# %%
inputs = keras.Input(shape=(NUM_POINTS, 3))
x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='pointnet')

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
    run_eagerly=False  # 필요시 True로 설정
)

model.summary()

# %%
model.fit(x_train, y_train, epochs=20, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))

# %%
# 예측 테스트
chosen = np.random.randint(0, len(x_test), 8)
points = x_test[chosen]
labels = y_test[chosen]

preds = model.predict(points)
preds = tf.math.argmax(preds, axis=-1)

# %%
# 시각화
fig = plt.figure(figsize=(15, 4))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection='3d')
    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2], s=1, c='g')
    ax.set_title(f"pred: {classes[preds[i].numpy()]}\nGT: {classes[labels[i]]}", fontsize=12)
    ax.set_axis_off()

plt.tight_layout()
plt.show()
