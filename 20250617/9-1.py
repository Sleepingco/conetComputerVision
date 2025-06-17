import os
import trimesh
import tensorflow as tf
import matplotlib.pyplot as plt

classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
           'monitor', 'night_stand', 'sofa', 'table', 'toilet']

# 다운로드 및 압축 해제
url = 'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
zip_path = tf.keras.utils.get_file('modelnet.zip', url, extract=True)

# 압축이 풀린 디렉터리 위치
data_dir = os.path.join(os.path.dirname(zip_path), 'ModelNet10')

# 시각화
fig = plt.figure(figsize=(50, 5))
for i in range(len(classes)):
    off_path = os.path.join(data_dir, classes[i], 'train', f'{classes[i]}_0001.off')
    if not os.path.exists(off_path):
        print(f"파일 없음: {off_path}")
        continue

    mesh = trimesh.load(off_path)
    points = mesh.sample(4096)

    ax = fig.add_subplot(1, 10, i + 1, projection='3d')
    ax.set_title(classes[i], fontsize=20)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='g')

plt.tight_layout()
plt.show()
