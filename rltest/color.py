import os
import h5py
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
import cv2

class DataReader:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir

    def get_max_steps(self, epoch):
        """
        讀取指定世代的最大步數
        :param epoch: 世代號碼
        :return: 最大步數 (int)
        """
        file_path = os.path.join(self.base_dir, f"ep{epoch:03d}_env.h5")
        
        if not os.path.exists(file_path):
            print(f"檔案 {file_path} 不存在")
            return None
        
        with h5py.File(file_path, 'r') as hdf5_file:
            if 'obs' not in hdf5_file:
                print(f"資料集 'obs' 在 {file_path} 中不存在")
                return None
            
            # 獲取 obs 資料集的 shape 的第一個維度作為最大步數
            max_steps = hdf5_file['obs'].shape[0]
            return max_steps

    def load_range_data(self, epoch, slice_obj):
        """
        使用 NumPy 樣式的切片方式讀取指定世代和步數範圍的資料
        :param epoch: 世代號碼
        :param slice_obj: NumPy 樣式的切片物件 (可以是切片或是步數範圍)
        :return: 讀取到的資料 (dict 格式，對應每個 dataset)
        """
        file_path = os.path.join(self.base_dir, f"ep{epoch:03d}_env.h5")
        
        if not os.path.exists(file_path):
            print(f"檔案 {file_path} 不存在")
            return None
        
        with h5py.File(file_path, 'r') as hdf5_file:
            # 確保資料集存在
            required_datasets = ['obs', 'angle_degrees', 'reward', 'reward_list', 'origin_image', 'yolo_boxes', 'yolo_scores', 'yolo_classes']
            for dataset in required_datasets:
                if dataset not in hdf5_file:
                    print(f"資料集 {dataset} 在 {file_path} 中不存在")
                    return None
            
            # 使用切片 (slice) 來讀取範圍內的資料
            data = {
                'obs': hdf5_file['obs'][slice_obj],
                'angle_degrees': hdf5_file['angle_degrees'][slice_obj],
                'reward': hdf5_file['reward'][slice_obj],
                'reward_list': hdf5_file['reward_list'][slice_obj],
                'origin_image': hdf5_file['origin_image'][slice_obj],
                'yolo_boxes': hdf5_file['yolo_boxes'][slice_obj],
                'yolo_scores': hdf5_file['yolo_scores'][slice_obj],
                'yolo_classes': hdf5_file['yolo_classes'][slice_obj],
            }
            
            print(f"成功使用切片讀取世代 {epoch} 的資料")
            return data

# 使用範例
data_reader = DataReader(base_dir="./train_logs/env_data")
epoch = 1

# 取得指定世代的最大步數
max_steps = data_reader.get_max_steps(epoch)
if max_steps is not None:
    print(f"世代 {epoch} 的最大步數為: {max_steps}")

data_range = data_reader.load_range_data(epoch, slice(0, 5000, 10))

if data_range:
    print("成功讀取範圍資料:")
    print("觀察空間 shape:", data_range['obs'].shape)
    print("動作角度:", data_range['angle_degrees'])
    print("獎勵:", data_range['reward'])
else:
    print("無範圍資料")
    
CameraInputs=data_range['origin_image']
YOLO=data_range['obs']
# R=data_range['reward']
RList=data_range['reward_list']
Gain=np.array([1,10,5,2,4,5,10,0.5,0.25,10,100,1])
R=np.dot(RList,Gain)
A=data_range['angle_degrees']


newSize=50
ResizeCamera=np.zeros([len(CameraInputs),newSize,newSize,3])
for i in range(len(CameraInputs)):
    ResizeCamera[i]=cv2.resize(CameraInputs[i],(newSize,newSize))

# CreateColorSpace
colorSpace=np.zeros([256*256*256,3])
for i in range(256*256*256):
    colorSpace[i,0]=i%256
    colorSpace[i,1]=(i//256)%256
    colorSpace[i,2]=(i//256//256)%256
    
    
D=np.reshape(ResizeCamera,[-1,3])






pca = decomposition.PCA(n_components=2)
pca.fit(colorSpace)


X_pca = pca.transform(D)
C_pca = pca.transform(colorSpace)


plt.figure(figsize=(12,10))
plt.scatter(X_pca[:,0],X_pca[:,1],marker='+',c='b', zorder=2)
plt.scatter(C_pca[:,0],C_pca[:,1],marker='x',c='r', zorder=1)


a=np.argmin(C_pca[:,1])

uBound1=np.argmax(C_pca[:,1])
uBound2=np.argmax(C_pca[:,0])
lBound1=np.argmin(C_pca[:,1])
lBound2=np.argmin(C_pca[:,0])

UB=np.argmin(np.linalg.norm(X_pca-C_pca[uBound1],2,1))
RB=np.argmin(np.linalg.norm(X_pca-C_pca[uBound2],2,1))
LB=np.argmin(np.linalg.norm(X_pca-C_pca[lBound1],2,1))
LeftB=np.argmin(np.linalg.norm(X_pca-C_pca[lBound2],2,1))

FarPoint=np.argmax([UB,RB,LB,LeftB])
chosenColor=[uBound1,uBound2,lBound1,lBound2][FarPoint]
print('choose Color : ',colorSpace[chosenColor])


colorError=np.zeros([len(CameraInputs)])
for i in range(len(colorSpace)):
    temp=colorSpace[i].reshape([1,3])-CameraInputs
# plt.imshow(CameraInputs[0])

for i in range(len(CameraInputs)):
    cv2.imshow('Out',CameraInputs[i])
    k=cv2.waitKey(10)
    if k=='q':
        break
cv2.destroyAllWindows()

# o=np.zeros([100,100,3])+np.array([ 0, 15, 68])
# cv2.imshow('Color',o.astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

PicsFlat=CameraInputs.reshape([len(CameraInputs),-1])/255
pcaCamera = decomposition.PCA(n_components=1)
pcaCamera.fit(PicsFlat)

Camera2D= pcaCamera.transform(PicsFlat)


YOLOFlat=YOLO.reshape([len(YOLO),-1])/255
pcaYOLO = decomposition.PCA(n_components=1)
pcaYOLO.fit(YOLOFlat)

YOLO2D= pcaYOLO.transform(YOLOFlat)

# plt.figure()
# plt.scatter(Camera2D[:,0],R)
# plt.figure()
# plt.scatter(YOLO2D[:,0],R)





def estimate_mutual_information(X, Y, n_neighbors=3):

    mi_scores = []
    for y in Y.T:
        mi_score = mutual_info_regression(X, y, n_neighbors=n_neighbors)
        mi_scores.append(np.mean(mi_score))
    
    return np.mean(mi_scores)

# mutual_info = estimate_mutual_information(PicsFlat, R.reshape([len(R),1]))
# print("Mutual Information between X and Y:", mutual_info)

mi01=mutual_info_score(Camera2D[:,0],R)
mi02=mutual_info_score(YOLO2D[:,0],R)
print("\nMutual Information camera and yolo (PCA) are :", mi01,mi02)
print('\n\n')