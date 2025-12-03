import numpy as np

# 加载 .npz 文件
npz_file = np.load('/home/zhzhang/桌面/Grasp/Text2HOI/data/grab/data.npz')

# 查看包含的数组名称
for key in npz_file:
    print(key, npz_file[key])
