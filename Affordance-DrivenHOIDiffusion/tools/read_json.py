import json

# 假设你有一个名为 'data.json' 的 JSON 文件
file_path = '/home/zhzhang/桌面/Grasp/Text2HOI/data/grab/text.json'

# 打开文件并读取 JSON 数据
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
cnt=0
# 打印解析后的数据
for i in data.items():
    print(i)
    cnt+=1
print(cnt,len(data))