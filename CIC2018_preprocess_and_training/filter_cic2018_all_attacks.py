import pandas as pd
import os

# 设定你的数据文件夹路径
folder_path = '/SSD/p76111262/CIC-IDS2018/'

# 用于存储所有DataFrame的列表
dfs = []

# 遍历文件夹内的所有文件
for filename in os.listdir(folder_path):
    # 确保只处理包含"2018"且以".csv"结尾的文件
    if '2018' in filename and filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 过滤掉Label为'Benign'的数据行
        df_filtered = df[df['Label'] != 'Benign']
        
        # 将过滤后的DataFrame添加到列表中
        dfs.append(df_filtered)

# 使用concat合并所有的DataFrame
df_concat = pd.concat(dfs)

# 将合并后的DataFrame保存为新的CSV文件
df_concat.to_csv('/SSD/p76111262/attack.csv', index=False)

print("所有2018文件处理完成，合并后的数据已保存。")