import pandas as pd
import json
from tqdm import tqdm

# 读取处理后的数据
df = pd.read_csv("models/data/processed_data.csv")

# 添加新列用于标记是否符合要求
df['is_valid_reconstruction'] = False

# 逐行检查
for index, row in tqdm(df.iterrows(), total=len(df)):
    original_comment = row["content"]
    
    # 跳过没有子评论的行
    if pd.isna(row["sub_comments"]):
        continue
    
    try:
        # 解析子评论JSON
        sub_comments_data = json.loads(row["sub_comments"])
        
        # 提取所有子评论文本并连接
        reconstructed_comment = "".join([item["sub_comment"] for item in sub_comments_data])
        
        # 检查重建的评论是否与原始评论匹配
        is_valid = reconstructed_comment == original_comment
        
        # 更新结果列
        df.at[index, 'is_valid_reconstruction'] = is_valid
        
    except Exception as e:
        print(f"处理行 {index} 时出错: {e}")

# 统计结果
valid_count = df['is_valid_reconstruction'].sum()
total_count = df['sub_comments'].notna().sum()
percentage = (valid_count / total_count * 100) if total_count > 0 else 0

print(f"\n检查结果:")
print(f"总共有效行数: {total_count}")
print(f"正确重建数量: {valid_count}")
print(f"正确率: {percentage:.2f}%")

# 保存结果
df.to_csv("models/data/processed_data.csv", index=False)
print("结果已保存到 models/data/processed_data.csv")
