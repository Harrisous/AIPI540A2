import json
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from llm_api import llm_generate

# 中间文件路径
TEMP_FILE = "models/data/processing_temp.csv"
OUTPUT_FILE = "models/data/processed_data.csv"
# 并行处理的线程数
NUM_WORKERS = 20

# 定义处理单条记录的函数
@retry(
    stop=stop_after_attempt(3),  # 最多重试3次
    wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数退避，2-10秒
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),  # 网络相关错误才重试
    reraise=True,  # 最终失败时重新抛出异常
    before_sleep=lambda retry_state: print(f"重试 {retry_state.attempt_number}/3 记录 {retry_state.args[0][0]}, 等待 {retry_state.next_action.sleep} 秒...")
)
def process_single_record(record):
    index, content = record
    try:
        sub_comments = llm_generate(content)
        serialized = [user.model_dump() for user in sub_comments]
        json_str = json.dumps(serialized, ensure_ascii=False)
        return index, json_str
    except Exception as e:
        print(f"处理记录 {index} 失败: {e}")
        return index, None

# 检查是否存在中间文件
if os.path.exists(TEMP_FILE):
    print(f"找到中间文件，继续处理...")
    df = pd.read_csv(TEMP_FILE)
    # 计算已处理的记录数
    processed = df["sub_comments"].notna().sum()
    print(f"已处理 {processed}/{len(df)} 条记录")
else:
    print("从头开始处理...")
    df = pd.read_csv("models/data/sentiment-analysis-dataset-google-play-app-reviews.csv")
    # 创建新列
    df["sub_comments"] = None
    # 保存初始状态
    df.to_csv(TEMP_FILE, index=False)

try:
    # 获取未处理的记录
    unprocessed_df = df[df["sub_comments"].isna()]
    
    if len(unprocessed_df) > 0:
        print(f"开始并行处理 {len(unprocessed_df)} 条记录，使用 {NUM_WORKERS} 个工作线程...")
        # 准备任务列表
        tasks = [(index, row["content"]) for index, row in unprocessed_df.iterrows()]
        
        # 创建进度条
        pbar = tqdm(total=len(tasks))
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # 提交所有任务
            future_to_index = {executor.submit(process_single_record, task): task[0] for task in tasks}
            
            # 处理完成的任务
            for future in as_completed(future_to_index):
                index, result = future.result()
                if result is not None:
                    df.at[index, "sub_comments"] = result
                
                # 更新进度条
                pbar.update(1)
                
                # 每完成一条记录就保存进度
                if pbar.n % 5 == 0 or pbar.n == len(tasks):  # 每5条记录保存一次，减少I/O操作
                    df.to_csv(TEMP_FILE, index=False)
                
                # 添加短暂延迟，防止API请求过于频繁
                # time.sleep(0.1)
            
            pbar.close()
    
    # 全部处理完成后，保存到最终文件并删除中间文件
    df.to_csv(OUTPUT_FILE, index=False)
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)
    print("处理完成！已删除中间文件。")
    
except Exception as e:
    # 发生异常时保存当前进度
    df.to_csv(TEMP_FILE, index=False)
    print(f"处理中断: {e}")
    print(f"进度已保存到 {TEMP_FILE}，下次运行时将从断点继续。")
