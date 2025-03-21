#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
获取数据模块：负责获取原始数据并存储到data/raw目录
"""

import os
import shutil
import pandas as pd

def get_data():
    """
    获取数据并保存到data/raw目录
    
    如果数据已经存在，该函数不会重复获取
    """
    # 定义数据路径
    raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    raw_data_file = os.path.join(raw_data_dir, 'sentiment-analysis-dataset-google-play-app-reviews.csv')
    
    # 检查数据是否已经存在
    if os.path.exists(raw_data_file):
        print(f"数据文件已存在: {raw_data_file}")
        return
    
    # 如果需要从网络下载数据，可以在这里添加下载代码
    # 例如使用requests库下载
    # import requests
    # url = "https://example.com/dataset.csv"
    # response = requests.get(url)
    # with open(raw_data_file, 'wb') as f:
    #     f.write(response.content)
    
    print(f"请手动下载数据文件并放置到: {raw_data_file}")

def check_data():
    """
    检查原始数据是否存在并可用
    
    返回:
        bool: 数据是否可用
    """
    raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    raw_data_file = os.path.join(raw_data_dir, 'sentiment-analysis-dataset-google-play-app-reviews.csv')
    
    if not os.path.exists(raw_data_file):
        print(f"错误: 原始数据文件不存在: {raw_data_file}")
        return False
        
    try:
        # 尝试读取文件，确保格式正确
        df = pd.read_csv(raw_data_file)
        print(f"数据集包含 {len(df)} 条记录")
        print(f"数据列: {df.columns.tolist()}")
        return True
    except Exception as e:
        print(f"读取数据文件时出错: {e}")
        return False

if __name__ == "__main__":
    get_data()
    check_data() 