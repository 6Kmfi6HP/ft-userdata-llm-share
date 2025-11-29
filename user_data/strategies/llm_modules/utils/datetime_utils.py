"""
日期时间工具模块
提供统一的时区处理函数，避免代码重复
"""
from datetime import datetime, timezone
from typing import Tuple, Optional


def normalize_timestamps(
    entry_time: Optional[datetime], 
    exit_time: Optional[datetime]
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    统一两个时间对象的时区信息
    
    确保两个时间都是 timezone-aware 或都是 timezone-naive，
    避免在时间计算时出现 TypeError。
    
    Args:
        entry_time: 入场时间
        exit_time: 出场时间
        
    Returns:
        统一时区后的 (entry_time, exit_time) 元组
    """
    if entry_time is None or exit_time is None:
        return entry_time, exit_time
    
    if entry_time.tzinfo is None and exit_time.tzinfo is not None:
        # entry_time 是 naive，exit_time 是 aware，统一为 aware
        entry_time = entry_time.replace(tzinfo=timezone.utc)
    elif entry_time.tzinfo is not None and exit_time.tzinfo is None:
        # entry_time 是 aware，exit_time 是 naive，统一为 aware
        exit_time = exit_time.replace(tzinfo=timezone.utc)
    
    return entry_time, exit_time

