# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: AlbertMao
# Date: 2025-03-07
# Vesion: 1.0

import random

class RandomUtils:
    @staticmethod
    def generate_unique_numbers(value_range, count):
        """
        生成指定范围内的不同随机数。

        :param value_range: (tuple) 指定的范围 (min, max)
        :param count: (int) 生成的随机数个数
        :return: list 生成的随机数列表
        """
        min_val, max_val = value_range

        # 检查是否可能生成足够的唯一数
        if max_val - min_val + 1 < count:
            raise ValueError("范围内的数不足以生成指定个数的唯一值")

        return random.sample(range(min_val, max_val + 1), count)


