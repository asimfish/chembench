# 测试日志目录

此目录用于存储批量测试的结果日志文件。

## 日志文件命名规则

自动生成的日志文件命名格式：`test_results_YYYYMMDD_HHMMSS.log`

例如：
- `test_results_20251231_150000.log`
- `test_results_20251231_160530.log`

## 日志内容

每个日志文件包含：

### 1. 测试配置信息
- 生成时间
- 测试参数（num_envs, max_episode, max_step）

### 2. 统计概览
- 总计任务数
- 成功/失败任务数
- 任务完成率

### 3. 详细测试结果
每个任务的：
- 任务名称（中英文）
- 执行状态（成功/失败）
- 成功率统计
- 成功集数/总集数
- 平均步数

### 4. 成功率汇总
- 按成功率降序排列的任务列表
- 平均成功率
- 成功率分布统计（高/中/低）

### 5. 失败任务列表
列出所有失败的任务及错误信息

## 使用方法

### 使用默认日志文件名
```bash
python batch_test_universal.py --config test_config.yaml
```

### 指定日志文件路径
```bash
python batch_test_universal.py --config test_config.yaml --log-file my_test_results.log
```

### 指定特定任务测试
```bash
python batch_test_universal.py --config test_config.yaml --tasks task1 task2 --log-file specific_test.log
```

