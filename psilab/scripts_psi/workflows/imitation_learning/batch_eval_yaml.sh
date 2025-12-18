#!/bin/bash
# ==============================================================================
# 批量评估脚本 (YAML配置版)
# 
# 功能:
#   - 支持RGB和State两种模式
#   - 自动修改物体USD路径
#   - 有头模式运行
#   - 输出详细日志和汇总CSV
#
# 使用方法:
#   1. 编辑 eval_configs.yaml 配置文件
#   2. 运行: bash batch_eval_yaml.sh
#   3. 查看日志: cat data/eval_logs/summary_*.csv
# ==============================================================================

# 注意：不使用 set -e，因为 ((total++)) 等操作在某些情况下会返回非零值

# ==================== 基础配置 ====================
WORKSPACE_DIR="/home/psibot/chembench"
SCRIPT_DIR="${WORKSPACE_DIR}/psilab/scripts_psi/workflows/imitation_learning"
CONFIG_FILE="${SCRIPT_DIR}/eval_configs.yaml"
ROOM_CFG_FILE="${WORKSPACE_DIR}/psilab/source/psilab_tasks/psilab_tasks/imitation_learning/grasp_bottle_v1/scenes/room_cfg.py"
LOG_DIR="${WORKSPACE_DIR}/data/eval_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/batch_eval_${TIMESTAMP}.log"
SUMMARY_FILE="${LOG_DIR}/summary_${TIMESTAMP}.csv"

# 备份文件
ROOM_CFG_BACKUP="${ROOM_CFG_FILE}.backup_${TIMESTAMP}"

# 创建日志目录
mkdir -p ${LOG_DIR}

# ==================== 函数定义 ====================

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "${LOG_FILE}"
}

separator() {
    local line="================================================================"
    echo "$line"
    echo "$line" >> "${LOG_FILE}"
}

# 备份room_cfg.py
backup_room_cfg() {
    if [ ! -f "${ROOM_CFG_BACKUP}" ]; then
        cp "${ROOM_CFG_FILE}" "${ROOM_CFG_BACKUP}"
        log "已备份 room_cfg.py -> ${ROOM_CFG_BACKUP}"
    fi
}

# 恢复room_cfg.py
restore_room_cfg() {
    if [ -f "${ROOM_CFG_BACKUP}" ]; then
        cp "${ROOM_CFG_BACKUP}" "${ROOM_CFG_FILE}"
        log "已恢复 room_cfg.py"
    fi
}

# 修改room_cfg.py中的usd_path
update_usd_path() {
    local new_usd_path=$1
    log "  更新USD路径: ${new_usd_path}"
    
    # 使用Python进行更可靠的替换
    python3 << PYTHON_SCRIPT
import re

room_cfg_file = "${ROOM_CFG_FILE}"
new_usd_path = "${new_usd_path}"

with open(room_cfg_file, 'r') as f:
    content = f.read()

# 查找 PSI_DC_Grasp_CFG 配置块中的 usd_path 并替换
# 匹配模式: usd_path="..." 或 usd_path='...'
pattern = r'(PSI_DC_Grasp_CFG\s*=.*?rigid_objects_cfg\s*=\s*\{.*?"bottle"\s*:\s*RigidObjectCfg\(.*?spawn\s*=\s*sim_utils\.UsdFileCfg\(\s*usd_path\s*=\s*)["\']([^"\']+)["\']'

# 使用正则表达式替换（DOTALL模式允许.匹配换行符）
def replace_usd_path(match):
    return match.group(1) + '"' + new_usd_path + '"'

new_content = re.sub(pattern, replace_usd_path, content, flags=re.DOTALL)

# 如果上面的方法没有成功，尝试更简单的替换
if new_content == content:
    # 查找所有 usd_path 行并替换第一个在 PSI_DC_Grasp_CFG 附近的
    lines = content.split('\n')
    in_grasp_cfg = False
    modified = False
    new_lines = []
    
    for i, line in enumerate(lines):
        if 'PSI_DC_Grasp_CFG' in line:
            in_grasp_cfg = True
        
        if in_grasp_cfg and 'usd_path=' in line and not modified:
            # 替换这行的 usd_path
            new_line = re.sub(r'usd_path\s*=\s*["\'][^"\']+["\']', f'usd_path="{new_usd_path}"', line)
            new_lines.append(new_line)
            modified = True
        else:
            new_lines.append(line)
    
    if modified:
        new_content = '\n'.join(new_lines)

with open(room_cfg_file, 'w') as f:
    f.write(new_content)

print("USD路径更新完成")
PYTHON_SCRIPT
}

# 解析YAML配置
parse_config() {
    python3 << 'PYTHON_SCRIPT'
import yaml
import sys
import os

config_file = os.environ.get('CONFIG_FILE', 'eval_configs.yaml')

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# 全局配置
global_cfg = config.get('global', {})
scene_config = config.get('scene_config', 'PSI_DC_Grasp_CFG')
usd_paths = config.get('usd_paths', {})
tasks = config.get('tasks', {})

# 输出启用的评估任务
for eval_cfg in config.get('evaluations', []):
    if not eval_cfg.get('enabled', True):
        continue
    
    name = eval_cfg.get('name', 'unnamed')
    checkpoint = eval_cfg.get('checkpoint', '')
    mode = eval_cfg.get('mode', 'rgb')
    object_name = eval_cfg.get('object_name', '')
    
    # 获取USD路径（优先使用任务中指定的，否则从映射中获取）
    usd_path = eval_cfg.get('usd_path', usd_paths.get(object_name, ''))
    
    # 获取task名称
    task = eval_cfg.get('task', tasks.get(mode, 'Psi-IL-Grasp-Beaker-003-v1'))
    
    # 获取其他配置（使用全局配置作为默认值）
    num_envs = eval_cfg.get('num_envs', global_cfg.get('num_envs', 10))
    max_episode = eval_cfg.get('max_episode', global_cfg.get('max_episode', 100))
    max_step = eval_cfg.get('max_step', global_cfg.get('max_step', 500))
    seed = eval_cfg.get('seed', global_cfg.get('seed', 17))
    headless = eval_cfg.get('headless', global_cfg.get('headless', False))
    
    # 输出配置行
    print(f"{name}|{checkpoint}|{mode}|{scene_config}|{object_name}|{task}|{num_envs}|{max_episode}|{max_step}|{seed}|{headless}|{usd_path}")

PYTHON_SCRIPT
}

# 运行单个评估
run_eval() {
    local name=$1
    local checkpoint=$2
    local mode=$3
    local scene=$4
    local object_name=$5
    local task=$6
    local num_envs=$7
    local max_episode=$8
    local max_step=$9
    local seed=${10}
    local headless=${11}
    local usd_path=${12}
    
    log "开始评估: ${name}"
    log "  物体: ${object_name}"
    log "  模式: ${mode}"
    log "  场景: ${scene}"
    log "  Task: ${task}"
    log "  权重: ${checkpoint}"
    log "  USD: ${usd_path}"
    log "  参数: envs=${num_envs}, episodes=${max_episode}, steps=${max_step}"
    
    # 检查checkpoint是否存在
    if [ ! -f "${checkpoint}" ]; then
        log "  ❌ 错误: checkpoint文件不存在"
        echo "${name},${object_name},${mode},ERROR,0,0,0%,0s,文件不存在" >> "${SUMMARY_FILE}"
        return 1
    fi
    
    # 检查usd_path是否存在
    if [ -n "${usd_path}" ] && [ ! -f "${usd_path}" ]; then
        log "  ⚠️ 警告: USD文件不存在: ${usd_path}"
    fi
    
    # 更新USD路径
    if [ -n "${usd_path}" ]; then
        update_usd_path "${usd_path}"
    fi
    
    # 构建命令（有头模式不使用--headless）
    local headless_flag=""
    if [ "$headless" == "True" ] || [ "$headless" == "true" ]; then
        headless_flag="--headless"
    fi
    
    # 根据模式决定是否启用相机
    local camera_flag=""
    if [ "$mode" == "rgb" ]; then
        camera_flag="--enable_cameras"
    fi
    
    local cmd="cd ${WORKSPACE_DIR} && python ${SCRIPT_DIR}/play.py \
        --task ${task} \
        --num_envs ${num_envs} \
        --seed ${seed} \
        --scene room_cfg:${scene} \
        ${camera_flag} \
        --async_reset \
        --checkpoint ${checkpoint} \
        --max_step ${max_step} \
        --max_episode ${max_episode} \
        --enable_eval \
        ${headless_flag}"
    
    log "  执行中..."
    
    # 运行评估
    local start_time=$(date +%s)
    local output
    local detail_log="${LOG_DIR}/${name}_${TIMESTAMP}.log"
    
    # 执行命令并保存输出
    set +e
    output=$(eval ${cmd} 2>&1)
    local exit_code=$?
    set -e
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # 保存详细日志
    echo "$output" > "${detail_log}"
    
    # 解析结果
    local success_num=$(echo "$output" | grep -oP '成功次数/总次数: \K[0-9]+(?=/)' | tail -1)
    local total_num=$(echo "$output" | grep -oP '成功次数/总次数: [0-9]+/\K[0-9]+' | tail -1)
    local success_rate="0.00"
    
    if [ -z "$success_num" ]; then
        success_num=$(echo "$output" | grep -oP '成功率: \K[0-9]+(?=/)' | tail -1)
    fi
    if [ -z "$total_num" ]; then
        total_num=$(echo "$output" | grep -oP '成功率: [0-9]+/\K[0-9]+' | tail -1)
    fi
    
    if [ -n "$success_num" ] && [ -n "$total_num" ] && [ "$total_num" -gt 0 ]; then
        success_rate=$(echo "scale=2; $success_num * 100 / $total_num" | bc)
    fi
    
    if [ $exit_code -eq 0 ]; then
        log "  ✅ 评估完成"
        log "  成功率: ${success_num:-0}/${total_num:-0} (${success_rate}%)"
        log "  耗时: ${duration}秒 ($(echo "scale=1; $duration/60" | bc)分钟)"
        echo "${name},${object_name},${mode},SUCCESS,${success_num:-0},${total_num:-0},${success_rate}%,${duration}s" >> "${SUMMARY_FILE}"
    else
        log "  ❌ 评估失败 (exit code: ${exit_code})"
        log "  查看详细日志: ${detail_log}"
        echo "${name},${object_name},${mode},FAILED,0,0,0%,${duration}s" >> "${SUMMARY_FILE}"
    fi
    
    log "  详细日志: ${detail_log}"
    
    return $exit_code
}

# 清理函数
cleanup() {
    log ""
    log "正在清理..."
    restore_room_cfg
    log "清理完成"
}

# 设置退出时清理
trap cleanup EXIT

# ==================== 主程序 ====================

separator
log "批量评估开始"
log "配置文件: ${CONFIG_FILE}"
log "日志文件: ${LOG_FILE}"
log "Room配置: ${ROOM_CFG_FILE}"
separator

# 检查配置文件
if [ ! -f "${CONFIG_FILE}" ]; then
    log "❌ 错误: 配置文件不存在: ${CONFIG_FILE}"
    exit 1
fi

# 备份room_cfg.py
backup_room_cfg

# 初始化汇总文件
echo "名称,物体,模式,状态,成功数,总数,成功率,耗时" > "${SUMMARY_FILE}"

# 统计变量
total=0
success=0
failed=0

# 解析并执行评估任务
export CONFIG_FILE
while IFS='|' read -r name checkpoint mode scene object_name task num_envs max_episode max_step seed headless usd_path; do
    [ -z "$name" ] && continue
    
    separator
    total=$((total + 1))
    log "评估任务 ${total}: ${name}"
    
    if run_eval "$name" "$checkpoint" "$mode" "$scene" "$object_name" "$task" "$num_envs" "$max_episode" "$max_step" "$seed" "$headless" "$usd_path"; then
        success=$((success + 1))
    else
        failed=$((failed + 1))
    fi
    
    # 恢复配置（每次测试后恢复，确保下一次测试使用正确的配置）
    restore_room_cfg
    
done < <(parse_config)

# ==================== 汇总报告 ====================
separator
log ""
log "==================== 评估汇总 ===================="
log "总计: ${total} 个模型"
log "成功: ${success} 个"
log "失败: ${failed} 个"
log ""
log "汇总文件: ${SUMMARY_FILE}"
log "备份文件: ${ROOM_CFG_BACKUP}"
log ""

# 显示汇总表格
if [ -f "${SUMMARY_FILE}" ]; then
    log "结果表格:"
    column -t -s',' "${SUMMARY_FILE}" 2>/dev/null | while read line; do
        log "  $line"
    done
fi

log ""
log "=================================================="
separator
log "批量评估完成"

# 输出最终汇总到控制台
echo ""
echo "==================== 最终汇总 ===================="
cat "${SUMMARY_FILE}" | column -t -s',' 2>/dev/null || cat "${SUMMARY_FILE}"
echo "=================================================="
echo ""
echo "日志目录: ${LOG_DIR}"
echo "汇总文件: ${SUMMARY_FILE}"
