#!/bin/bash
# ==============================================================================
# 批量评估脚本 - Diffusion Policy 模型评估
# 
# 使用方法:
#   1. 编辑下方的配置数组，添加要测试的模型和场景
#   2. 运行: bash batch_eval.sh
#   3. 查看日志: cat logs/batch_eval_YYYYMMDD_HHMMSS.log
#
# Author: Auto-generated
# ==============================================================================

# ==================== 基础配置 ====================
WORKSPACE_DIR="/home/psibot/chembench"
SCRIPT_DIR="${WORKSPACE_DIR}/psilab/scripts_psi/workflows/imitation_learning"
LOG_DIR="${WORKSPACE_DIR}/data/eval_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/batch_eval_${TIMESTAMP}.log"

# Isaac Sim 相关
NUM_ENVS=1
SEED=17
MAX_EPISODE=100
MAX_STEP=500

# 创建日志目录
mkdir -p ${LOG_DIR}

# ==================== 测试配置 ====================
# 格式: "checkpoint_path|mode|scene_config|object_name"
# - checkpoint_path: 模型权重路径
# - mode: rgb 或 state
# - scene_config: 场景配置名称 (如 PSI_DC_Beaker_003_CFG)
# - object_name: 物体名称（用于日志记录）

EVAL_CONFIGS=(
    # === 示例配置 - 请根据需要修改 ===
    
    # 100ml玻璃烧杯 - RGB模式
    "/home/psibot/chembench/data/outputs/grasp_rgb/100ml玻璃烧杯/20251218_103000_n50/checkpoints/latest.ckpt|rgb|PSI_DC_Beaker_003_CFG|100ml玻璃烧杯"
    
    # 100ml玻璃烧杯 - State模式
    # "/home/psibot/chembench/data/outputs/grasp_state/100ml玻璃烧杯/20251218_110000_n50/checkpoints/latest.ckpt|state|PSI_DC_Beaker_003_CFG|100ml玻璃烧杯"
    
    # 500ml玻璃烧杯 - RGB模式
    # "/home/psibot/chembench/data/outputs/grasp_rgb/500ml玻璃烧杯/xxx/checkpoints/latest.ckpt|rgb|PSI_DC_Beaker_005_CFG|500ml玻璃烧杯"
    
    # 坩埚 - RGB模式
    # "/home/psibot/chembench/data/outputs/grasp_rgb/坩埚/xxx/checkpoints/latest.ckpt|rgb|PSI_DC_Crucible_CFG|坩埚"
)

# ==================== 函数定义 ====================

# 日志函数
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "${LOG_FILE}"
}

# 分隔线
separator() {
    local line="================================================================"
    echo "$line"
    echo "$line" >> "${LOG_FILE}"
}

# 运行单个评估
run_eval() {
    local checkpoint=$1
    local mode=$2
    local scene=$3
    local object_name=$4
    
    # 根据模式选择task
    local task_name
    if [ "$mode" == "rgb" ]; then
        task_name="Psi-IL-Grasp-Beaker-003-v1"  # RGB模式的task
    else
        task_name="Psi-IL-Grasp-Beaker-003-State-v1"  # State模式的task (如果有的话)
    fi
    
    log "开始评估:"
    log "  物体: ${object_name}"
    log "  模式: ${mode}"
    log "  场景: ${scene}"
    log "  权重: ${checkpoint}"
    log "  Task: ${task_name}"
    
    # 检查checkpoint是否存在
    if [ ! -f "${checkpoint}" ]; then
        log "  ❌ 错误: checkpoint文件不存在: ${checkpoint}"
        return 1
    fi
    
    # 构建命令
    local cmd="cd ${WORKSPACE_DIR} && python ${SCRIPT_DIR}/play.py \
        --task ${task_name} \
        --num_envs ${NUM_ENVS} \
        --seed ${SEED} \
        --scene room_cfg:${scene} \
        --enable_cameras \
        --async_reset \
        --checkpoint ${checkpoint} \
        --max_step ${MAX_STEP} \
        --max_episode ${MAX_EPISODE} \
        --enable_eval \
        --headless"
    
    log "  执行命令..."
    
    # 运行评估并捕获输出
    local start_time=$(date +%s)
    local output
    output=$(eval ${cmd} 2>&1)
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # 解析成功率
    local success_rate=$(echo "$output" | grep -oP '成功率: \K[0-9]+/[0-9]+' | tail -1)
    local policy_rate=$(echo "$output" | grep -oP 'Policy成功率: \K[0-9.]+%' | tail -1)
    
    if [ $exit_code -eq 0 ]; then
        log "  ✅ 评估完成"
        log "  成功率: ${success_rate:-N/A}"
        log "  Policy成功率: ${policy_rate:-N/A}"
        log "  耗时: ${duration}秒"
    else
        log "  ❌ 评估失败 (exit code: ${exit_code})"
        log "  错误输出:"
        echo "$output" | tail -20 >> "${LOG_FILE}"
    fi
    
    # 保存完整输出到单独的日志文件
    local detail_log="${LOG_DIR}/${object_name}_${mode}_${TIMESTAMP}.log"
    echo "$output" > "${detail_log}"
    log "  详细日志: ${detail_log}"
    
    return $exit_code
}

# ==================== 主程序 ====================

separator
log "批量评估开始"
log "日志文件: ${LOG_FILE}"
log "评估配置数量: ${#EVAL_CONFIGS[@]}"
separator

# 统计变量
total=0
success=0
failed=0

# 汇总结果数组
declare -a results

# 遍历所有配置
for config in "${EVAL_CONFIGS[@]}"; do
    # 跳过注释行
    [[ "$config" =~ ^# ]] && continue
    [[ -z "$config" ]] && continue
    
    # 解析配置
    IFS='|' read -r checkpoint mode scene object_name <<< "$config"
    
    separator
    ((total++))
    log "评估 ${total}/${#EVAL_CONFIGS[@]}: ${object_name} (${mode})"
    
    if run_eval "$checkpoint" "$mode" "$scene" "$object_name"; then
        ((success++))
        results+=("✅ ${object_name} (${mode}): 成功")
    else
        ((failed++))
        results+=("❌ ${object_name} (${mode}): 失败")
    fi
done

# ==================== 汇总报告 ====================
separator
log ""
log "==================== 评估汇总 ===================="
log "总计: ${total} 个模型"
log "成功: ${success} 个"
log "失败: ${failed} 个"
log ""
log "详细结果:"
for result in "${results[@]}"; do
    log "  ${result}"
done
log ""
log "日志目录: ${LOG_DIR}"
log "=================================================="
separator

log "批量评估完成"

