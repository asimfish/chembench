#!/bin/bash
# 脚本名称: sync_diffusion_policy.sh
# 功能: 拉取远端代码，并安全地提交 diffusion_policy 目录下的更改

BRANCH="main"

echo "=== 开始同步流程 ==="

# 1. 检查是否有本地修改（包括未追踪的文件）
if [[ -n $(git status --porcelain) ]]; then
    echo "发现本地有修改，正在暂存(Stash)以避免拉取冲突..."
    # 包含 untracked 文件 (-u)
    git stash push -u -m "Auto stash before sync $(date)"
    STASHED=1
else
    echo "本地干净，无需暂存。"
    STASHED=0
fi

# 2. 拉取远端代码
echo "--------------------------------"
echo "正在从远端拉取最新代码..."
git pull origin $BRANCH

if [ $? -ne 0 ]; then
    echo "❌ 错误: 拉取代码失败。"
    if [ $STASHED -eq 1 ]; then
        echo "正在尝试恢复暂存的内容..."
        git stash pop
    fi
    exit 1
fi

# 3. 恢复暂存的更改
if [ $STASHED -eq 1 ]; then
    echo "--------------------------------"
    echo "正在恢复暂存的本地修改..."
    git stash pop
    
    if [ $? -ne 0 ]; then
        echo "⚠️ 警告: 恢复暂存时发生冲突。"
        echo "请手动解决冲突（打开有冲突的文件修改），解决后执行："
        echo "  git add diffusion_policy/"
        echo "  git commit -m 'Your message'"
        echo "  git push origin $BRANCH"
        exit 1
    fi
fi

# 4. 提交 diffusion_policy
echo "--------------------------------"
# 检查 diffusion_policy 下是否有变化
if [[ -n $(git status --porcelain diffusion_policy/) ]]; then
    echo "检测到 diffusion_policy 目录下有更改。"
    read -p "是否要提交这些更改到远端? (y/n) " answer

    if [[ "$answer" =~ ^[Yy]$ ]]; then
        echo "正在添加 diffusion_policy 目录..."
        git add diffusion_policy/

        read -p "请输入提交信息 (回车默认: 'Update diffusion_policy'): " commit_msg
        if [ -z "$commit_msg" ]; then
            commit_msg="Update diffusion_policy"
        fi
        
        git commit -m "$commit_msg"
        
        echo "正在推送到远端..."
        git push origin $BRANCH
        
        if [ $? -eq 0 ]; then
            echo "✅ 成功: diffusion_policy 代码已更新并推送到远端。"
        else
            echo "❌ 错误: 推送失败。"
        fi
    else
        echo "已跳过提交。"
    fi
else
    echo "diffusion_policy 目录下没有需要提交的更改。"
fi

echo "=== 流程结束 ==="

