#!/usr/bin/env bash

# 自动修改vscode extension中的codex，为其添加指定端口的proxy端口代理，支持`~/.vscode, ~/.vscode-server, ~/.vscode-server-container`三个文件夹下的自动修改
# 使用方法：./codex_vscode_extension_add_proxy.sh --porxy 7890

# 默认端口
PROXY_PORT=7898

# 处理参数 --port
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port) PROXY_PORT="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

# 可能存在的扩展路径列表
PATHS=(
    "$HOME/.vscode/extensions/openai.chatgpt-*/bin/linux-*"
    "$HOME/.vscode-server/extensions/openai.chatgpt-*/bin/linux-*"
    "$HOME/.vscode-server-container/extensions/openai.chatgpt-*/bin/linux-*"
)

FOUND=false

for pattern in "${PATHS[@]}"; do
    # 使用 shopt 确保通配符匹配失败时不报错
    for dir in $pattern; do
        if [ -d "$dir" ]; then
            echo "找到目标目录: $dir"
            cd "$dir" || continue

            # 检查是否已经修改过，避免重复操作
            if [ -f "codex" ] && [ ! -f "codex.real" ]; then
                echo "正在备份并替换 codex..."
                mv codex codex.real
            elif [ -f "codex.real" ]; then
                echo "检测到 codex.real 已存在，更新脚本内容..."
            else
                echo "未找到二进制文件 codex，跳过此目录。"
                continue
            fi

            # 写入 wrapper 脚本
            cat > codex <<EOF
#!/usr/bin/env bash
export HTTPS_PROXY="http://127.0.0.1:$PROXY_PORT"
export HTTP_PROXY="http://127.0.0.1:$PROXY_PORT"
# ------------------------------------------------
HERE="\$(cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")" && pwd)"
exec "\$HERE/codex.real" "\$@"
EOF
            chmod +x codex
            echo "成功为 $dir 应用代理端口: $PROXY_PORT"
            FOUND=true
        fi
    done
done

if [ "$FOUND" = false ]; then
    echo "错误：未找到任何 ChatGPT 扩展安装路径。"
    exit 1
fi

echo "所有操作已完成！请重启 VS Code。"
