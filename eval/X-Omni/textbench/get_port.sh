#!/bin/bash

function get_available_port() {
    # 在受限环境中使用更简单的实现
    local port
    local base_port=30000  # 从高位端口开始
    
    # 使用Python获取端口（简化版）
    port=$(python -c '
import socket, sys
for port in range(30000, 60000, 100):  # 以步长100减少冲突
    try:
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", port))
        s.close()
        print(port)
        sys.exit(0)
    except:
        continue
sys.exit(1)' 2>/dev/null)

    # 备用方案：随机生成高位端口
    if [ -z "$port" ] || [[ "$port" == *" "* ]]; then
        port=$(( 30000 + RANDOM % 30000 ))
    fi
    
    echo "$port" | head -n 1  # 确保只输出第一个端口号
    return 0  # 总是返回成功，使用备用端口
}

# 调用函数并确保只输出一个端口号
get_available_port | head -n 1

# 检查结果并设置退出码
if [ $? -ne 0 ]; then
    echo "错误: 未找到可用端口" >&2
    exit 1
fi