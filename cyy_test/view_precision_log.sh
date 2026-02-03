#!/bin/bash
# 快速查看精度测试日志的辅助脚本

LOG_FILE="cyy_test/outputs/precision_test/precision_test_log.txt"
NPZ_FILE="cyy_test/outputs/precision_test/pytorch_results.npz"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ 日志文件不存在: $LOG_FILE"
    echo "请先运行: python cyy_test/test_precision.py"
    exit 1
fi

echo "📄 精度测试日志查看器"
echo "===================="
echo ""

# 显示文件信息
echo "文件信息:"
ls -lh "$LOG_FILE" "$NPZ_FILE" 2>/dev/null
echo ""

# 显示菜单
echo "选择查看选项:"
echo "  1) 查看完整日志"
echo "  2) 查看 VAE Encoder 部分"
echo "  3) 查看 VAE Decoder 部分"
echo "  4) 查看 LDM Denoiser 部分"
echo "  5) 查看测试摘要"
echo "  6) 查看所有张量统计信息"
echo "  7) 查看前50行"
echo "  8) 查看最后50行"
echo ""

if [ "$1" != "" ]; then
    choice=$1
else
    read -p "请输入选项 (1-8): " choice
fi

case $choice in
    1)
        echo ""
        echo "=== 完整日志 ==="
        cat "$LOG_FILE"
        ;;
    2)
        echo ""
        echo "=== VAE Encoder 测试 ==="
        sed -n '/测试 VAE Encoder/,/测试 VAE Decoder/p' "$LOG_FILE" | head -n -2
        ;;
    3)
        echo ""
        echo "=== VAE Decoder 测试 ==="
        sed -n '/测试 VAE Decoder/,/测试完整 VAE/p' "$LOG_FILE" | head -n -2
        ;;
    4)
        echo ""
        echo "=== LDM Denoiser 测试 ==="
        sed -n '/测试 LDM Denoiser/,/测试 LDM 完整采样/p' "$LOG_FILE" | head -n -2
        ;;
    5)
        echo ""
        echo "=== 测试摘要 ==="
        grep -E "(测试结构:|原始结构:|重建误差:|生成的结构:|测试完成)" "$LOG_FILE"
        ;;
    6)
        echo ""
        echo "=== 所有张量统计信息 ==="
        grep -E "(Shape:|Mean:|Std:|Min:|Max:|First .* values:)" "$LOG_FILE"
        ;;
    7)
        echo ""
        echo "=== 前50行 ==="
        head -50 "$LOG_FILE"
        ;;
    8)
        echo ""
        echo "=== 最后50行 ==="
        tail -50 "$LOG_FILE"
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac
