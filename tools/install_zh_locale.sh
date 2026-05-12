#!/usr/bin/env bash
# 在 Ubuntu/Debian 基础的镜像里装中文 locale + CJK 字体
# 主要解决两件事:
#   1. Python / mmcv / opencv 读 "建筑物" 这类中文路径不再报 UnicodeEncodeError
#   2. (可选) cv2.putText 画中文时也有字体可用
#
# 用法(在容器里跑,然后 docker commit 持久化到镜像):
#   docker exec -it <容器ID> bash
#   bash install_zh_locale.sh
#
# 装完后退出,在宿主机:
#   docker commit <容器ID> mindie-gdal-mmdet-mmseg-zh:v1
#   # 以后启动用新 tag

set -e
export DEBIAN_FRONTEND=noninteractive

echo "==> [1/4] apt update + 安装 locales 和 CJK 字体"
apt-get update
apt-get install -y --no-install-recommends \
    locales \
    fonts-noto-cjk \
    fonts-wqy-zenhei \
    fonts-wqy-microhei \
    fontconfig
fc-cache -fv > /dev/null 2>&1 || true
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "==> [2/4] 生成 zh_CN.UTF-8 / en_US.UTF-8 / C.UTF-8 locale"
# 取消注释对应行
sed -i 's/^# *zh_CN.UTF-8 UTF-8/zh_CN.UTF-8 UTF-8/' /etc/locale.gen || true
sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen || true
# 兼容某些精简镜像没 locale.gen 的情况
grep -q "zh_CN.UTF-8 UTF-8" /etc/locale.gen 2>/dev/null || \
    echo "zh_CN.UTF-8 UTF-8" >> /etc/locale.gen
grep -q "en_US.UTF-8 UTF-8" /etc/locale.gen 2>/dev/null || \
    echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen

locale-gen
# 默认走 C.UTF-8 最稳(UTF-8 支持但不会触发 zh 翻译相关副作用)
update-locale LANG=C.UTF-8 LC_ALL=C.UTF-8

echo "==> [3/4] 写默认环境变量,下次 bash / docker exec 进来自动生效"
cat > /etc/profile.d/01-locale.sh <<'EOF'
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8
EOF
chmod 644 /etc/profile.d/01-locale.sh

# /etc/bash.bashrc 也加一份(交互 bash 进来会读)
if ! grep -q "PYTHONIOENCODING=utf-8" /etc/bash.bashrc 2>/dev/null; then
    cat >> /etc/bash.bashrc <<'EOF'

# Chinese / UTF-8 support
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8
EOF
fi

# /etc/environment 给非交互 shell(systemd / supervisord 启动的进程)
{
    echo "LANG=C.UTF-8"
    echo "LC_ALL=C.UTF-8"
    echo "PYTHONIOENCODING=utf-8"
} > /etc/environment

echo "==> [4/4] 验证"
echo "--- locale -a 包含的 UTF-8 locale ---"
locale -a | grep -iE "(zh_CN|en_US|C\.utf8|C\.UTF-8)" || echo "(没找到,可能 locale-gen 失败)"

echo
echo "--- Python 文件系统编码 ---"
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
python3 -c "
import sys, locale, os
print('  sys.getfilesystemencoding =', sys.getfilesystemencoding())
print('  sys.getdefaultencoding    =', sys.getdefaultencoding())
print('  locale.getpreferredencoding =', locale.getpreferredencoding())
# 实际写一个中文路径试试
tmp = '/tmp/中文测试_zh_locale'
os.makedirs(tmp, exist_ok=True)
print('  写入/列出中文目录:', os.listdir('/tmp/'))
os.rmdir(tmp)
print('  OK: 中文路径读写正常')
"

echo
echo "--- CJK 字体 ---"
fc-list :lang=zh 2>/dev/null | head -5 || echo "(fc-list 失败)"

echo
echo "[done] 容器内中文环境就绪。下一步在 *宿主机* 执行:"
echo "  docker commit <你的容器ID> mindie-gdal-mmdet-mmseg-zh:v1"
echo "  以后 docker run 用 mindie-gdal-mmdet-mmseg-zh:v1 即可"
