#!/usr/bin/env bash
# 建筑物大图 NPU 推理 容器入口(单时相)
# 完全靠 docker run -e 传环境变量驱动
#
# 必填 env:
#   DATA_INPUT_DIR1, DATA_OUTPUT_DIR
# 可选 env:
#   VECTOR_WKT          WGS84 多边形,推理结果会按这个 ROI 裁剪
#   PATH_MODEL_RESOURCE 模型目录,默认 /app/module,里面应有 building_seg.om
#   PATH_WORKING        临时工作目录,默认 ${DATA_OUTPUT_DIR}/working
#   BAND_ORDER          通道映射,默认 "3,2,1" (GF 多光谱选 R/G/B)
#   THRESHOLD           二值化阈值,默认 0.5
#   TILE/STRIDE         滑窗,默认 512/384
#   SAVE_PROB           "1" 同时输出概率图
#   MIN_AREA_PX         矢量化最小图斑像元数,默认 25
#   KAFKA_*             有就上报,没有就只打日志
#   YC_TASK_ID          任务编号,用于日志显示

set -e
cd "$(dirname "$0")"

LANG=${LANG:-C.UTF-8}
LC_ALL=${LC_ALL:-C.UTF-8}
export LANG LC_ALL PYTHONIOENCODING=utf-8

notify() { python send_kafka_msg.py "$1" "$2" "$3" || true; }

echo "======================================================"
echo "🏠 建筑物分割大图推理 启动  task=${YC_TASK_ID:-N/A}"
echo "======================================================"

echo "[1/4] 打印输入参数"
for v in DATA_INPUT_DIR1 VECTOR_WKT DATA_OUTPUT_DIR \
         PATH_MODEL_RESOURCE PATH_WORKING BAND_ORDER THRESHOLD \
         TILE STRIDE SAVE_PROB MIN_AREA_PX YC_TASK_ID \
         KAFKA_SERVER_IP_PORT KAFKA_TOPIC KAFKA_TASK_ID; do
    val="$(printenv "$v" || true)"
    if [ "$v" = "VECTOR_WKT" ] && [ ${#val} -gt 100 ]; then
        val="${val:0:80}...(${#val} chars)"
    fi
    echo "  $v = $val"
done

echo "[2/4] 检查必要环境变量"
[ -n "${DATA_INPUT_DIR1}" ] || { notify 0 failed "DATA_INPUT_DIR1 未设置"; exit 1; }
[ -n "${DATA_OUTPUT_DIR}" ] || { notify 0 failed "DATA_OUTPUT_DIR 未设置"; exit 1; }
[ -f "${DATA_INPUT_DIR1}" ] || { notify 0 failed "DATA_INPUT_DIR1 不存在: ${DATA_INPUT_DIR1}"; exit 1; }

PATH_MODEL_RESOURCE=${PATH_MODEL_RESOURCE:-/app/module}
PATH_WORKING=${PATH_WORKING:-${DATA_OUTPUT_DIR}/working}
MIN_AREA_PX=${MIN_AREA_PX:-25}

mkdir -p "${DATA_OUTPUT_DIR}" "${PATH_WORKING}"

# 找模型
MODEL=""
for cand in "${PATH_MODEL_RESOURCE}/building_seg.om" "${PATH_MODEL_RESOURCE}/building_seg.onnx"; do
    if [ -f "$cand" ]; then MODEL="$cand"; break; fi
done
if [ -z "$MODEL" ]; then
    notify 0 failed "找不到模型(尝试: ${PATH_MODEL_RESOURCE}/building_seg.{om,onnx})"
    exit 1
fi
echo "  使用模型: $MODEL"

notify 5 running "启动建筑物推理任务"

# [3/4] 推理 (WKT 裁剪在 Python 端用 shapely 直接做)
notify 10 running "开始大图滑窗推理"
python infer_large_image.py \
    --input "${DATA_INPUT_DIR1}" \
    --output-dir "${DATA_OUTPUT_DIR}" \
    --om "$MODEL" \
    ${VECTOR_WKT:+--wkt "${VECTOR_WKT}"} \
    --progress-lo 10 --progress-hi 90

STEM=$(basename "${DATA_INPUT_DIR1}")
STEM="${STEM%.*}"
MASK="${DATA_OUTPUT_DIR}/${STEM}_building_mask.tif"

# [4/4] 矢量化
notify 92 running "栅格 -> 矢量(SHP)"
python postprocess.py \
    --mask "$MASK" \
    --out  "${DATA_OUTPUT_DIR}/${STEM}_building.shp" \
    --min-area "${MIN_AREA_PX}"

echo "======================================================"
echo "✅ 任务完成"
ls -la "${DATA_OUTPUT_DIR}/" || true
echo "======================================================"
notify 100 completed "建筑物推理任务完成"
