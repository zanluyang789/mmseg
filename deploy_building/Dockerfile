# 建筑物大图推理镜像
# 基于 ascendhub/mindie-gdal-mmdet-mmseg (openEuler / linux/arm64),只加业务代码
# - 不动包管理器:base 镜像已经有 GDAL/mmseg/rasterio/shapely/onnxruntime 等
# - 不动 locale: openEuler 默认就支持 UTF-8
# - 业务依赖里 fiona/kafka-python 用 pip 兜底装一次,base 里有就 skip
#
# 构建(先把本地 base 重 tag 成单段式,避免 BuildKit 去 docker.io 找):
#   docker tag ascendhub/mindie-gdal-mmdet-mmseg:v1.3 mindie-base:v1.3
#   docker build -t building-infer:v1 .
#
# 或一行带 build-arg:
#   docker build --build-arg BASE_IMAGE=mindie-base:v1.3 -t building-infer:v1 .
#
# 跑(NPU 实机, 模型挂载方式):
#   docker run --rm \
#       --device=/dev/davinci0 --device=/dev/davinci_manager \
#       --device=/dev/devmm_svm --device=/dev/hisi_hdc \
#       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
#       -v /share:/share \
#       -v /your/host/module:/app/module:ro \
#       -e DATA_INPUT_DIR1=/share/.../xxx.tif \
#       -e DATA_OUTPUT_DIR=/share/.../6733 \
#       -e VECTOR_WKT="MULTIPOLYGON(((...)))" \
#       -e KAFKA_SERVER_IP_PORT=192.168.1.166:12007 \
#       -e KAFKA_TOPIC=ib-theme.algorithm_message_topic \
#       -e KAFKA_TASK_ID=KAFKA_TASKID_6733 \
#       -e YC_TASK_ID=6733 \
#       building-infer:v1

ARG BASE_IMAGE=mindie-base:v1.3
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8

# 兜底装一下业务依赖(基础镜像里有就 pip 直接 skip,网络不通也允许失败)
RUN pip install --no-cache-dir \
        fiona \
        kafka-python==2.0.2 \
    || pip3 install --no-cache-dir \
        fiona \
        kafka-python==2.0.2 \
    || echo "[warn] pip 装 fiona/kafka-python 失败,如果 base 镜像里没有,运行时会报错"

# ---- 业务代码 ----
WORKDIR /workspace
COPY MessageClient/        ./MessageClient/
COPY send_kafka_msg.py     ./
COPY postprocess.py        ./
COPY infer_large_image.py  ./
COPY run_building.sh       ./
RUN chmod +x /workspace/run_building.sh && \
    sed -i 's/\r$//' /workspace/run_building.sh

ENTRYPOINT ["/workspace/run_building.sh"]
