# 建筑物大图 NPU 推理 部署包

参照 mmseg/1 里水体/海岸线那套范式做的,差异:
- 单时相单模型直接出图,没有变化检测
- 只接受一张 GeoTIFF 输入(`DATA_INPUT_DIR1`)
- 滑窗 512×512 stride 384,概率叠加 + 羽化拼缝
- 输出: 二值 mask GeoTIFF + 矢量 SHP(自动 polygonize),与原图同 CRS

## 目录结构

```
deploy_building/
├── run_building.sh           # 容器入口 (env 驱动)
├── infer_large_image.py      # 核心: 滑窗 + OM/ONNX 推理 + 写 GeoTIFF
│                              #       (VECTOR_WKT 在这里用 shapely 直接裁剪)
├── postprocess.py            # mask -> SHP 矢量化
├── send_kafka_msg.py         # Kafka 进度上报封装
├── MessageClient/
│   ├── __init__.py
│   └── ProgressMessageSender.py
├── Dockerfile
└── README.md
```

## 构建

```bash
cd C:\Users\zanly\Desktop\mmseg\deploy_building

# 拷贝 ATC 转出的 building_seg.om 到镜像里(或 docker run 时挂载,见下)
# 这里两种方案二选一:

# 方案 A: 模型 bake 进镜像
cp /path/to/building_seg.om ./building_seg.om
# 然后在 Dockerfile 末尾加:
#   COPY building_seg.om /app/module/building_seg.om
docker build -t building-infer:v1 .

# 方案 B: 模型外挂(更灵活,推荐)
docker build -t building-infer:v1 .
# 跑时: -v /your/host/module:/app/module:ro
```

## 跑(NPU 实机)

```bash
docker run --rm \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
    -v /share:/share \
    -v /your/host/path/to/module:/app/module:ro \
    -e DATA_INPUT_DIR1="/share/高分影像库/GF1_PMS2_E121.6_N37.4_20250606_xxx.tif" \
    -e DATA_OUTPUT_DIR="/share/高分影像解译输出库/6733" \
    -e VECTOR_WKT="MULTIPOLYGON(((121.86 37.49, ...)))" \
    -e PATH_WORKING="/share/高分影像解译输出库/6733/working" \
    -e KAFKA_SERVER_IP_PORT=192.168.1.166:12007 \
    -e KAFKA_TOPIC=ib-theme.algorithm_message_topic \
    -e KAFKA_TASK_ID=KAFKA_TASKID_6733 \
    -e YC_TASK_ID=6733 \
    building-infer:v1
```

## 环境变量速查

| 名字 | 必填 | 默认 | 含义 |
|---|---|---|---|
| `DATA_INPUT_DIR1` | ✓ | - | 输入 GeoTIFF |
| `DATA_OUTPUT_DIR` | ✓ | - | 输出目录 |
| `PATH_MODEL_RESOURCE` |   | `/app/module` | 找 `building_seg.om` 的目录 |
| `PATH_WORKING` |   | `${DATA_OUTPUT_DIR}/working` | 临时文件 |
| `VECTOR_WKT` |   | - | WGS84 多边形,推理结果按它裁剪 |
| `BAND_ORDER` |   | `3,2,1` | 多光谱波段映射为 R/G/B |
| `THRESHOLD` |   | `0.5` | 概率二值化阈值 |
| `TILE` / `STRIDE` |   | `512/384` | 滑窗大小/步长 |
| `SAVE_PROB` |   | `0` | 是否同时输出概率图 |
| `MIN_AREA_PX` |   | `25` | 矢量化时过滤小图斑(像素数) |
| `KAFKA_SERVER_IP_PORT` |   | - | 不填则只 log 不上报 |
| `KAFKA_TOPIC` |   | - | 同上 |
| `KAFKA_TASK_ID` |   | uuid | 上报消息里的 taskId |
| `YC_TASK_ID` |   | - | 仅用于日志显示 |

## 输出文件

stem = 输入 TIF 文件名去扩展名:

```
${DATA_OUTPUT_DIR}/
├── <stem>_building_mask.tif      二值 mask, uint8, 与原图同 CRS+分辨率
├── <stem>_building_prob.tif      仅当 SAVE_PROB=1, 概率图 uint8 [0,255]
├── <stem>_building.shp           矢量结果(多边形)
├── <stem>_building.dbf
├── <stem>_building.shx
└── <stem>_building.prj
```

## Kafka 进度阶段

```
  0    启动
  5    初始化完成
 10    开始大图推理
 10-90 滑窗推理中(按瓦片数线性映射)
 92    矢量化
100    completed
```

中途任何步骤失败会发 `failed`,带具体原因。

## 本地 PC 调试(没有 NPU 环境)

模型用 `.onnx` 不用 `.om`,脚本会自动回落到 onnxruntime:

```bash
export DATA_INPUT_DIR1=/some/test.tif
export DATA_OUTPUT_DIR=/tmp/out
export PATH_MODEL_RESOURCE=/path/with/building_seg.onnx
# 不设 KAFKA_*, 进度只打到 stdout

bash run_building.sh
```

## 已知约束 / 后续可优化

- **内存**: 当前实现一次读全图到内存。对 10k×10k 的影像 OK,50k×50k 会吃紧。
  后续可改成 rasterio `Window` 流式读 + 流式写。
- **后处理**: 当前 polygonize 简单过滤面积,没做形态学(开闭运算),建筑物轮廓
  可能有毛边。如果系统侧要"规整化",在 `postprocess.py` 加 `cv2.morphologyEx`。
