# Teaware MuJoCo Collector

一个可独立运行的 MuJoCo 茶具场景数据采集仓库。它将场景定义、随机化、传感器采集、数据契约和网页检查放在同一套小型工具中，不依赖 `waic-demo4` 的 Python 包或运行目录。

当前支持单臂两指夹爪、单臂 xHand 和双臂 xHand 三种场景，以及参数化茶壶、茶杯、公道杯和茶叶罐。xHand 使用原项目中的左右手 URDF、视觉/碰撞 mesh、惯量、关节轴和限位，并保持与原控制栈一致的 12 关节顺序；仓库不包含 SDK 或真机配置。

## 快速开始

要求 Python 3.11+。推荐使用 [uv](https://docs.astral.sh/uv/)：

```bash
git clone <this-repository>
cd teaware-mujoco-collector
uv sync --extra dev

# 采集 3 个 episode，seed 分别为 100、101、102
uv run teaware-mj collect \
  --config configs/tea_table.yaml \
  --output data/teaware \
  --episodes 3 \
  --seed 100

# 检查所有 episode 的文件、shape 和有限值
uv run teaware-mj validate --dataset data/teaware

# 启动网页采集台
uv run teaware-mj serve \
  --config configs/tea_table.yaml \
  --dataset data/teaware \
  --host 127.0.0.1 \
  --port 8080
```

浏览器打开 `http://127.0.0.1:8080`。网页可随机化场景、触发批量采集、选择相机和 RGB/Depth/Segmentation、逐帧浏览历史 episode，并显示数据校验状态。

三套开箱即用的场景配置：

| 配置 | 机器人 |
|---|---|
| `configs/single_gripper.yaml` | 单 xArm7 + 两指夹爪 |
| `configs/single_xhand.yaml` | 单 xArm7 + 右 xHand（12 DoF） |
| `configs/dual_xhand.yaml` | 双 xArm7 + 左/右 xHand（14 + 24 DoF） |

将上面的 `--config` 切换为对应文件即可采集或启动网页。三个配置共享同一数据契约。

不安装 uv 时也可以使用普通虚拟环境：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
teaware-mj --help
```

Linux 无显示器环境可先设置合适的 MuJoCo 后端，例如 `MUJOCO_GL=egl`；纯 CPU 环境通常使用 `MUJOCO_GL=osmesa`。具体可用后端取决于系统驱动。

## 仓库结构

```text
configs/                               单夹爪、单 xHand、双 xHand 场景配置
src/teaware_mujoco/
  robots.py                            xHand q12 顺序、限位和命名契约
  scene.py                             多臂/手、茶具和相机的 YAML -> MJCF
  collector.py                         仿真推进、多模态渲染、原子写盘
  schema.py                            episode 发现与完整性校验
  release_audit.py                     许可证和敏感信息发布门禁
  web.py                               FastAPI 和图像/episode API
  static/                              无构建步骤的网页前端
  assets/ufactory_xarm7/               vendored xArm7 MJCF、mesh、上游许可证
  assets/xhand/                        左右 xHand URDF、STL 视觉与 OBJ 凸包资产
tests/                                 配置、MJCF、采集和网页回归测试
data/                                  默认输出，Git 忽略
```

模块边界是有意保持的：`scene.py` 不写数据，`collector.py` 不包含网页逻辑，`web.py` 只通过 collector/schema 访问仿真和数据集。训练或策略代码应作为新的消费者读取公开数据契约，不要直接依赖网页实现。

## 场景配置

先复制默认配置再修改：

```bash
uv run teaware-mj init-config my_scene.yaml
uv run teaware-mj --config my_scene.yaml scene --output scene.generated.xml
```

YAML 中的长度均为米、角度为度。主要字段：

- `simulation`: MuJoCo timestep、落稳时间、episode 时长和采样帧率；
- `renderer`: 所有相机的离屏渲染宽高；
- `table`: 茶桌中心和完整尺寸；
- `scene_profile`: 数据中记录的场景标识；
- `robots`: 一台或多台机械臂的 id、手型、左右手、基座位姿、arm/hand home joint 和运动幅度；
- `cameras`: 固定相机位置、观察目标和垂直视场角；
- `objects`: 茶具 preset、颜色与平面随机范围；
- `randomization`: 物体中心最小间距和最大重采样次数。

配置在启动时严格校验。向已有 dataset 写入时，配置哈希必须与 `dataset.json` 一致，避免把不同相机、分辨率或物理参数的数据静默混在一起。需要换配置时应使用新的 dataset 目录。

## 数据契约

```text
data/teaware/
  dataset.json                          数据集级配置、生成器和 MuJoCo 版本
  dataset.jsonl                         每个已完整提交 episode 的一行索引
  episodes/episode_000000/
    manifest.json                       seed、标定、标签、帧索引和 provenance
    trajectory.npz                      同步状态序列
    frames/000000/
      front_left_rgb.jpg
      front_left_depth.npy              float32，米，shape=(H,W)
      front_left_depth_preview.png      仅用于浏览，不用于训练
      front_left_segmentation.npy       int32，shape=(H,W,2)
      front_left_instance.npy           int32 body id，shape=(H,W)
      front_left_segmentation_preview.png
      ...                               其余相机同样命名
```

`segmentation.npy[..., 0]` 是 MuJoCo object id，`[..., 1]` 是 object type；它是保真保存的 MuJoCo 原始输出。`instance.npy` 将 geom 聚合为 body id，背景为 `-1`，同一个茶壶的壶身、壶嘴和把手具有相同实例 id。`manifest.json` 同时保存 geom 和 body label map。不要从彩色 preview 反推标签。

`trajectory.npz` 包含：

| 数组 | shape | 含义 |
|---|---:|---|
| `time` | `(T,)` | episode 相对时间，秒 |
| `qpos` | `(T,nq)` | MuJoCo generalized position |
| `qvel` | `(T,nv)` | MuJoCo generalized velocity |
| `ctrl` | `(T,nu)` | actuator control |
| `body_position` | `(T,N,3)` | 茶具 world position，米 |
| `body_quaternion` | `(T,N,4)` | 茶具 world quaternion，wxyz |
| `body_names` | `(N,)` | 上述物体轴的名称 |
| `arm_qpos/qvel/ctrl` | `(T,R,7)` | 每台机械臂的七轴状态与控制量 |
| `hand_qpos/qvel/ctrl` | `(T,R,12)` | 每只手的状态与控制量；超过 `hand_dof` 的位置补零 |
| `tcp_position` | `(T,R,3)` | 每台机器人 TCP 的 world position |
| `tcp_quaternion` | `(T,R,4)` | 每台机器人 TCP 的 world quaternion，wxyz |
| `robot_ids/hand_types/hand_dof` | `(R,)` | 机器人轴的语义和有效手自由度 |

schema v2 的 `manifest.json` 还保存每台机器人的 id、hand type、handedness、基座位姿和关节名。旧 schema v1 episode 仍可验证。相机 `intrinsics`、`T_world_camera`、`T_camera_world` 和 MuJoCo 相机轴约定记录在每个 manifest。RGB、深度、分割和状态使用相同 `frame_id`；episode 先写入隐藏临时目录，全部完成后再原子改名并追加索引，因此浏览器不会看到半写入记录。

## 网页/API

网页调用的稳定接口如下：

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/api/status` | 数据集、相机、物体和当前 seed |
| `POST` | `/api/randomize` | 用指定或随机 seed 重置场景 |
| `POST` | `/api/collect` | 采集 1-100 个 episode |
| `GET` | `/api/live/{camera}/{modality}` | 当前 RGB/depth/segmentation preview |
| `GET` | `/api/episodes` | episode 列表与校验状态 |
| `GET` | `/api/episodes/{id}` | manifest 与校验错误 |
| `GET` | `/api/episodes/{id}/image` | 历史帧 preview |

FastAPI 的机器可读接口文档在 `/docs`。

## 开发与验证

```bash
uv run ruff check src tests
uv run pytest -q
uv run teaware-mj audit-release --root .
```

最小 smoke test 会真正编译 MJCF、创建离屏 renderer、采集 RGB/depth/segmentation，并对落盘数据执行 schema 校验。它不连接任何真实机器人或相机。

## 当前边界

- 这是场景/传感器数据采集仓库，不包含抓取策略、逆运动学、厂商 SDK 或真机控制。
- xHand 从随仓库发布的左右手 URDF 生成 MJCF，保留原始 mesh、惯量、关节原点、轴向、限位和双臂安装变换；MuJoCo position actuator 增益属于本采集环境参数。
- 茶具是参数化近似几何体；替换 mesh 时应同时核对单位、质心、惯量和碰撞简化。
- 当前随机化覆盖平面位置和 yaw；材质、光照、相机扰动可以继续在 `scene.py` 和 YAML schema 中扩展。
- `depth_preview.png` 使用逐帧百分位拉伸，只适合人工查看；算法必须读取 `depth.npy`。

## 第三方资产

`src/teaware_mujoco/assets/ufactory_xarm7/` 来自 MuJoCo Menagerie 的 UFACTORY xArm7 模型，使用 BSD-3-Clause License；上游 `LICENSE`、`README.md` 和 `CHANGELOG.md` 已原样保留。`src/teaware_mujoco/assets/xhand/` 包含原项目使用的 xHand URDF 与 mesh。完整说明见 `THIRD_PARTY_NOTICES.md` 和 `docs/PUBLIC_RELEASE.md`。仓库不包含 xHand SDK、业务代码或运行数据。
