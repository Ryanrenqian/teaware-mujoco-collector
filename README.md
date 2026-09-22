# Teaware MuJoCo Collector

一个可独立运行的 MuJoCo 茶具场景数据采集仓库。它将场景定义、随机化、传感器采集、数据契约和网页检查放在同一套小型工具中，不依赖 `waic-demo4` 的 Python 包或运行目录。

当前支持单臂两指夹爪、单臂 xHand 和双臂 xHand 三种场景，以及 10 套、40 件真实茶具模型。每件茶具使用原始视觉 OBJ 和 16 个凸碰撞 OBJ；默认场景加载 `teapot_porcelain_red` 四件套。xHand 使用原项目中的左右手 URDF、视觉/碰撞 mesh、惯量、关节轴和限位，并保持与原控制栈一致的 12 关节顺序；仓库不包含 SDK 或真机配置。

采集执行支持统一 policy 接口：内置 staged 运控 baseline、独立 TRO 抓取、进程内 VLA/model callable，以及 HTTP VLA server。所有后端都输出带时间间隔的关节位置 `ActionChunk`，共用同一套限位、仿真执行和数据记录逻辑。TRO 抓取链只依赖独立 TRO checkout、配置和权重，不 import 或读取 `waic-demo4`。

## 快速开始

要求 Python 3.11+。推荐使用 [uv](https://docs.astral.sh/uv/)：

```bash
git clone <this-repository>
cd teaware-mujoco-collector
uv sync --extra dev

# 采集 3 个 episode，seed 分别为 100、101、102
uv run teaware-mj --config configs/tea_table.yaml collect \
  --output data/teaware \
  --episodes 3 \
  --seed 100

# 检查所有 episode 的文件、shape 和有限值
uv run teaware-mj validate --dataset data/teaware

# 启动网页采集台
uv run teaware-mj --config configs/tea_table.yaml serve \
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
| `configs/tro_xhand.yaml` | 独立本地 TRO + MuJoCo IK 抓取 |
| `configs/tro_xhand_mock.yaml` | 不加载模型的 TRO 抓取闭环验收 |

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
  policy/                              运控、TRO、本地模型和远端 VLA 的统一接口与 runner
  schema.py                            episode 发现与完整性校验
  release_audit.py                     许可证和敏感信息发布门禁
  web.py                               FastAPI 和图像/episode API
  static/                              无构建步骤的网页前端
  assets/ufactory_xarm7/               vendored xArm7 MJCF、mesh、上游许可证
  assets/xhand/                        左右 xHand URDF、法兰、STL 视觉与 OBJ 凸包资产
  assets/teaware/                      10 套茶具的视觉/碰撞 OBJ、catalog 和物理 profile
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
- `policy`: policy 类型、任务文本、控制频率和 action horizon；
- `table`: 茶桌中心和完整尺寸；
- `scene_profile`: 数据中记录的场景标识；
- `robots`: 一台或多台机械臂的 id、手型、左右手、基座位姿、arm/hand home joint 和运动幅度；
- `cameras`: 固定相机位置、观察目标和垂直视场角；
- `objects`: 茶具 preset、可选 `asset_id`、颜色与平面随机范围；
- `randomization`: 物体中心最小间距和最大重采样次数。

配置在启动时严格校验。向已有 dataset 写入时，配置哈希必须与 `dataset.json` 一致，避免把不同相机、分辨率或物理参数的数据静默混在一起。需要换配置时应使用新的 dataset 目录。

默认四件套的 `asset_id` 分别为 `teapot_porcelain_red__object_000` 至 `object_003`，对应茶壶、茶杯、公道杯和茶叶罐。`assets/teaware/catalog.json` 列出全部 40 个可用 id；替换 YAML 中的 id 即可切换茶具，同一对象的 `name` 无需改变，因此 trajectory 和 instance label 契约保持稳定。不写 `asset_id` 时仍可使用旧的参数化 preset。

## TRO 抓取

`tro_grasp` 是独立的仿真抓取链：从当前茶具 mesh 和随机位姿构造目标/环境点云，在机器人 base frame 调用 TRO，依次尝试候选；每个候选用当前 MuJoCo 模型的 Jacobian 做预抓取、最终抓取和抬升 IK。选中候选后生成 `pregrasp → grasp → close → lift → return → release` 的机械臂与 xHand 同步轨迹。

先运行不需要权重的闭环验收：

```bash
uv run teaware-mj --config configs/tro_xhand_mock.yaml collect \
  --output data/tro-mock \
  --episodes 1 \
  --seed 7

uv run teaware-mj validate --dataset data/tro-mock
```

正式本地 TRO 需要安装可选依赖，并提供独立 TRO checkout、部署 YAML 和主策略 checkpoint：

```bash
uv sync --extra dev --extra tro

export TRO_ROOT=/absolute/path/to/tro
export TRO_CONFIG=/absolute/path/to/infer_xhand.yaml
export TRO_CHECKPOINT=/absolute/path/to/step_xxx.pth

uv run teaware-mj --config configs/tro_xhand.yaml collect \
  --output data/tro-xhand \
  --episodes 10 \
  --seed 100
```

TRO checkout 需要包含 `model/vqvae_encoder.py` 以及部署 YAML 所引用的 object/env encoder。`configs/tro_xhand.yaml` 默认使用 `cuda:0`；无 CUDA 时可以将 `device` 改为 `cpu`，但推理速度会明显降低。

本地 loader 直接读取 TRO 的 link-pose 输出。机械臂 IK 和 xHand link retarget 均由当前仓库中的 MuJoCo 模型完成，不依赖 Pyroki、cuRobo、硬件驱动或 `waic-demo4`。当前规划器是仿真用的阻尼最小二乘 IK 和关节空间平滑插值，不等价于 cuRobo 的全局避障规划；不可达候选会被丢弃，并继续尝试下一个 TRO 候选。

也可以把 TRO 单独部署成 HTTP 服务，将 `tro.backend` 改为 `http` 并设置 `tro.url`。客户端调用 `POST /v1/grasp-candidates`，请求中包含 base64 NPY 格式的 `object_points_npy_base64`、`environment_points_npy_base64`、`hand_type` 和 `num_candidates`。响应格式为：

```json
{
  "candidates": [
    {
      "rank": 0,
      "root_link_name": "right_hand_link",
      "root_pose_base": [[1, 0, 0, 0.4], [0, 1, 0, 0], [0, 0, 1, 0.2], [0, 0, 0, 1]],
      "hand_q": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
  ]
}
```

`hand_q` 可以省略：本地 raw TRO backend 会把 `pred_links` 交给 MuJoCo xHand IK；HTTP backend 也可以返回 `finger_q`/`q_pk` 的 joint-name mapping。候选位姿采用机器人 base frame，长度单位为米，旋转矩阵为右手系。

`tro.grasp_constraint.enabled` 控制仿真抓取稳定器。启用后，只有手掌进入 `max_distance_m` 且阶段进入 `close` 才建立当前相对位姿的 MuJoCo weld，`release` 阶段自动解除。该设置会写入 episode policy provenance；要评估纯接触物理时将其设为 `false`。

## Policy 与 VLA

默认配置使用确定性的 staged joint-space 运控 policy：home、approach、close、lift、return。它负责快速验证 action/采集链：

```yaml
policy:
  type: scripted_motion
  task: Move through approach, close, lift, and return stages around the tea set.
  control_hz: 10
  action_horizon: 4
```

远端 VLA 使用同一配置位置：

```yaml
policy:
  type: remote_vla
  task: Pick up the red teapot and pour into the fairness pitcher.
  url: http://127.0.0.1:8090
  timeout_s: 30
  include_depth: false
  jpeg_quality: 90
  control_hz: 10
  action_horizon: 4
```

仓库提供 hold-position mock server，用来先验证 HTTP 和采集闭环：

```bash
uv run teaware-mj serve-policy --host 127.0.0.1 --port 8090
```

VLA server 实现 `POST /v1/actions`。请求包含任务文本、多相机 JPEG、机器人状态、TCP 位姿和茶具状态；响应为：

```json
{
  "action": {
    "arm_q_target": [[[0, 0, 0, 0, 0, 0, 0]]],
    "hand_q_target": [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]],
    "dt_s": 0.1,
    "action_mode": "joint_position"
  },
  "model": {"name": "model-name", "checkpoint": "model-version"}
}
```

数组 shape 分别是 `(H,R,7)` 和 `(H,R,12)`。`H` 是 action horizon，`R` 是机器人数量；二指夹爪只使用 hand 数组的第一个位置。当前只接受 joint-position action，所有输出都经过有限值、shape、机器人数量和 actuator limit 校验。

进程内模型通过 `LocalModelPolicy` 注入，无需启动 HTTP server：

```python
from teaware_mujoco.collector import TeawareCollector
from teaware_mujoco.policy import LocalModelPolicy

policy = LocalModelPolicy(model.predict, name="my-vla", model_metadata={"checkpoint": "v1"})
collector = TeawareCollector(config, "data/local-vla", policy=policy)
```

外部规划器产生的 timed trajectory 也可以包装为相同 policy；对象只需暴露 `positions`、`times` 或 `cmd_dt`，运行时不需要原规划工程：

```python
from teaware_mujoco.policy import TimedTrajectoryPolicy

policy = TimedTrajectoryPolicy.from_waic_timed_trajectory(
    timed_trajectory,
    hand_positions=xhand_q12_trajectory,
    control_hz=10,
    action_horizon=4,
    provenance={"planner": "curobo"},
)
```

每个观测帧采用 receding-horizon 方式请求新的 chunk。VLA 调用失败会使临时 episode 回滚，不会回退到 scripted policy，从而避免不同来源动作混入同一条轨迹。

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
| `policy_arm_q_target` | `(T,R,7)` | policy 输出的机械臂关节目标 |
| `policy_hand_q_target` | `(T,R,12)` | policy 输出的手部关节目标 |
| `policy_latency_ms` | `(T,)` | 每个观测帧的推理延迟 |
| `policy_action_horizon` | `(T,)` | 每个 action chunk 的 horizon |
| `policy_action_mode/policy_stage` | `(T,)` | 动作模式和运控阶段标签 |

schema v3 的 `manifest.json` 还保存 policy 类型、名称、任务、模型或 server provenance；schema v2 保存每台机器人的 id、hand type、handedness、基座位姿和关节名。旧 schema v1/v2 episode 仍可验证。相机 `intrinsics`、`T_world_camera`、`T_camera_world` 和 MuJoCo 相机轴约定记录在每个 manifest。RGB、深度、分割、policy action 和状态使用相同 `frame_id`；episode 先写入隐藏临时目录，全部完成后再原子改名并追加索引，因此浏览器不会看到半写入记录。

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
- xHand 从随仓库发布的左右手 URDF 生成 MJCF，保留原始法兰、mesh、惯量、关节原点、轴向、限位和双臂安装变换；MuJoCo position actuator 增益属于本采集环境参数。
- 茶具 catalog 的质量是估计值；仿真按 `assets/teaware/physics_profile.json` 等比缩放并将单件质量限制在 100 g。视觉颜色继续由 YAML `rgba` 控制。
- 当前随机化覆盖平面位置和 yaw；材质、光照、相机扰动可以继续在 `scene.py` 和 YAML schema 中扩展。
- `depth_preview.png` 使用逐帧百分位拉伸，只适合人工查看；算法必须读取 `depth.npy`。

## 第三方资产

`src/teaware_mujoco/assets/ufactory_xarm7/` 来自 MuJoCo Menagerie 的 UFACTORY xArm7 模型，使用 BSD-3-Clause License；上游 `LICENSE`、`README.md` 和 `CHANGELOG.md` 已原样保留。`src/teaware_mujoco/assets/xhand/` 包含原项目使用的 xHand URDF 与 mesh。完整说明见 `THIRD_PARTY_NOTICES.md` 和 `docs/PUBLIC_RELEASE.md`。仓库不包含 xHand SDK、业务代码或运行数据。

`src/teaware_mujoco/assets/teaware/` 包含 10 套真实茶具的预处理 OBJ、凸碰撞网格和 catalog；资产已转换为 Z-up、XY 居中且最低点为 `z=0`。
