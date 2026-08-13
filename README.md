# sEEG Operator Workbench

一个面向 sEEG 的、任务无关且可追溯的数学算子原型。它不是“自动解释脑机制”的工具，而是把不同格式的数据转换为统一的命名张量，并让每个特征保留：

- 输入文件指纹；
- 命名维度、坐标和单位；
- 算子及全部参数；
- 重参考矩阵；
- 数学与信号约束检查；
- 可复现的来源链。

## 当前样本数据

`/Users/heting/Documents/readGripData/0807华山grip flight`

| 文件 | 内容 | 形状 | 采样率 | 时长 |
|---|---|---:|---:|---:|
| `testS001R09.dat.larkcache` | 256通道主流 | 256 × 286800 | 2000 Hz | 143.40 s |
| `testS001R09_1.dat` | 握力/飞行任务流 | 1 × 31488 | 256 Hz | 123.00 s |
| `testS001R11.dat.larkcache` | 256通道主流 | 256 × 314320 | 2000 Hz | 157.16 s |
| `testS001R11_1.dat` | 握力/飞行任务流 | 1 × 38208 | 256 Hz | 149.25 s |

主流文件头标为 `SignalGeneratorADC`，并明确包含 `NoiseAmplitude=30muV`、`DCOffset=60muV`，通道名为空，设备ID、在线参考和真实触点映射也未写入文件。前10秒数据进一步表现为每通道约299个整数电平、范围约`-150..150 ADU`、通道间近零相关、256通道RMS异常一致。这组证据高度支持它是BCI2000信号发生器产生的独立模拟噪声，而不是医院采集器的真实sEEG。除非另有采集链证据，不能用它验证任何神经生理特征。

任务流包含 `GripForceRaw`、`GripForceNormalized`、`GamePhase`、`Collision`、`Feedback`、球位置与试次结果等BCI2000 states。

## 运行

本项目只依赖 NumPy。当前Codex工作区可这样运行：

```bash
PYTHONPATH=src /Users/heting/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m seegops.cli inspect \
  '/Users/heting/Documents/readGripData/0807华山grip flight/testS001R09.dat.larkcache'
```

生成两个run的检查报告：

```bash
PYTHONPATH=src /Users/heting/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  examples/inspect_huashan.py
```

运行测试：

```bash
PYTHONPATH=src /Users/heting/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest discover -s tests -v
```

## 算子示例

```python
from seegops.io import read_bci2000
from seegops.operators import ApplyGain, BipolarReference, Window, FFTPowerSpectrum, BandPower

recording = read_bci2000("recording.dat")
signal = ApplyGain()(recording.signal)
signal = BipolarReference()(signal)
windows = Window(length_s=1.0, step_s=0.25)(signal)
spectrum = FFTPowerSpectrum()(windows)
theta = BandPower(
    4, 8,
    require_cycles=4,
    source_duration_s=1.0,
)(spectrum)
```

如果把窗口改成0.2秒，最低频率4 Hz只包含0.8个周期，算子会拒绝执行，而不是返回一个看似合法的Theta数值。

## 第一版已有内容

- BCI2000 `int16/int32/float32` 内存映射读取；
- BCI2000 state按位解码；
- 选择、连续分窗、TrialTable/EventTable驱动的事件分段；
- 原始增益换算；
- 双极参考和CAR；
- 重参考矩阵和触点对来源记录；
- 基线减法、比值及dB；
- Hann/boxcar FFT功率谱；
- 频带平均/积分和最低周期数约束；
- 均值、方差及基础QC；
- `StorageTime`起点诊断；
- 单元测试和真实数据集成测试。

## 尚未自动完成的事情

1. **两条流的精确同步**：开始时间相差约0.13–0.17秒，但持续时间不同。仅凭`StorageTime`不足以证明采样锁定和时钟无漂移；应寻找共享触发、状态、脉冲或设备时钟映射。
2. **主流真实性质**：当前证据高度指向模拟噪声；必须找到原始医院sEEG文件或证明缓存桥接确实覆盖了发生器输出，才可进入生理分析。
3. **触点元数据**：需要通道—触点映射、在线参考、SOZ、坏触点、灰白质与坐标。
4. **滤波器**：第一版故意没有自己实现临床级FIR/IIR，以免在缺少SciPy和明确边界策略时制造相位/边缘伪迹。
5. **连接、PAC与复杂度**：需要先实现配套代理数据、参数敏感性和共享触点检查，再进入主库。

## 设计原则

算子统一，但解释不自动化：

```text
文件适配器 → Signal(命名维度/坐标/单位) → 表征算子 → 选择 → 归约 → Feature
                                      ↘ provenance / QC / constraints ↗
```

任何新算子至少需要：数学定义、维度契约、单位、最低数据要求、模拟恢复测试、零模型和参数敏感性说明。

## Trial、Event和Epoch

三者现在严格分离：

- `TrialTable`：实验范式定义的完整行为周期；
- `EventTable`：trial内部或trial外的时间点/区间；
- `Epoch`：围绕某一类event截取的固定长度分析片段。

```python
from seegops import Event, EventTable, Trial, TrialTable
from seegops.operators import Epoch

trials = TrialTable([
    Trial("trial-01", onset_s=10.0, offset_s=18.0, condition="success"),
    Trial("trial-02", onset_s=20.0, offset_s=27.0, condition="failure"),
])

events = EventTable([
    Event("target-01", "target", onset_s=11.0, trial_id="trial-01"),
    Event("feedback-01", "feedback", onset_s=17.0, trial_id="trial-01"),
    Event("target-02", "target", onset_s=21.0, trial_id="trial-02"),
    Event("feedback-02", "feedback", onset_s=26.0, trial_id="trial-02"),
], trials=trials)

feedback_epochs = Epoch(
    events=events,
    event_type="feedback",
    tmin_s=-1.0,
    tmax_s=1.0,
    trial_boundary="reject",
    overlap="flag",
)(signal)
```

输出维度为`channel × epoch × time`，而不是`channel × trial × time`。每个epoch附带`event_id`、`event_type`、`trial_id`、事件时间、源样本、trial越界和epoch重叠标志。没有语义表时仍可用`event_samples=`兼容旧式入口，但输出会标记`legacy_event_samples=True`。
