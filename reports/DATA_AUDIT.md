# 0807华山 grip flight 数据审计

## 结论

这组目录包含两类BCI2000数据：握力/飞行任务流是真实的任务状态记录；标称的256通道主流高度疑似BCI2000内置`SignalGeneratorADC`产生的模拟噪声，不能视为真实sEEG。

## 直接证据

- 主流文件头：`SourceCh=256`、`SamplingRate=2000`、`SignalType=0 (int16)`。
- 采集模块：`SignalGeneratorADC`，而不是临床放大器模块。
- 参数明确写有：`NoiseAmplitude=30muV`、`DCOffset=60muV`、`SineAmplitude=0`。
- `ID_Amp`、`ID_Montage`、`ID_System`均为空。
- `ChannelNames=0`，没有真实触点名称。
- 每通道取值范围约为`-150..150 ADU`，约299个整数电平。
- 256通道RMS高度一致，通道间相关接近零，符合独立均匀噪声而非sEEG的表现。

## 任务流

`*_1.dat`为单通道、256 Hz的握力任务流，包含完整BCI2000 states：

- `GripForceRaw` / `GripForceNormalized`
- `GamePhase`
- `Feedback`
- `Collision` / `CollisionObject`
- `BallWorldX/Y` / `BallVelocityY`
- `FlightTrialResult`

任务流可用于测试状态解码、事件抽取和行为特征算子。

## 同步

- Run 09：任务流比主流记录起点早约168 ms。
- Run 11：任务流比主流记录起点早约135 ms。
- 两条流的最终时长不同。

`StorageTime`只能提供粗起点。若未来取得真实sEEG，需要用共享TTL、状态计数器、同步脉冲或明确时钟映射估计偏移与漂移，不能只根据开始时间插值合并。

## 下一份真实sEEG需要的最小信息

1. 原始文件及厂商/放大器型号；
2. 采样率、硬件滤波和在线参考；
3. 通道—触点映射；
4. 触点坐标和电极杆顺序；
5. SOZ、IED、病灶与坏触点标注；
6. 与BCI2000任务流共享的同步字段或TTL通道。

