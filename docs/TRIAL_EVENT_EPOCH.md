# Trial、Event与Epoch数据模型

## TrialTable

Trial是实验范式定义的完整行为周期，使用记录时钟上的秒表示：

```text
trial_id | onset_s | offset_s | condition | valid | metadata
```

它回答“什么构成一次任务尝试”，不要求所有trial等长。

## EventTable

Event是时间点或区间，可通过`trial_id`连接到TrialTable：

```text
event_id | event_type | onset_s | duration_s | trial_id | value | valid | metadata
```

连接后，工具验证事件是否位于trial边界内。一个trial可拥有多个不同类型的event。

## Epoch

Epoch是分析产生物，不是范式本体。它围绕选定事件建立固定时间窗口：

```text
event onset + [tmin_s, tmax_s)
```

输出：

```text
channel × epoch × relative_time
```

默认行为：

- 越过记录起止边界的epoch被拒绝；
- 越过所连接trial边界的epoch被拒绝；
- epoch互相重叠时保留但明确标记；
- 无效event以及属于无效trial的event不进入分析；
- `tmax_s`采用右开区间。

同一TrialTable/EventTable可以产生目标锁定、动作锁定、碰撞锁定和反馈锁定等多种epoch集合。

BCI2000等离散状态可以通过上升沿生成EventTable：

```python
feedback_events = EventTable.from_state_edges(
    recording.state("Feedback"),
    sampling_rate=recording.signal.sampling_rate,
    event_type="feedback",
    trials=trials,
)
```

这一步只建立事件；若event来自任务流而Signal来自另一台sEEG设备，仍必须先应用经过验证的时钟映射，不能直接共享秒坐标。
