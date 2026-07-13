# Topo9-FiLM L3-Only

## 实验目的
验证 **L4 是否是唯一有效的耦合点**。将 FiLM 耦合从 L4 移到 L3，观察性能变化。

## 配置
- L3: `TopoCoupledPointConvD_v2` (FiLM + BN)
- L4: 标准 `PointConvD` (无拓扑耦合)

## 修改内容
`ppwc.py`:
```python
# 原始
self.level3 = PointConvD(...)
self.level4 = TopoCoupledPointConvD_v2(...)

# 改为
self.level3 = TopoCoupledPointConvD_v2(...)
self.level4 = PointConvD(...)
```

forward 中对应交换 topo 的传递位置。

## 预期结果
**~8.0+**。如果确实很差，就坐实了 "L4-only" 的不可替代性。
