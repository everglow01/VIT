import onnx

model = onnx.load("detection_model.onnx")

# 打印输入输出
print("=== Inputs ===")
for inp in model.graph.input:
    print(inp.name, [d.dim_value for d in inp.type.tensor_type.shape.dim])

print("=== Outputs ===")
for out in model.graph.output:
    print(out.name, [d.dim_value for d in out.type.tensor_type.shape.dim])

# 打印节点类型统计
from collections import Counter
ops = Counter(node.op_type for node in model.graph.node)
print("=== Op Types ===")
for op, count in ops.most_common():
    print(f"  {op}: {count}")