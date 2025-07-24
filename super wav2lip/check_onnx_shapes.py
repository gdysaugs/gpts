#!/usr/bin/env python3
import onnx

model = onnx.load('/app/models/onnx/wav2lip_gan.onnx')
print('入力テンソル:')
for inp in model.graph.input:
    shape = [d.dim_value if d.dim_value > 0 else "dynamic" for d in inp.type.tensor_type.shape.dim]
    print(f'{inp.name}: {shape}')

print('\n出力テンソル:')
for out in model.graph.output:
    shape = [d.dim_value if d.dim_value > 0 else "dynamic" for d in out.type.tensor_type.shape.dim]
    print(f'{out.name}: {shape}')