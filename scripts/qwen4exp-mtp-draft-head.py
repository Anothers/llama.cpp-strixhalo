#!/usr/bin/env python3
# Add a separately quantized draft head to a qwen4exp MTP GGUF.
#
# The MTP draft step otherwise multiplies the main model's output.weight (Q8_0, 644 MiB),
# about 80% of each draft step. This dequantizes output.weight, requantizes it to <qtype>
# and appends it as blk.<n>.nextn.shared_head_head.weight, which qwen4exp prefers for drafting.
# Q4_0 (357 MiB) takes the draft step from 4.3 to 2.86 ms on Strix Halo; decode +6-13%.
#
# usage: qwen4exp-mtp-draft-head.py <qtype: Q4_0|Q4_K|Q5_K|Q6_K> <mtp.gguf> <main-model dir> <out.gguf> [mtp layer, default 48]
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gguf-py'))
import gguf  # noqa: E402
from gguf import quants  # noqa: E402

qtype = getattr(gguf.GGMLQuantizationType, sys.argv[1])
src, main_dir, out = sys.argv[2], sys.argv[3], sys.argv[4]
layer = int(sys.argv[5]) if len(sys.argv) > 5 else 48

head = None
for f in sorted(glob.glob(os.path.join(main_dir, '*.gguf'))):
    for t in gguf.GGUFReader(f).tensors:
        if t.name == 'output.weight':
            head = t
if head is None:
    sys.exit(f'no output.weight in {main_dir}')
w = quants.dequantize(head.data, head.tensor_type)
print('output.weight', head.tensor_type.name, w.shape, flush=True)
q = quants.quantize(w, qtype)
del w

r = gguf.GGUFReader(src)
wr = gguf.GGUFWriter(out, 'qwen4exp')
for k, f in r.fields.items():
    if k.startswith('GGUF.') or k == 'general.architecture':
        continue
    vt = f.types[0]
    if vt == gguf.GGUFValueType.ARRAY:
        sub = f.types[-1]
        vals = [bytes(f.parts[i]).decode() if sub == gguf.GGUFValueType.STRING else f.parts[i].tolist()[0] for i in f.data]
        wr.add_key_value(k, vals, vt, sub_type=sub)
    elif vt == gguf.GGUFValueType.STRING:
        wr.add_key_value(k, bytes(f.parts[f.data[0]]).decode(), vt)
    else:
        wr.add_key_value(k, f.parts[f.data[0]].tolist()[0], vt)
for t in r.tensors:
    wr.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
wr.add_tensor(f'blk.{layer}.nextn.shared_head_head.weight', q, raw_shape=q.shape, raw_dtype=qtype)
wr.write_header_to_file()
wr.write_kv_data_to_file()
wr.write_tensors_to_file()
wr.close()
print('wrote', out, flush=True)
