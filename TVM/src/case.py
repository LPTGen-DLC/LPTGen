import tvm
from tvm import te, topi, tir
import numpy as np


def compile_mod(mod, np_input_list, output_shape, opt_level=3):
    with tvm.transform.PassContext(opt_level):
        ref_mod = tvm.build(mod, target='llvm')
    mod_output = tvm.nd.empty(output_shape, device=tvm.cpu(0))

    tvm_inputs = [tvm.nd.array(x) for x in np_input_list]
    ref_mod(*tvm_inputs, mod_output)
    return mod_output


data = te.placeholder((1, 3, 224, 224), dtype='float32', name='data')
op_config = {'data': data, 'output_size': (112, 112), 'pool_type': 'max', 'layout': 'NCHW'}
op_output = topi.nn.adaptive_pool(**op_config)
np_inputs = [np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype('float32')]


sch = tir.Schedule(te.create_prim_func([data, op_output]))
print(sch.mod)

ref_output = compile_mod(sch.mod, np_inputs, op_output.shape, opt_level=0)

import collects_apis
opt_mod = collects_apis.select_transforms(sch.mod)
opt_mod_output = compile_mod(opt_mod, np_inputs, op_output.shape, opt_level=3)
print(opt_mod)

np.testing.assert_allclose(
    ref_output.numpy(), opt_mod_output.numpy(), rtol=1e-3, atol=1e-3, err_msg=f"An Inconsistency bug detected."
)
