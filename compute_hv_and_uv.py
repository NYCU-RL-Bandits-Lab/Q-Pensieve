import argparse
import numpy as np
from copy import deepcopy
from hypervolume import InnerHyperVolume
objs = []

Ref_X_ = 0
Ref_Y_ = -4000


def compute_hypervolume_sparsity_3d(obj_batch, ref_point):
    HV = InnerHyperVolume(ref_point)
    hv = HV.compute(obj_batch)

    sparsity = 0.0
    m = len(obj_batch[0])
    for dim in range(m):
        objs_i = np.sort(deepcopy(obj_batch.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity += np.square(objs_i[i] - objs_i[i - 1])
    sparsity /= (len(obj_batch) - 1)
    
    return hv, sparsity


parser = argparse.ArgumentParser()
parser.add_argument('--log-path', type=str, required=True)
parser.add_argument('--pref-table-path', type=str, required=True)
parser.add_argument('--ref-point', type=float, nargs='+', default=[Ref_X_, Ref_Y_])

args = parser.parse_args()


data = np.load(args.log_path)
pref_table = np.load(args.pref_table_path)

hypervolume, sparsity = compute_hypervolume_sparsity_3d(np.array(data), args.ref_point)

print('hypervolume = {:.0f}, sparsity = {:.0f}'.format(hypervolume, sparsity))


print('ut = {:.0f}'.format(np.mean(np.sum(np.array(data)*np.array(pref_table),axis=1))))
