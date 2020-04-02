import os
from nas_201_api import NASBench201API as API
api = API('$path_to_meta_nas_bench_file')
# api = API('NAS-Bench-201-v1_0-e61699.pth')
# api = API('{:}/{:}'.format(os.environ['TORCH_HOME'], 'NAS-Bench-201-v1_0-e61699.pth'))


num = len(api)
for i, arch_str in enumerate(api):
  print ('{:5d}/{:5d} : {:}'.format(i, len(api), arch_str))