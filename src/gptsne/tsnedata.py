import cachetools
from defaultlist import defaultlist

data_t = None
labels = None
P_tsne = None
max_P_tsne = None
degrees_of_freedom = None
_DOF = None
accesses = 0
fitnessCache = defaultlist(lambda: cachetools.LRUCache(maxsize=1e6))
dataset = None
outdir = None
nobj = 2
max_depth = 8
max_height = 14  # 17
pop_size = 100  # 1024#256  # 1024
cxpb = 0.8
mutpb = 0.2  # 1.
mutercpb = 0.  # 1.
post_cmaes = False
perplexity = 40.0
num_trees = 2
neighbours = None
NUM_NEIGHBOURS = 3
stores = 0