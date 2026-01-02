
# Config for reproduction
graph_d = 100
graph_l = 5
graph_vocab_size = 'auto' # corresponds to 1max
graph_holdout_percentage = 0.2
num_pause_tokens = 5
use_undirected = True
use_directional_tokens = True # from values [false, true], picking true
weight_tying = True # Not in get_default_config?? Wait, need to check model.py
interleave_dataset = True

# Model
n_layer = 1
n_head = 8
n_embd = 32
dropout = 0
embd_dropout = 0
bias = True

learning_rate = 1e-3
epochs = 500
gradient_accumulation_steps = 1
compile = False # Faster startup
eval_interval = 1000
log_interval = 1
wandb_log = False
device = 'cpu' # Use CPU for deterministic/easy run
