# Raw observation vector fed to locomotion policy (for debugging/logging)
std_msgs/Header header
float32[] obs_vector  # flattened, length = 72 * 10 (10-step history)
int32 history_len
int32 obs_dim
