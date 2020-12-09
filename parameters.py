#actor_lr = 0.00001  # learning rate of actor model
#critic_lr = 0.00005  # learning rate of critic model
actor_lr = 0.000001  # learning rate of actor model
critic_lr = 0.000005  # learning rate of critic model

lr_decay_iters = 500  # the frequency of decay the learning rate
show_action = True

#sequence_list = ["parkour","india"]
frame_strart = 5

dataset_root = "/home/smj/DataSet/DAVIS2017_o/"
#dataset_root = "/Data_HDD/smj_data/DAVIS2017/"
#dataset_root = "data1/smj_data/DAVIS2017/"

state_size = 4096  # the size of the state after flatten
action_size = 2  # the number of actions of actor
e_iters = 20000  # the number of training epoch

# big ratio:1.2, normal:0.4, small:0.1
region_ratio_list = {
"blackswan":1.2,
"drift-chicane":1.2,
"lab-coat":0.1,
"loading":0.1,
"shooting":0.1,
"soapbox":0.1,
"judo":0.1
}
