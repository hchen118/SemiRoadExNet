import torch

BATCH_SIZE=8

NUM_WORKER=1
FILE_NAME='train_12'
#FILE_NAME='base_fmwdct'
#FILE_NAME='base_dlinknet'
#FILE_NAME='base_st_pp'
LEARN_RATE=1e-4
WEIGHT_DECAY=1e-5
# LA_1,LA_2,LA_3,LA_4,LA_5=1,1,1,1,1
DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SET_NAME='deep_global_road'
#SET_NAME='massachusetts_roads'
#SET_NAME='cug_wash'
LABEL_RATE=5
EXTRA_WORD=''
PRE_EPOCH='best'
START_EPOCH=0
END_EPOCH=100
TEST_EPOCH_NUM='best'
sever_root=''

