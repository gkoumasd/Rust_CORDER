PARSER=treesitter_rust
TEST_PATH=data/treesitter_rust-buckets-.pkl
NODE_TYPE_VOCABULARY_PATH=../vocab/${PARSER}/node_type/type.txt
TOKEN_VOCABULARY_PATH=../vocab/${PARSER}/node_token/token.txt
MODEL_PATH=../model/
BATCH_SIZE=1
CHECKPOINT_EVERY=1
TREE_SIZE_THRESHOLD_UPPER=100
TREE_SIZE_THRESHOLD_LOWER=0
CUDA=-1
VALIDATING=1
NODE_TYPE_DIM=100
NODE_TOKEN_DIM=100
CONV_OUTPUT_DIM=100
NUM_CONV=2
EPOCH=120
PYTHON=python3
NODE_INIT=2
BEST_F1=0.0
${PYTHON} test_tbcnn.py \
--test_path ${TEST_PATH} --batch_size ${BATCH_SIZE} \
--checkpoint_every ${CHECKPOINT_EVERY} --cuda ${CUDA} --validating ${VALIDATING} \
--tree_size_threshold_upper ${TREE_SIZE_THRESHOLD_UPPER} \
--tree_size_threshold_lower ${TREE_SIZE_THRESHOLD_LOWER} \
--model_path ${MODEL_PATH} \
--node_type_dim ${NODE_TYPE_DIM} --node_token_dim ${NODE_TOKEN_DIM} \
--node_type_vocabulary_path ${NODE_TYPE_VOCABULARY_PATH} \
--token_vocabulary_path ${TOKEN_VOCABULARY_PATH} --epochs ${EPOCH} --parser ${PARSER} \
--node_init ${NODE_INIT} --num_conv ${NUM_CONV} --conv_output_dim ${CONV_OUTPUT_DIM}