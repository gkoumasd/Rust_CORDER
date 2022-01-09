PARSER=treesitter_rust
NODE_TYPE_VOCAB_PATH=../vocab/${PARSER}/node_type/type.txt
NODE_TOKEN_VOCAB_PATH=../vocab/${PARSER}/node_token/token.txt
PYTHON=python3
TEST_PATH=/tmp/treesitter_rust-buckets.pkl
MODEL_PATH=../model/
BATCH_SIZE=1
CHECKPOINT_EVERY=1
TREE_SIZE_THRESHOLD_UPPER=100
TREE_SIZE_THRESHOLD_LOWER=0
CUDA=0
VALIDATING=1
NODE_TYPE_DIM=100
NODE_TOKEN_DIM=100
CONV_OUTPUT_DIM=100
NUM_CONV=2
EPOCH=120
NODE_INIT=2
BEST_F1=0.0

${PYTHON} process_data.py \
 --node_type_vocab_path ${NODE_TYPE_VOCAB_PATH} \
 --node_token_vocab_path ${NODE_TOKEN_VOCAB_PATH} \
 --data_path ${TEST_PATH} \
 --parser ${PARSER} \
$@

${PYTHON} test_tbcnn.py \
--test_path ${TEST_PATH} \
--batch_size ${BATCH_SIZE} \
--checkpoint_every ${CHECKPOINT_EVERY} \
--cuda ${CUDA} \
--validating ${VALIDATING} \
--tree_size_threshold_upper ${TREE_SIZE_THRESHOLD_UPPER} \
--tree_size_threshold_lower ${TREE_SIZE_THRESHOLD_LOWER} \
--model_path ${MODEL_PATH} \
--node_type_dim ${NODE_TYPE_DIM} \
--node_token_dim ${NODE_TOKEN_DIM} \
--node_type_vocabulary_path ${NODE_TYPE_VOCAB_PATH} \
--token_vocabulary_path ${NODE_TOKEN_VOCAB_PATH} \
--epochs ${EPOCH} \
--parser ${PARSER} \
--node_init ${NODE_INIT} \
--num_conv ${NUM_CONV} \
--conv_output_dim ${CONV_OUTPUT_DIM}

