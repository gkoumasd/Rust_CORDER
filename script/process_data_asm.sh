PARSER=treesitter_rust
TRAIN_DATA_PATH=../${PARSER}_train_test_val/train
TEST_DATA_PATH=../${PARSER}_train_test_val/test
VAL_DATA_PATH=../${PARSER}_train_test_val/val
NODE_TYPE_VOCAB_PATH=../vocab/${PARSER}/node_type/type_asm.txt
NODE_TOKEN_VOCAB_PATH=../vocab/${PARSER}/node_token/token.txt
LANGUAGE=asm
PYTHON=python3
${PYTHON} process_data.py \
--data_path ${TRAIN_DATA_PATH} --node_type_vocab_path ${NODE_TYPE_VOCAB_PATH} --node_token_vocab_path ${NODE_TOKEN_VOCAB_PATH} \
--parser ${PARSER} --language ${LANGUAGE}
${PYTHON} process_data.py \
--data_path ${TEST_DATA_PATH} --node_type_vocab_path ${NODE_TYPE_VOCAB_PATH} --node_token_vocab_path ${NODE_TOKEN_VOCAB_PATH} \
--parser ${PARSER} --language ${LANGUAGE}
${PYTHON} process_data.py \
--data_path ${VAL_DATA_PATH} --node_type_vocab_path ${NODE_TYPE_VOCAB_PATH} --node_token_vocab_path ${NODE_TOKEN_VOCAB_PATH} \
--parser ${PARSER} --language ${LANGUAGE}