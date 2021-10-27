PARSER=treesitter_rust
DATA_PATH=data/
NODE_TYPE_VOCAB_PATH=../vocab/${PARSER}/node_type/type.txt
NODE_TOKEN_VOCAB_PATH=../vocab/${PARSER}/node_token/token.txt
PYTHON=python3
${PYTHON} process_data.py \
--data_path ${DATA_PATH} --node_type_vocab_path ${NODE_TYPE_VOCAB_PATH} --node_token_vocab_path ${NODE_TOKEN_VOCAB_PATH} \
--parser ${PARSER}
