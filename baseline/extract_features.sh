TRAIN_DATA_PATH=../codeBERT/data_train_test_val/train
TEST_DATA_PATH=../codeBERT/data_train_test_val/test
VAL_DATA_PATH=../codeBERT/data_train_test_val/val
MAX_SEQ=80
TOKEN_VOC=vocab/cov.txt
FLAG_VAL=val
PYTHON=python3
${PYTHON} extract_features_script.py \
--data_path ${TRAIN_DATA_PATH} --max_seq ${MAX_SEQ}  --token_voc ${TOKEN_VOC}$
${PYTHON} extract_features_script.py \
--data_path ${TEST_DATA_PATH} --max_seq ${MAX_SEQ}  --token_voc ${TOKEN_VOC}$
${PYTHON} extract_features_script.py \
--data_path ${VAL_DATA_PATH} --max_seq ${MAX_SEQ}  --token_voc ${TOKEN_VOC}$
