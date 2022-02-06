TRAIN_DATA_PATH=data_train_test_val/train
TEST_DATA_PATH=data_train_test_val/test
VAL_DATA_PATH=data_train_test_val/val
FLAG_TRAIN=train
FLAG_TEST=test
FLAG_VAL=val
TYPE=rs
PYTHON=python3
${PYTHON} process_data.py \
--data_path ${TRAIN_DATA_PATH} --flag ${FLAG_TRAIN} --file_type ${TYPE}
${PYTHON} process_data.py \
--data_path ${TEST_DATA_PATH} --flag ${FLAG_TEST}  --file_type ${TYPE}
${PYTHON} process_data.py \
--data_path ${VAL_DATA_PATH} --flag ${FLAG_VAL} --file_type ${TYPE}