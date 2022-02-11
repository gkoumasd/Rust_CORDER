TRAIN_DATA_PATH=data_train_test_val/train
VAL_DATA_PATH=data_train_test_val/val
FLAG_TRAIN=train
FLAG_VAL=val
TYPE=asm
PYTHON=python3
${PYTHON} create_synthetic_data.py \
--data_path ${TRAIN_DATA_PATH} --flag ${FLAG_TRAIN} --file_type ${TYPE}
${PYTHON} create_synthetic_data.py \
--data_path ${VAL_DATA_PATH} --flag ${FLAG_VAL} --file_type ${TYPE}