#!/bin/bash
export BASH_SILENCE_DEPRECATION_WARNING=1
if ! command -v cargo >/dev/null 2>&1; then
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
fi
if ! command -v sccache >/dev/null 2>&1; then 
	cargo install sccache --features=openssl/vendored
fi
export RUSTC_WRAPPER=sccache
if ! command -v cargo-geiger >/dev/null 2>&1; then 
	cargo install cargo-geiger --features vendored-openssl
fi
if ! command -v tree-sitter >/dev/null 2>&1; then 
	cargo install tree-sitter-cli
fi
if ! command -v tree-grepper  >/dev/null 2>&1; then 
	cargo install --git https://github.com/BrianHicks/tree-grepper
fi
function pip_install() {
	n=$(pip3 list | grep $1 | wc -l)
	#if ! python3 -c "import $1" &> /dev/null; then
	if [ "$n" == "0" ]; then
	   if [ "$1" == "torch" ]; then
		pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
           else
		pip3 install $1
	   fi
	fi
}
export -f pip_install
pip_install tqdm
pip_install sklearn
pip_install transformers
pip_install torch
pip_install bidict
pip_install tree-sitter-parsers
pip_install powerline-status
pip_install tensorflow
pip_install keras-radam
pip_install pickle5
export LD_LIBRARY_PATH=$HOME/anaconda3/lib
if ! command -v s3cmd >/dev/null 2>&1; then 
	git clone https://github.com/s3tools/s3cmd.git
	pip_install s3cmd
fi
alias s3cmd="python3 $HOME/s3cmd/s3cmd"
export PATH=$HOME/bin:$PATH
if ! command -v parallel >/dev/null 2>&1; then 
	wget http://git.savannah.gnu.org/cgit/parallel.git/plain/10seconds_install
	sh 10seconds_install
	echo will cite | parallel --citation
fi
if ! command -v conda >/dev/null 2>&1; then 
	wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
	sh Anaconda3-2021.11-Linux-x86_64.sh
       conda install -c anaconda cudnn
fi

PYTHON=python3
TEST_PATH=/tmp/treesitter_rust-buckets.pkl
PARSER=treesitter_rust
NODE_TYPE_VOCAB_PATH=../vocab/${PARSER}/node_type/type.txt
NODE_TOKEN_VOCAB_PATH=../vocab/${PARSER}/node_token/token.txt

${PYTHON} process_data.py \
 --node_type_vocab_path ${NODE_TYPE_VOCAB_PATH} \
 --node_token_vocab_path ${NODE_TOKEN_VOCAB_PATH} \
 --data_path ${TEST_PATH} \
 --parser ${PARSER} \
$@

MODEL_PATH=../model/rust_classification_task/
CUDA=0
BATCH_SIZE=1
NODE_TYPE_DIM=100
NODE_TOKEN_DIM=100
CONV_OUTPUT_DIM=100
NUM_CONV=2
NODE_INIT=2

${PYTHON} test_tbcnn.py \
--test_path ${TEST_PATH} \
--batch_size ${BATCH_SIZE} \
--cuda ${CUDA} \
--model_path ${MODEL_PATH} \
--node_type_dim ${NODE_TYPE_DIM} \
--node_type_vocabulary_path ${NODE_TYPE_VOCAB_PATH} \
--node_token_dim ${NODE_TOKEN_DIM} \
--token_vocabulary_path ${NODE_TOKEN_VOCAB_PATH} \
--conv_output_dim ${CONV_OUTPUT_DIM} \
--parser ${PARSER} \
--node_init ${NODE_INIT} \
--num_conv ${NUM_CONV} \

