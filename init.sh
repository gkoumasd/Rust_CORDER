if [ ! -f unique.tar.xz ]; then
	wget https://github.com/yijunyu/cargo-geiger/raw/datasets/datasets/unique.tar.xz
fi
if [ ! -d data/safe ]; then
	tar xfJ unique.tar.xz -C data/
fi
if [ ! -f treesitter_rust_train_test_val/train/safe/abi-all_names.rs ]; then
	pip install -r requirements.txt
	python split_data.py
fi
if [ ! -f treesitter_rust_train_test_val/treesitter_rust-buckets-train.pkl ]; then
	cd script
	source process_data.sh
	cd -
fi
