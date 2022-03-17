if [ ! -f new-syntax.tar.bz2 ]; then
	wget http://bertrust.s3.amazonaws.com/new-syntax.tar.bz2
	mkdir data; cd data; tar xfj ../new-syntax.tar.bz2; cd ..
fi
mkdir -p treesitter_asm_train_test_val/{train,test,val}/{safe,unsafe}
python split_data.py
