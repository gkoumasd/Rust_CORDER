find treesitter_asm_train_test_val/ -name "*.asm" | while read f; do
     mv $f /tmp/$(basename $f)
     head -200 /tmp/$(basename $f) > $f
     rm -f /tmp/$(basename $f)
done
