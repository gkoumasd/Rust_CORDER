find treesitter_asm_train_test_val/ -name "*.asm" | while read f; do
     mv $f /tmp/$(basename $f)
     tree-grepper -f json-lines -q asm '(instruction) @i' /tmp/$(basename $f) | jq -r '.matches[].text' > $f
     rm -f /tmp/$(basename $f)
done
