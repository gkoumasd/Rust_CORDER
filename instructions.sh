#!/bin/bash
function process() {
     tmp_asm=$(mktemp).asm
     mv $1 $tmp_asm
     sed -i -e 's/^[ ]*//g' $tmp_asm
     tree-grepper -f json-lines -q asm '(instruction) @i' $tmp_asm | jq -r '.matches[].text' > $1
     rm -f $tmp_asm
}
export -f process

if [ "$1" == "test" ]; then
     file=$(find treesitter_asm_train_test_val/ -name "*.asm" | head -1)
     echo $file
     cat $file
     process $file
     exit 0
fi

parallel --bar process ::: $(find treesitter_asm_train_test_val/ -name "*.asm") 
