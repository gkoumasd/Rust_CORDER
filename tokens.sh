find treesitter_asm_train_test_val/ -name "*.asm" | while read f; do
#   tree-sitter parse -q $f
#    tree-grepper -q asm '(operand) @op' $f | cut -d: -f5
	~/Documents/github.com/yijunyu/tree-sitter-parsers/tree-sitter-asm-0.0.1/tokens.sh $f
done | sort | uniq > vocab/treesitter_asm/node_token/token.txt
