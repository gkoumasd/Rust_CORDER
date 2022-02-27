find data -name "*.asm" | while read f; do
#   tree-sitter parse -q $f
    tree-grepper -q asm '(operand) @op' $f | cut -d: -f5
done | sort | uniq > vocab/treesitter_asm/node_token/token.txt
